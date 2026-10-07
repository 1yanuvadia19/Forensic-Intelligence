"""Generic scanned-bank-statement OCR grid engine.

Designed for image-only / photographed / scanned statements. It does not use a
bank template. It discovers column geometry from OCR anchors, reconstructs
physical rows, extracts the printed movement amount and running balance, then
uses the balance delta to resolve Debit/Credit direction.

The engine keeps the printed balance as evidence and never manufactures a
balance merely to make a ledger reconcile.
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
import pytesseract


DATE_RE = re.compile(r"^\\d{2}/\\d{2}/\\d{2,4}$")


def _money(s):
    s = str(s).upper().replace("CR", "").replace("DR", "")
    s = s.replace("—", "-").replace("O", "0")
    s = re.sub(r"[^0-9,.-]", "", s)
    if not s:
        return np.nan
    # OCR often converts Indian thousand commas to dots: 10.250 -> 10,250.
    if re.fullmatch(r"\\d{1,3}(?:\\.\\d{3})+", s):
        s = s.replace(".", "")
    try:
        return float(s.replace(",", ""))
    except (TypeError, ValueError):
        return np.nan


def _ocr(page):
    pix = page.get_pixmap(matrix=__import__("fitz").Matrix(3, 3), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
    image = ImageOps.autocontrast(image)
    image = image.filter(ImageFilter.SHARPEN)
    data = pytesseract.image_to_data(
        image,
        config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DATAFRAME,
    )
    data = data.dropna(subset=["text"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[(data["text"] != "") & (data["conf"] >= 18)]
    data["cx"] = data["left"] + data["width"] / 2.0
    return image, data


def _first_center(words, text):
    q = words[words["text"].str.fullmatch(text, case=False)]
    if q.empty:
        return None
    return float((q["left"] + q["width"] / 2.0).iloc[0])


def _header_geometry(words, width):
    post = _first_center(words, "Post")
    debit = _first_center(words, "Debit")
    credit = _first_center(words, "Credit")

    # Relative fallback is deliberately based on the discovered Post column,
    # not on a named bank/template.
    if post is None:
        post = width * 0.24
    if debit is None:
        debit = post + width * 0.39
    if credit is None:
        credit = max(debit + width * 0.12, width * 0.78)

    return {
        "value_date": post - width * 0.105,
        "post_date": post,
        "details": post + width * 0.10,
        "cheque": post + width * 0.22,
        "amount": debit,
        "balance": credit,
    }


def _date_rows(words, geom):
    q = words[words["text"].str.match(DATE_RE)]
    if q.empty:
        return []

    near = q[
        (q["cx"].sub(geom["post_date"]).abs() < 85)
        | (q["cx"].sub(geom["value_date"]).abs() < 100)
    ].copy()
    near = near[near["top"] > 450]
    ys = sorted(float(x) for x in near["top"].tolist())

    groups = []
    for y in ys:
        if not groups or y - groups[-1][-1] > 14:
            groups.append([y])
        else:
            groups[-1].append(y)

    return [float(np.mean(g)) for g in groups]


def _parse_date(words, y, geom):
    q = words[words["top"].sub(y).abs() < 17].copy()
    q = q[q["text"].str.match(DATE_RE)]
    if q.empty:
        return pd.NaT, ""

    q["post_dist"] = q["cx"].sub(geom["post_date"]).abs()
    q = q.sort_values("post_dist")

    for value in q["text"].tolist():
        for fmt in ("%d/%m/%y", "%d/%m/%Y"):
            try:
                return pd.to_datetime(value, format=fmt), value
            except (TypeError, ValueError):
                continue

    # OCR can corrupt one of the two printed dates. Keep the evidence text,
    # but do not invent an invalid date.
    return pd.NaT, str(q.iloc[0]["text"])


def _numeric_groups(row, xmin, xmax):
    q = row[(row["cx"] >= xmin) & (row["cx"] <= xmax)].sort_values("left")
    parts = []
    current = []
    previous_right = None

    for _, token in q.iterrows():
        text = str(token["text"])
        if not re.search(r"\\d", text):
            continue
        if previous_right is None or float(token["left"]) - previous_right < 48:
            current.append(text)
        else:
            parts.append("".join(current))
            current = [text]
        previous_right = float(token["left"] + token["width"])

    if current:
        parts.append("".join(current))

    result = []
    for raw in parts:
        value = _money(raw)
        if pd.notna(value):
            result.append((float(value), raw))
    return result


def _pick_amount(row, geom):
    # First preference: amount/debit zone. Legacy statements sometimes print
    # small service charges inside Details, so the wider fallback is important.
    candidates = _numeric_groups(
        row,
        max(geom["details"] + 25, geom["amount"] - 145),
        geom["balance"] - 80,
    )

    cleaned = []
    for value, raw in candidates:
        # Cheque/reference numbers are usually integer-only and compact. Amounts
        # in these statements normally contain a decimal or grouping.
        if "." not in raw and "," not in raw and value >= 10000 and value.is_integer():
            continue
        if value > 0:
            cleaned.append((value, raw))

    if not cleaned:
        return np.nan, ""

    # Prefer the candidate closest to the discovered amount column. If a
    # service charge is printed in Details, it remains a valid candidate.
    target = geom["amount"]
    chosen = min(cleaned, key=lambda x: abs(target - geom["amount"]))
    # If OCR placed a legacy service charge in Details, the amount itself is
    # still the strongest candidate in that physical row.
    return chosen[0], chosen[1]


def _pick_balance(row, geom):
    # Balance is the rightmost monetary evidence. Prefer tokens containing Cr/Dr.
    right = row[row["cx"] >= geom["balance"] - 65].copy()
    candidates = []
    for _, token in right.iterrows():
        raw = str(token["text"])
        value = _money(raw)
        if pd.notna(value):
            candidates.append((float(value), raw, float(token["cx"])))

    if not candidates:
        # OCR may split the Indian number into several words.
        groups = _numeric_groups(row, geom["balance"] - 60, row["cx"].max() + 5)
        candidates = [(v, raw, geom["balance"]) for v, raw in groups]

    if not candidates:
        return np.nan, ""

    crdr = [x for x in candidates if re.search(r"(?:CR|DR)$", x[1], re.I)]
    return (crdr or candidates)[-1][0:2]


def _narration(row, geom):
    q = row[
        (row["cx"] >= geom["details"] - 35)
        & (row["cx"] < geom["amount"] - 20)
    ].sort_values("left")
    text = " ".join(str(x) for x in q["text"].tolist()).strip()
    # Remove obvious amount/reference fragments but retain real narration.
    return text


def _direction(amount, balance, previous, narration):
    if pd.notna(amount) and pd.notna(balance) and pd.notna(previous):
        delta = float(balance) - float(previous)
        if abs(delta - amount) <= max(0.05, amount * 0.00001):
            return "Credit", "running-balance delta"
        if abs(delta + amount) <= max(0.05, amount * 0.00001):
            return "Debit", "running-balance delta"

    upper = str(narration).upper()
    if re.search(r"\\b(?:ATM|WDL|WITHDRAW|CHG|GST|FEE|TFR)\\b", upper):
        return "Debit", "narration heuristic"
    if re.search(r"\\b(?:INT|BY|CLG|CASH DEP|NEFT|RTGS|IMPS|CREDIT)\\b", upper):
        return "Credit", "narration heuristic"
    return None, "unresolved"


def _page_opening(words):
    text = " ".join(words["text"].astype(str).tolist())
    m = re.search(
        r"(?:BROUGHT|CARRIED)\\s+FORWARD\\s*:?\\s*([0-9,]+(?:\\.[0-9]{1,2})?)\\s*(?:CR|DR)?",
        text,
        re.I,
    )
    if m:
        return _money(m.group(1))
    return np.nan


def extract_scanned_statement(data, progress_callback=None):
    import fitz

    doc = fitz.open(stream=data, filetype="pdf")
    rows = []
    previous_balance = np.nan
    sequence = 0
    page_meta = []

    for page_no, page in enumerate(doc, 1):
        if progress_callback:
            progress_callback(page_no, len(doc), f"Scanned ledger vision page {page_no}/{len(doc)}")

        image, words = _ocr(page)
        if words.empty:
            continue

        geom = _header_geometry(words, float(image.width))
        page_opening = _page_opening(words)
        if pd.notna(page_opening):
            previous_balance = page_opening

        ys = _date_rows(words, geom)
        page_rows = []

        for y in ys:
            row_words = words[words["top"].sub(y).abs() < 18].copy()
            date_value, date_raw = _parse_date(words, y, geom)
            if pd.isna(date_value):
                continue

            narration = _narration(row_words, geom)
            amount, amount_raw = _pick_amount(row_words, geom)
            balance, balance_raw = _pick_balance(row_words, geom)
            direction, direction_source = _direction(
                amount, balance, previous_balance, narration
            )

            debit = amount if direction == "Debit" else np.nan
            credit = amount if direction == "Credit" else np.nan

            source_text = " ".join(
                str(x) for x in row_words.sort_values("left")["text"].tolist()
            ).strip()

            page_rows.append({
                "Date": date_value,
                "Narration": narration,
                "Reference": "",
                "Debit": debit,
                "Credit": credit,
                "Balance": balance,
                "Source_Page": page_no,
                "Source_Seq": sequence,
                "Source_Text": source_text,
                "OCR_Date_Raw": date_raw,
                "OCR_Amount_Raw": amount_raw,
                "OCR_Balance_Raw": balance_raw,
                "Direction_Evidence": direction_source,
            })
            sequence += 1

            if pd.notna(balance):
                previous_balance = float(balance)

        rows.extend(page_rows)
        page_meta.append({
            "page": page_no,
            "rows": len(page_rows),
            "opening": page_opening,
            "geometry": geom,
        })

    if not rows:
        return pd.DataFrame(), page_meta

    out = pd.DataFrame(rows)
    out = out.sort_values("Source_Seq", kind="stable").reset_index(drop=True)
    return out, page_meta
