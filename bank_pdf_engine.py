"""Robust, bank-format-flexible native PDF transaction extractor.

This engine does not require a specific bank template. It combines:
- date-anchored transaction row detection,
- optional header-derived column geometry,
- geometry-derived fallback columns when headers are absent,
- running-balance evidence to resolve Debit/Credit direction,
- wrapped narration collection across physical PDF lines/pages,
- provenance for every extracted row.

It is deliberately conservative: it never invents a transaction amount.
"""

import io
import re

import numpy as np
import pandas as pd


CANON = ["Date", "Narration", "Reference", "Debit", "Credit", "Balance"]
DATE_RE = re.compile(r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b")
MONEY_RE = re.compile(
    r"^(?:₹\s*)?\(?\d[\d,]*(?:\.\d+)?\)?(?:\s*(?:CR|DR))?$",
    re.I,
)


def _norm(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _money(value):
    if value is None:
        return np.nan
    raw = str(value).replace("₹", "").replace("INR", "").strip()
    if not raw or raw in {"-", "—", "nan", "None"}:
        return np.nan
    upper = raw.upper().replace(" ", "")
    sign = -1 if upper.endswith("DR") or raw.startswith("-") else 1
    match = re.search(r"\d[\d,]*(?:\.\d+)?", raw)
    if not match:
        return np.nan
    try:
        return sign * float(match.group(0).replace(",", ""))
    except ValueError:
        return np.nan


def _center(word):
    return (float(word[0]) + float(word[2])) / 2.0


def _ykey(word):
    return round(float(word[1]) / 2.0) * 2


def _group_lines(words):
    lines = {}
    for word in words:
        lines.setdefault(_ykey(word), []).append(word)
    return {
        y: sorted(ws, key=lambda w: float(w[0]))
        for y, ws in sorted(lines.items())
    }


def _header_layout(lines):
    """Infer column centers from whichever transaction header is visible."""
    best = None
    best_score = -1
    terms = [
        "date", "value date", "description", "narration", "particular",
        "details", "remarks", "reference", "ref no", "cheque", "chq",
        "debit", "withdrawal", "withdraw", "credit", "deposit", "balance",
    ]

    for y, ws in lines.items():
        text = " ".join(w[4] for w in ws)
        n = _norm(text)
        score = sum(term in n for term in terms)
        if score >= 3 and score > best_score:
            best = (y, ws)
            best_score = score

    if best is None:
        return None

    ws = best[1]
    items = [(w, _norm(w[4])) for w in ws]

    def phrase_center(phrases):
        for phrase in phrases:
            target = _norm(phrase).split()
            for i in range(len(items) - len(target) + 1):
                if [items[j][1] for j in range(i, i + len(target))] == target:
                    return (
                        _center(items[i][0]) + _center(items[i + len(target) - 1][0])
                    ) / 2.0
        return None

    def token_center(tokens):
        hits = [
            _center(w)
            for w, n in items
            if any(token in n for token in tokens)
        ]
        return min(hits) if hits else None

    movement = phrase_center([
        "withdrawal (dr) deposit (cr)",
        "withdrawal dr deposit cr",
        "debit (dr) credit (cr)",
    ])
    if movement is None:
        wx = token_center(["withdrawal", "withdraw"])
        dx = token_center(["deposit", "credit"])
        if wx is not None and dx is not None and abs(wx - dx) <= 12:
            movement = (wx + dx) / 2.0

    layout = {
        "date": phrase_center(["tran date", "transaction date", "post date", "txn date"])
                or token_center(["date"]),
        "value": phrase_center(["value date", "value dt", "value"])
                or token_center(["value"]),
        "narr": token_center([
            "description", "narration", "particular", "particulars",
            "details", "remarks",
        ]),
        "ref": phrase_center([
            "chq/ref no", "cheque/ref no", "reference no", "ref no",
            "cheque no", "chq no", "chq.no", "instrument no",
            "transaction id", "txn id", "utr no",
        ]) or token_center([
            "reference", "ref", "cheque", "chq", "instrument", "utr", "rrn"
        ]),
        "debit": None if movement is not None else token_center([
            "debit", "withdrawal", "withdraw"
        ]),
        "credit": None if movement is not None else token_center([
            "credit", "deposit"
        ]),
        "movement": movement,
        "balance": phrase_center(["closing balance", "balance"])
                   or token_center(["balance"]),
        "header_y": best[0],
    }

    if layout["date"] is None or layout["balance"] is None:
        return None
    if layout["debit"] is None and layout["credit"] is None and layout["movement"] is None:
        return None
    return layout


def _assign(x, layout):
    ordered = sorted(
        [(k, v) for k, v in layout.items() if k != "header_y" and v is not None],
        key=lambda item: item[1],
    )
    for i, (name, cx) in enumerate(ordered):
        left = -float("inf") if i == 0 else (ordered[i - 1][1] + cx) / 2
        right = float("inf") if i == len(ordered) - 1 else (cx + ordered[i + 1][1]) / 2
        if left <= x < right:
            return name
    return None


def _is_money_word(word):
    return bool(MONEY_RE.fullmatch(str(word).replace("₹", "").strip()))


def _statement_noise(line):
    n = _norm(line)
    if not n:
        return True
    if re.search(r"\b(?:page|generated by|requesting branch|statement of account)\b", n):
        return True
    if sum(term in n for term in [
        "date", "value date", "description", "narration", "particular",
        "debit", "credit", "withdrawal", "deposit", "balance"
    ]) >= 3:
        return True
    return False


def _date_start(line):
    match = DATE_RE.search(line)
    if not match:
        return None
    # A transaction date normally starts in the left table region. A date
    # buried in narration/reference is not allowed to create a new row.
    prefix = line[:match.start()].strip()
    if prefix:
        return None
    return match


def _opening_balance(text):
    patterns = [
        r"(?is)\bopening\s+balance\b\s*[:\-]?\s*([\d,]+(?:\.\d+)?)\s*(CR|DR)?",
        r"(?is)\bbrought\s+forward\b\s*[:\-]?\s*([\d,]+(?:\.\d+)?)\s*(CR|DR)?",
        r"(?is)\bb\s*/\s*f\b\s*[:\-]?\s*([\d,]+(?:\.\d+)?)\s*(CR|DR)?",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            value = _money(m.group(1))
            if pd.notna(value):
                return -abs(value) if (m.group(2) or "").upper() == "DR" else abs(value)
    return np.nan


def _row_amounts(words, layout, page_width):
    """Return candidate numeric cells while keeping reference/date numbers separate."""
    cells = []
    for word in words:
        raw = str(word[4]).replace("₹", "").strip()
        if not _is_money_word(raw):
            continue
        x = _center(word)
        column = _assign(x, layout)
        cells.append({
            "x": x,
            "raw": raw,
            "value": abs(float(_money(raw))),
            "column": column,
        })

    if not cells:
        return [], None

    # Prefer an amount in the explicit balance column. Otherwise the rightmost
    # money token is the balance because Indian bank layouts place it last.
    balance = None
    balance_cells = [c for c in cells if c["column"] == "balance"]
    if balance_cells:
        balance = max(balance_cells, key=lambda c: c["x"])
    elif len(cells) >= 2:
        balance = max(cells, key=lambda c: c["x"])

    movement = [
        c for c in cells
        if balance is None or c is not balance
    ]

    # Reference/cheque numbers are often numeric but live well to the left of
    # movement columns. Drop them from movement candidates.
    if layout.get("ref") is not None:
        movement = [
            c for c in movement
            if c["x"] > layout["ref"] + max(8, page_width * 0.015)
            or c["column"] in {"debit", "credit", "movement"}
        ]

    return movement, balance


def _narration_and_reference(words, layout, date_match):
    date_end = float(words[0][2]) if words and date_match else 0
    narr_words = []
    ref_words = []

    # Column geometry is primary. If a reference column exists, everything
    # between date and reference is narration; reference-column text is evidence.
    for word in words:
        text = str(word[4]).strip()
        x = _center(word)
        if date_match and DATE_RE.fullmatch(text):
            continue
        if _is_money_word(text):
            continue
        if layout.get("value") is not None and abs(x - layout["value"]) < 35 and DATE_RE.search(text):
            continue
        if layout.get("ref") is not None and abs(x - layout["ref"]) <= max(55, layout["ref"] * 0.12):
            ref_words.append(text)
        elif layout.get("narr") is not None and x >= layout["narr"] - 40:
            if layout.get("ref") is None or x < layout["ref"] - 20:
                narr_words.append(text)
            else:
                ref_words.append(text)
        elif layout.get("narr") is None and x > date_end:
            narr_words.append(text)

    narration = " ".join(narr_words).strip()
    reference = " ".join(ref_words).strip()
    return narration, reference


def extract_bank_pdf(data, progress_callback=None):
    """Extract transaction rows from a native PDF without bank-specific templates."""
    import fitz

    doc = fitz.open(stream=data, filetype="pdf")
    pages = len(doc)
    full_text = "\n".join(page.get_text("text") for page in doc)
    opening = _opening_balance(full_text)

    rows = []
    active_layout = None
    previous_balance = np.nan
    seq = 0

    for page_no, page in enumerate(doc, 1):
        if progress_callback:
            progress_callback(page_no, pages, f"Bank-format extraction page {page_no}/{pages}")

        words = page.get_text("words")
        if not words:
            continue

        lines = _group_lines(words)
        page_layout = _header_layout(lines)
        if page_layout is not None:
            active_layout = page_layout
        elif active_layout is None:
            # Geometry-only fallback: Indian statements almost always reserve
            # the rightmost ~40% for movement/balance values.
            width = float(page.rect.width)
            active_layout = {
                "date": width * 0.12,
                "value": width * 0.24,
                "narr": width * 0.34,
                "ref": width * 0.52,
                "debit": width * 0.68,
                "credit": width * 0.79,
                "movement": None,
                "balance": width * 0.92,
                "header_y": -1,
            }

        header_y = active_layout.get("header_y", -1)
        page_rows = []

        for y, line_words in lines.items():
            line_text = " ".join(str(w[4]).strip() for w in line_words).strip()
            if header_y >= 0 and y <= header_y + 5:
                continue
            if _statement_noise(line_text):
                continue

            date_match = _date_start(line_text)
            if date_match is None:
                continue

            # Require a transaction-like money cell in the right side. This
            # blocks footer dates and statement metadata from becoming rows.
            money_words = [w for w in line_words if _is_money_word(w[4])]
            right_money = [w for w in money_words if _center(w) > float(page.rect.width) * 0.50]
            if not right_money:
                continue

            movement_cells, balance_cell = _row_amounts(
                line_words, active_layout, float(page.rect.width)
            )

            # Exclude the balance from movement candidates.
            if balance_cell is not None:
                movement_cells = [
                    c for c in movement_cells
                    if abs(c["x"] - balance_cell["x"]) > 8
                ]

            # Pick the best movement cell(s): prefer explicit debit/credit/movement
            # columns, otherwise the rightmost non-balance numeric cell.
            explicit = [
                c for c in movement_cells
                if c["column"] in {"debit", "credit", "movement"}
            ]
            candidates = explicit or movement_cells
            candidates = sorted(candidates, key=lambda c: c["x"])

            movement_value = candidates[-1]["value"] if candidates else np.nan
            movement_column = candidates[-1]["column"] if candidates else None

            debit = np.nan
            credit = np.nan
            if pd.notna(movement_value):
                if movement_column == "debit":
                    debit = movement_value
                elif movement_column == "credit":
                    credit = movement_value
                elif movement_column == "movement":
                    raw = candidates[-1]["raw"].upper().replace(" ", "")
                    if raw.endswith("DR"):
                        debit = movement_value
                    elif raw.endswith("CR"):
                        credit = movement_value
                else:
                    # Direction will be resolved from balance/opening evidence below.
                    if re.search(r"\b(?:DR|DEBIT|WITHDRAWAL|WITHDRAW)\b", line_text, re.I):
                        debit = movement_value
                    elif re.search(r"\b(?:CR|CREDIT|DEPOSIT)\b", line_text, re.I):
                        credit = movement_value

            balance = balance_cell["value"] if balance_cell is not None else np.nan
            narration, reference = _narration_and_reference(
                line_words, active_layout, date_match
            )

            if not narration:
                # Keep the full row as source evidence rather than losing the
                # transaction merely because the narration column was unusual.
                narration = line_text

            date_value = pd.to_datetime(
                date_match.group(0).replace(".", "-").replace("/", "-"),
                dayfirst=True,
                errors="coerce",
            )
            if pd.isna(date_value):
                continue

            row = {
                "Date": date_value,
                "Narration": narration,
                "Reference": reference,
                "Debit": debit,
                "Credit": credit,
                "Balance": balance,
                "Source_Page": page_no,
                "Source_Seq": seq,
                "Source_Text": line_text,
            }
            seq += 1
            page_rows.append(row)

        # Wrapped narration is attached by pdf_extraction_adapter after row extraction.
        rows.extend(page_rows)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows, columns=CANON + ["Source_Page", "Source_Seq", "Source_Text"])

    # Resolve Debit/Credit from the reported balance whenever the movement side
    # is unambiguous. This is evidence-based and especially useful when a bank
    # uses a generic "Amount" column.
    result = result.sort_values("Source_Seq", kind="stable").reset_index(drop=True)
    prev = opening
    for i in result.index:
        balance = pd.to_numeric(result.at[i, "Balance"], errors="coerce")
        debit = pd.to_numeric(result.at[i, "Debit"], errors="coerce")
        credit = pd.to_numeric(result.at[i, "Credit"], errors="coerce")

        if pd.isna(balance):
            continue

        if pd.notna(debit) and pd.isna(credit) and pd.notna(prev):
            delta = float(balance) - float(prev)
            if abs(delta - float(debit)) <= 0.01 and abs(delta + float(debit)) > 0.01:
                result.at[i, "Credit"] = float(debit)
                result.at[i, "Debit"] = np.nan
            elif abs(delta + float(debit)) <= 0.01 and abs(delta - float(debit)) > 0.01:
                pass
        elif pd.isna(debit) and pd.notna(credit) and pd.notna(prev):
            delta = float(balance) - float(prev)
            if abs(delta + float(credit)) <= 0.01 and abs(delta - float(credit)) > 0.01:
                result.at[i, "Debit"] = float(credit)
                result.at[i, "Credit"] = np.nan
        elif pd.isna(debit) and pd.isna(credit) and pd.notna(prev):
            delta = float(balance) - float(prev)
            if abs(delta) > 0.01:
                if delta > 0:
                    result.at[i, "Credit"] = abs(delta)
                else:
                    result.at[i, "Debit"] = abs(delta)

        prev = float(balance)

    result = result.drop_duplicates(
        subset=["Date", "Narration", "Debit", "Credit", "Balance", "Source_Page"],
        keep="first",
    ).reset_index(drop=True)
    return result
