import io
import re
from pathlib import Path

import numpy as np
import pandas as pd


CANON = ["Date", "Value_Date", "Narration", "Reference", "Debit", "Credit", "Balance"]


def money(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).replace(",", "").replace("₹", "").replace("INR", "").strip()
    if s in ("", "-", "—", "nan", "None"):
        return np.nan
    neg = "-" in s or s.upper().endswith("DR")
    m = re.search(r"\d+(?:\.\d+)?", s)
    if not m:
        return np.nan
    v = float(m.group())
    return -v if neg else v


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def detect_header(raw):
    best = (-1, None)
    targets = [
        "post date", "transaction date", "value date", "date", "description",
        "narration", "particular", "remarks", "cheque no", "reference",
        "debit", "withdrawal", "credit", "deposit", "balance"
    ]
    for i in range(min(len(raw), 60)):
        vals = [norm(v) for v in raw.iloc[i].tolist()]
        score = sum(any(t in v for t in targets) for v in vals)
        if score > best[0]:
            best = (score, i)
    if best[0] < 3:
        raise ValueError("Could not confidently locate the transaction header row.")
    return best[1], best[0]


def pick(cols, patterns):
    best = (0, None)
    for c in cols:
        n = norm(c)
        score = max([len(p) for p in patterns if p in n] or [0])
        if score > best[0]:
            best = (score, c)
    return best[1]


def standardize(raw):
    raw = raw.dropna(how="all").dropna(axis=1, how="all").copy()
    if raw.empty:
        raise ValueError("Empty worksheet.")
    hi, score = detect_header(raw)
    df = raw.iloc[hi + 1:].copy()
    df.columns = [str(x).strip() for x in raw.iloc[hi].tolist()]
    df = df.dropna(how="all")
    cols = df.columns

    date = pick(cols, ["post date", "transaction date", "txn date", "date"])
    value = pick(cols, ["value date"])
    narr = pick(cols, ["description", "narration", "particular", "remarks", "details"])
    ref = pick(cols, ["cheque no", "reference", "ref no", "utr", "transaction id", "txn id"])
    debit = pick(cols, ["debit rs", "debit", "withdrawal", "withdraw"])
    credit = pick(cols, ["credit rs", "credit", "deposit"])
    bal = pick(cols, ["balance rs", "balance", "closing balance"])

    if not narr:
        raise ValueError("Narration/Description column not found.")

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df[date], errors="coerce", dayfirst=True) if date else pd.NaT
    out["Value_Date"] = pd.to_datetime(df[value], errors="coerce", dayfirst=True) if value else pd.NaT
    out["Narration"] = df[narr].fillna("").astype(str).str.strip()
    out["Reference"] = df[ref].fillna("").astype(str).str.strip() if ref else pd.Series("", index=df.index)
    out["Debit"] = df[debit].map(money).abs() if debit else np.nan
    out["Credit"] = df[credit].map(money).abs() if credit else np.nan
    out["Balance"] = df[bal].map(money) if bal else np.nan

    amount_ok = out[["Debit", "Credit"]].notna().any(axis=1)
    keep = amount_ok | out["Balance"].notna() | out["Date"].notna()
    out = out[keep].reset_index(drop=True)

    if len(out) == 0:
        raise ValueError("Header found, but no transaction rows with dates/amounts were detected.")
    return out[CANON], hi, score


def rail(t):
    u = str(t).upper()
    rules = [
        ("UPI", r"\bUPI\b|UPI/|BHIM|PHONEPE|GPAY|GOOGLEPAY|PAYTM|VPA"),
        ("IMPS", r"\bIMPS\b"),
        ("NEFT", r"\bNEFT\b"),
        ("RTGS", r"\bRTGS\b"),
        ("NACH/ECS/ACH", r"\bNACH\b|\bECS\b|\bACH\b|ACHDR|NACHDR"),
        ("ATM/CASH", r"\bATM\b|CASH WITHDRAWAL|CASH DEPOSIT"),
        ("CHEQUE", r"\bCHQ\b|\bCHEQUE\b|\bCTS\b|CAS PRES"),
        ("CARD", r"CREDIT CARD|DEBIT CARD|CARD PAYMENT|POS"),
        ("BROKING/INVESTMENT", r"BROKING|BROKER|ZERODHA|ANGEL ONE|ANGELONE|SECURITIES"),
        ("INSURANCE/PREMIUM", r"PREMIUM|INSURANCE|LIC"),
        ("LOAN/FINANCE", r"LOAN|HOUSINGFINA|PIRAMAL|EMI"),
        ("BANK CHARGES", r"BANK CHARGES|CHARGES|GST ON"),
        ("INTEREST", r"INTEREST"),
        ("RECHARGE/UTILITY", r"RECHARGE|JIO|VODAFONE|ELECTRIC"),
        ("TRANSFER", r"\bTFR\b|TRANSFER|TRF"),
    ]
    for name, pattern in rules:
        if re.search(pattern, u):
            return name
    return "OTHER / UNIDENTIFIED"


def category(t, d, c):
    u = str(t).upper()
    if "CASH DEPOSIT" in u:
        return "Cash Deposit"
    if "CASH WITHDRAWAL" in u:
        return "Cash Withdrawal"
    if "BANK CHARGES" in u:
        return "Bank Charges"
    if "INTEREST" in u:
        return "Interest"
    if "PREMIUM" in u or "LIC" in u:
        return "Insurance / Premium"
    if "LOAN" in u or "HOUSINGFINA" in u or "PIRAMAL" in u or "EMI" in u:
        return "Loan / Finance"
    if "BROKING" in u or "ANGEL ONE" in u or "ZERODHA" in u or "SECURITIES" in u:
        return "Investment / Broking"
    if "CREDIT CARD" in u:
        return "Credit Card"
    if "RECHARGE" in u or "JIO" in u or "VODAFONE" in u:
        return "Recharge / Utility"
    if "ONLINE" in u or "AMAZON" in u or "FLIPKART" in u or "SHOPSY" in u:
        return "Online / Merchant"
    if "SELF" in u:
        return "Self / Internal"
    if pd.notna(c) and c > 0 and (pd.isna(d) or d == 0):
        return "Other Credit / Receipt"
    if pd.notna(d) and d > 0 and (pd.isna(c) or c == 0):
        return "Other Debit / Expense"
    return "Other"


def counterparty(t):
    """Extract an explicitly present counterparty/name from narration.

    This is evidence extraction, not identity inference. For UPI, a readable
    name is returned only when it is explicitly present in the narration.
    IDs, URLs, VPA fragments and pure numeric tokens are never treated as names.
    """
    t = re.sub(r"\\s+", " ", str(t)).strip()
    if not t:
        return ""

    u = t.upper()
    if u in {"CASH DEPOSIT", "CASH WITHDRAWAL", "SELF", "BANK CHARGES", "INTEREST CREDIT"}:
        return ""

    # Remove common payment metadata while preserving readable name text.
    cleaned = re.sub(
        r"(?i)\\b(?:UPI|IMPS|NEFT|RTGS|NACH|ECS|ACH|CMS|POS|ATM|VPA)\\b[/_:\-]*",
        " ",
        t,
    )
    cleaned = re.sub(r"https?://\\S+", " ", cleaned)
    cleaned = re.sub(r"\\b\\d{5,}\\b", " ", cleaned)
    cleaned = re.sub(r"[/|:_\\-]+", " ", cleaned)
    cleaned = re.sub(r"\\s+", " ", cleaned).strip(" -")

    # Prefer 2+ alphabetic words. Keep initials and normal business-name words.
    candidates = re.findall(
        r"(?<![A-Za-z])([A-Za-z][A-Za-z.'&]{1,}(?:\\s+[A-Za-z][A-Za-z.'&]{1,}){1,5})(?![A-Za-z])",
        cleaned,
    )
    stop = {
        "BANK", "TRANSFER", "PAYMENT", "PAY", "COLLECT", "REQUEST",
        "MERCHANT", "TRANSACTION", "SUCCESS", "FAILED", "SENT", "RECEIVED",
        "DIGITAL", "MOBILE", "INTERNET", "BANKING", "CHARGES",
    }
    for c in candidates:
        words = c.split()
        useful = [w for w in words if w.upper() not in stop]
        if len(useful) >= 2 and any(len(w) >= 3 for w in useful):
            return " ".join(useful)[:160]

    # A single explicit alphabetic token can be a person's name, but only
    # when it is not a known rail/provider/metadata word.
    singles = re.findall(r"(?<![A-Za-z])([A-Za-z][A-Za-z.'&]{2,})(?![A-Za-z])", cleaned)
    for c in singles:
        if c.upper() not in stop and c.upper() not in {"UPI", "NEFT", "IMPS", "RTGS", "NACH"}:
            return c[:160]

    return ""


def enrich(df):
    x = df.copy()
    x["Payment_Rail"] = [rail(v) for v in x["Narration"]]
    x["Counterparty"] = x["Narration"].map(counterparty)
    x["Category"] = [category(t, d, c) for t, d, c in zip(x.Narration, x.Debit, x.Credit)]
    x["Abs_Amount"] = x[["Debit", "Credit"]].fillna(0).sum(axis=1)

    positive = x.loc[x.Abs_Amount > 0, "Abs_Amount"]
    q95 = positive.quantile(.95) if len(positive) else np.nan
    q99 = positive.quantile(.99) if len(positive) else np.nan

    reasons, priorities = [], []
    for _, r in x.iterrows():
        rs = []
        if pd.notna(q99) and r.Abs_Amount >= q99 and r.Abs_Amount > 0:
            rs.append("Top 1% by transaction amount")
        elif pd.notna(q95) and r.Abs_Amount >= q95 and r.Abs_Amount > 0:
            rs.append("Top 5% by transaction amount")
        if r.Payment_Rail == "OTHER / UNIDENTIFIED":
            rs.append("Payment rail not identified from narration")
        priority = "NORMAL"
        if rs:
            priority = "REVIEW"
        if r.Abs_Amount >= max(q99 if pd.notna(q99) else 0, 1000000):
            priority = "CRITICAL"
        reasons.append("; ".join(rs))
        priorities.append(priority)

    x["Flag_Reason"] = reasons
    x["Priority"] = priorities
    x["Rapid_Movement"] = False

    credits = x[x.Credit.fillna(0) > 0]
    debits = x[x.Debit.fillna(0) > 0]
    for ci, c in credits.iterrows():
        if c.Credit < 100000 or pd.isna(c.Date):
            continue
        cand = debits[(debits.Date >= c.Date) & (debits.Date <= c.Date + pd.Timedelta(days=1))]
        hits = cand[cand.Debit >= c.Credit * .8]
        if len(hits):
            x.loc[ci, "Rapid_Movement"] = True
            x.loc[hits.index, "Rapid_Movement"] = True

    mask = x.Rapid_Movement
    x.loc[mask, "Flag_Reason"] = (
        x.loc[mask, "Flag_Reason"].fillna("").astype(str).str.strip("; ")
        + "; Same/next-day large onward movement"
    ).str.strip("; ")
    x.loc[mask, "Priority"] = "CRITICAL"
    return x.drop(columns=["Abs_Amount"])


def analyze_excel(data, name):
    book = pd.ExcelFile(io.BytesIO(data))
    frames, details = [], []
    for sheet in book.sheet_names:
        raw = pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=None)
        try:
            d, hi, score = standardize(raw)
            d["Source_Sheet"] = sheet
            d["Source_Row"] = range(hi + 2, hi + 2 + len(d))
            frames.append(d)
            details.append((sheet, hi + 1, score, len(d)))
        except Exception:
            continue
    if not frames:
        raise ValueError("No sheet contained a confident transaction table.")
    df = enrich(pd.concat(frames, ignore_index=True))
    flags = df[df.Priority.isin(["REVIEW", "CRITICAL"])].copy()
    meta = {
        "source_type": "Excel/CSV",
        "location": f"{len(book.sheet_names)} sheet(s); {', '.join(s for s, _, _, _ in details)}",
        "layout_confidence": f"{max(x[2] for x in details)} matched header fields",
        "warnings": []
    }
    return {"transactions": df, "flags": flags, "meta": meta}


def _parse_table_rows(table, page_no):
    if not table:
        return pd.DataFrame()
    rows = [[("" if v is None else str(v).strip()) for v in row] for row in table if row]
    if not rows:
        return pd.DataFrame()

    header_idx = None
    best = -1
    for i, row in enumerate(rows[:8]):
        joined = " ".join(norm(v) for v in row)
        score = sum(k in joined for k in ["date", "description", "narration", "debit", "credit", "withdrawal", "deposit", "balance"])
        if score > best:
            best, header_idx = score, i
    if header_idx is None or best < 3:
        return pd.DataFrame()

    headers = rows[header_idx]
    data_rows = rows[header_idx + 1:]
    out = []
    for r in data_rows:
        if len(r) < len(headers):
            r = r + [""] * (len(headers) - len(r))
        if len(r) > len(headers):
            r = r[:len(headers)]
        out.append(r)
    raw = pd.DataFrame(out, columns=headers)
    try:
        d, _, _ = standardize(raw)
        d["Source_Page"] = page_no
        return d
    except Exception:
        return pd.DataFrame()


def _native_pdf_tables(data):
    import pdfplumber
    frames = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page_no, page in enumerate(pdf.pages, 1):
            settings_list = [
                {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                {"vertical_strategy": "text", "horizontal_strategy": "text", "snap_tolerance": 3, "join_tolerance": 3},
                {"vertical_strategy": "lines", "horizontal_strategy": "text", "snap_tolerance": 3}
            ]
            for settings in settings_list:
                try:
                    tables = page.extract_tables(table_settings=settings) or []
                except Exception:
                    tables = []
                for table in tables:
                    d = _parse_table_rows(table, page_no)
                    if len(d):
                        frames.append(d)
                if frames:
                    break
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _native_pdf_text_rows(data, progress_callback=None):
    import fitz
    doc = fitz.open(stream=data, filetype="pdf")
    records = []
    for page_no, page in enumerate(doc, 1):
        if progress_callback:
            progress_callback(page_no, len(doc), f"Extracting digital PDF page {page_no}/{len(doc)}")
        words = page.get_text("words")
        if not words:
            continue

        # Find a transaction header using word coordinates.
        header_words = [w for w in words if any(k in norm(w[4]) for k in ["date", "description", "narration", "particular", "debit", "credit", "balance", "withdrawal", "deposit"])]
        if not header_words:
            continue

        lines = {}
        for w in words:
            key = round(w[1] / 2) * 2
            lines.setdefault(key, []).append(w)
        for _, ws in sorted(lines.items()):
            ws = sorted(ws, key=lambda z: z[0])
            line = " ".join(w[4] for w in ws).strip()
            dm = re.match(r"^(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b", line)
            if not dm:
                dm = re.search(r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b", line)
            if not dm:
                continue

            date_text = dm.group(1).replace(".", "-").replace("/", "-")
            tail = line[dm.end():].strip()
            nums = re.findall(r"(?<![A-Za-z])(?:\(?\d[\d,]*(?:\.\d+)?\)?)(?:\s*(?:CR|DR))?", tail, flags=re.I)
            if not nums:
                continue

            # Only use this fallback when the page clearly exposes debit/credit/balance
            # headings. Amount columns are assigned by their horizontal positions.
            header_line = " ".join(w[4] for w in header_words).lower()
            if not any(k in header_line for k in ["debit", "credit", "balance"]):
                continue

            numeric = []
            for w in ws:
                if re.fullmatch(r"[\d,]+(?:\.\d+)?", w[4].replace("₹", "")):
                    numeric.append((w[0], money(w[4])))
            if not numeric:
                continue

            # Conservative positional mapping: use the three rightmost numeric values.
            numeric = numeric[-3:]
            vals = [v for _, v in numeric]
            while len(vals) < 3:
                vals.insert(0, np.nan)

            records.append({
                "Date": pd.to_datetime(date_text, dayfirst=True, errors="coerce"),
                "Value_Date": pd.NaT,
                "Narration": tail,
                "Reference": "",
                "Debit": np.nan,
                "Credit": np.nan,
                "Balance": vals[-1],
                "Source_Page": page_no,
            })
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records, columns=CANON + ["Source_Page"])



def _native_pdf_position_rows(data, progress_callback=None):
    """Extract native/digital bank PDFs using header x-positions and row-level column boundaries.

    Important: amounts are assigned from their actual PDF x-position.  We do not
    treat a number appearing inside narration as Debit/Credit merely because it
    looks like money.
    """
    import fitz

    doc = fitz.open(stream=data, filetype="pdf")
    frames = []

    date_re = re.compile(r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b")
    amount_re = re.compile(r"^(?:₹\s*)?\(?\d[\d,]*(?:\.\d+)?\)?(?:\s*(?:CR|DR))?$", re.I)

    def ykey(word):
        return round(word[1] / 2) * 2

    def center(word):
        return (word[0] + word[2]) / 2

    def clean_words(words):
        return sorted(words, key=lambda z: z[0])

    def header_x_positions(hws):
        """Find one reliable x-center for each bank-statement column."""
        hws = clean_words(hws)
        items = [(w, norm(w[4])) for w in hws]

        def phrase_x(phrases):
            for phrase in phrases:
                target = norm(phrase).split()
                for i in range(len(items) - len(target) + 1):
                    if [items[j][1] for j in range(i, i + len(target))] == target:
                        return (
                            center(items[i][0]) + center(items[i + len(target) - 1][0])
                        ) / 2
            return None

        def token_x(tokens):
            xs = [center(w) for w, n in items if any(t in n for t in tokens)]
            return min(xs) if xs else None

        return {
            "date": phrase_x(["post date", "transaction date", "txn date"])
                    or token_x(["date"]),
            "value": phrase_x(["value date"]) or token_x(["value"]),
            "narr": token_x(["description", "narration", "particular", "details", "remarks"]),
            # Bank statements commonly label this column as CHQ.NO. / CHQ NO.
            # It must be recognized explicitly; otherwise the cheque number can sit
            # just inside the Debit boundary and be falsely extracted as a monetary
            # withdrawal (the exact failure seen in the Bank of Baroda benchmark).
            "ref": phrase_x([
                "reference", "ref no", "cheque no", "chq no", "chq.no",
                "transaction id", "txn id", "utr"
            ]) or token_x(["reference", "ref", "cheque", "chq", "utr"]),
            "debit": token_x(["debit", "withdrawal", "withdraw"]),
            "credit": token_x(["credit", "deposit"]),
            "balance": phrase_x(["closing balance"]) or token_x(["balance"]),
        }

    def assign_column(x, centers):
        """Assign by midpoint boundaries, not nearest-center distance."""
        ordered = sorted(
            [(k, v) for k, v in centers.items() if v is not None],
            key=lambda kv: kv[1]
        )
        if not ordered:
            return None
        for i, (name, cx) in enumerate(ordered):
            left = -float("inf") if i == 0 else (ordered[i - 1][1] + cx) / 2
            right = float("inf") if i == len(ordered) - 1 else (cx + ordered[i + 1][1]) / 2
            if left <= x < right:
                return name
        return ordered[-1][0]

    def parse_amounts(words, centers):
        buckets = {k: [] for k in centers}
        for w in words:
            raw = str(w[4]).replace("₹", "").strip()
            if amount_re.fullmatch(raw):
                key = assign_column(center(w), centers)
                if key in buckets:
                    buckets[key].append((w[0], raw))

        def last_money(key):
            vals = buckets.get(key, [])
            for _, raw in reversed(vals):
                value = money(raw)
                if pd.notna(value):
                    return abs(value)
            return np.nan

        return last_money("debit"), last_money("credit"), last_money("balance")

    for page_no, page in enumerate(doc, 1):
        words = page.get_text("words")
        if not words:
            continue

        lines = {}
        for w in words:
            lines.setdefault(ykey(w), []).append(w)

        # Locate the transaction header on this page.
        header = None
        header_score = -1
        for y, ws in lines.items():
            text_line = " ".join(w[4] for w in clean_words(ws))
            n = norm(text_line)
            score = sum(
                term in n
                for term in [
                    "date", "value date", "description", "narration",
                    "particular", "reference", "debit", "credit",
                    "withdrawal", "deposit", "balance"
                ]
            )
            if score > header_score and score >= 3:
                header_score = score
                header = (y, ws)

        if header is None:
            continue

        centers = header_x_positions(header[1])
        if centers["date"] is None or centers["balance"] is None:
            continue
        if centers["debit"] is None and centers["credit"] is None:
            # A statement without explicit amount columns is not safe to map.
            continue

        for y, ws in sorted(lines.items()):
            if y <= header[0] + 3:
                continue

            ws = clean_words(ws)
            line_text = " ".join(w[4] for w in ws).strip()
            if not line_text or not date_re.search(line_text):
                continue

            # Ignore repeated column headers / footer lines.
            if sum(term in norm(line_text) for term in ["date", "narration", "description", "debit", "credit", "balance"]) >= 3:
                continue

            buckets = {k: [] for k in centers}
            for w in ws:
                key = assign_column(center(w), centers)
                if key:
                    buckets[key].append(w[4])

            date_text = " ".join(buckets.get("date", []))
            dm = date_re.search(date_text) or date_re.search(line_text)
            if not dm:
                continue

            date_value = pd.to_datetime(
                dm.group(0).replace(".", "-").replace("/", "-"),
                dayfirst=True,
                errors="coerce",
            )
            if pd.isna(date_value):
                continue

            def amount_from_bucket(key):
                values = buckets.get(key, [])
                for raw in reversed(values):
                    value = money(raw)
                    if pd.notna(value):
                        return abs(value)
                return np.nan

            debit_value = amount_from_bucket("debit")
            credit_value = amount_from_bucket("credit")
            balance_value = amount_from_bucket("balance")

            narr_value = " ".join(buckets.get("narr", [])).strip()
            ref_value = " ".join(buckets.get("ref", [])).strip()

            value_date = pd.NaT
            value_text = " ".join(buckets.get("value", []))
            vm = date_re.search(value_text)
            if vm:
                value_date = pd.to_datetime(
                    vm.group(0).replace(".", "-").replace("/", "-"),
                    dayfirst=True,
                    errors="coerce",
                )

            if not narr_value:
                narr_value = line_text

            # Some banks place CR/DR beside the transaction amount.
            upper_line = line_text.upper()
            if pd.isna(debit_value) and pd.isna(credit_value):
                non_balance_numbers = []
                for w in ws:
                    raw = str(w[4]).replace("₹", "").strip()
                    if amount_re.fullmatch(raw):
                        x = center(w)
                        key = assign_column(x, centers)
                        if key not in {"balance", "date", "value"}:
                            z = money(raw)
                            if pd.notna(z):
                                non_balance_numbers.append((x, abs(z), raw))
                if len(non_balance_numbers) == 1:
                    amt = non_balance_numbers[0][1]
                    if re.search(r"\bDR\b", upper_line):
                        debit_value = amt
                    elif re.search(r"\bCR\b", upper_line):
                        credit_value = amt

            frames.append(
                pd.DataFrame(
                    [{
                        "Date": date_value,
                        "Value_Date": value_date,
                        "Narration": narr_value,
                        "Reference": ref_value,
                        "Debit": debit_value,
                        "Credit": credit_value,
                        "Balance": balance_value,
                        "Source_Page": page_no,
                    }]
                )
            )

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)

    # Remove exact repeated rows caused by a header/layout repeat.
    result = result.drop_duplicates(
        subset=["Date", "Narration", "Debit", "Credit", "Balance", "Source_Page"],
        keep="first",
    ).reset_index(drop=True)

    return result



def _ocr_pdf_position_rows(data, progress_callback=None):
    """OCR scanned bank PDFs into conservative transaction rows.

    The OCR path reconstructs the table from page coordinates. It does not
    invent amounts from narration. Pages without a confident header/amount
    layout are skipped and reported through Source_Page.
    """
    import fitz
    import pytesseract
    from PIL import Image

    doc = fitz.open(stream=data, filetype="pdf")
    all_rows = []

    date_re = re.compile(r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b")
    amount_re = re.compile(r"^\(?\d[\d,]*(?:\.\d+)?\)?(?:\s*(?:CR|DR))?$", re.I)

    def group_lines(words):
        lines = {}
        for w in words:
            y = round(float(w["top"]) / 3) * 3
            lines.setdefault(y, []).append(w)
        return {k: sorted(v, key=lambda z: z["left"]) for k, v in lines.items()}

    def find_header(lines):
        best = None
        score_best = -1
        for y, ws in lines.items():
            text = " ".join(w["text"] for w in ws)
            n = norm(text)
            score = sum(
                term in n for term in [
                    "date", "description", "narration", "particular",
                    "debit", "credit", "withdrawal", "deposit", "balance"
                ]
            )
            if score >= 3 and score > score_best:
                score_best = score
                best = (y, ws)
        return best

    def centers_from_header(ws):
        items = [(w["text"], w["left"] + w["width"] / 2) for w in ws]
        def token(tokens):
            hits = [x for text, x in items if any(t in norm(text) for t in tokens)]
            return min(hits) if hits else None
        return {
            "date": token(["date"]),
            "value": token(["value"]),
            "narr": token(["description", "narration", "particular", "details", "remarks"]),
            "ref": token(["reference", "ref", "cheque", "utr"]),
            "debit": token(["debit", "withdrawal", "withdraw"]),
            "credit": token(["credit", "deposit"]),
            "balance": token(["balance"]),
        }

    def assign(x, centers):
        ordered = sorted([(k, v) for k, v in centers.items() if v is not None], key=lambda z: z[1])
        for i, (name, cx) in enumerate(ordered):
            left = -1e9 if i == 0 else (ordered[i-1][1] + cx) / 2
            right = 1e9 if i == len(ordered)-1 else (cx + ordered[i+1][1]) / 2
            if left <= x < right:
                return name
        return None

    for page_no, page in enumerate(doc, 1):
        if progress_callback:
            progress_callback(page_no, len(doc), f"OCR scanning page {page_no}/{len(doc)}")
        pix = page.get_pixmap(matrix=fitz.Matrix(1.15, 1.15), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        data_dict = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            config="--psm 6",
        )
        words = []
        for i, text_value in enumerate(data_dict["text"]):
            text_value = str(text_value).strip()
            if not text_value:
                continue
            try:
                conf = float(data_dict["conf"][i])
            except Exception:
                conf = -1
            if conf < 20:
                continue
            words.append({
                "text": text_value,
                "left": float(data_dict["left"][i]),
                "top": float(data_dict["top"][i]),
                "width": float(data_dict["width"][i]),
            })

        if not words:
            continue

        lines = group_lines(words)
        header = find_header(lines)
        if header is None:
            continue

        header_y, header_words = header
        centers = centers_from_header(header_words)
        if centers["date"] is None or centers["balance"] is None:
            continue
        if centers["debit"] is None and centers["credit"] is None:
            continue

        for y, ws in sorted(lines.items()):
            if y <= header_y + 4:
                continue
            line = " ".join(w["text"] for w in ws)
            dm = date_re.search(line)
            if not dm:
                continue
            nline = norm(line)
            if sum(t in nline for t in ["date", "narration", "description", "debit", "credit", "balance"]) >= 3:
                continue

            buckets = {k: [] for k in centers}
            for w in ws:
                k = assign(w["left"] + w["width"]/2, centers)
                if k:
                    buckets[k].append(w["text"])

            date_text = " ".join(buckets.get("date", []))
            dm2 = date_re.search(date_text) or dm
            dt = pd.to_datetime(dm2.group(0).replace(".", "-").replace("/", "-"), dayfirst=True, errors="coerce")
            if pd.isna(dt):
                continue

            def amt(key):
                for raw in reversed(buckets.get(key, [])):
                    if amount_re.fullmatch(raw.replace("₹", "").strip()):
                        value = money(raw)
                        if pd.notna(value):
                            return abs(value)
                return np.nan

            debit_value = amt("debit")
            credit_value = amt("credit")
            balance_value = amt("balance")

            # Require an actual amount in debit/credit or a balance. Never use
            # a number embedded in narration as a transaction amount.
            if pd.isna(debit_value) and pd.isna(credit_value) and pd.isna(balance_value):
                continue

            narr = " ".join(buckets.get("narr", [])).strip()
            ref = " ".join(buckets.get("ref", [])).strip()
            if not narr:
                # Preserve the non-date text as evidence if column OCR missed it.
                narr = re.sub(re.escape(dm2.group(0)), "", line, count=1).strip()

            all_rows.append({
                "Date": dt,
                "Value_Date": pd.NaT,
                "Narration": narr,
                "Reference": ref,
                "Debit": debit_value,
                "Credit": credit_value,
                "Balance": balance_value,
                "Source_Page": page_no,
            })

    if not all_rows:
        return pd.DataFrame()

    result = pd.DataFrame(all_rows, columns=CANON + ["Source_Page"])
    result = result.drop_duplicates(
        subset=["Date", "Narration", "Debit", "Credit", "Balance", "Source_Page"],
        keep="first",
    ).reset_index(drop=True)
    return result


def analyze_pdf(data, name, progress_callback=None):
    import fitz

    doc = fitz.open(stream=data, filetype="pdf")
    pages = len(doc)
    texts = [p.get_text("text") for p in doc]
    nonempty = sum(bool(t.strip()) for t in texts)
    ratio = nonempty / pages if pages else 0

    if progress_callback:
        progress_callback(0, max(pages, 1), "Detecting PDF type and layout")

    if ratio < 0.5:
        # Scanned/image PDF: use the real OCR reconstruction path.
        df = _ocr_pdf_position_rows(data, progress_callback=progress_callback)
        extraction_method = "OCR table reconstruction"
        if df.empty:
            raise ValueError(
                f"{pages}-page scanned PDF was detected, but OCR could not reconstruct "
                "a reliable transaction table. No values were invented."
            )
    else:
        # Digital/native PDF: use coordinate-aware extraction first.
        df = _native_pdf_position_rows(data, progress_callback=progress_callback)
        extraction_method = "Native PDF column-position extraction"
        if df.empty:
            df = _native_pdf_tables(data)
            extraction_method = "Native PDF table extraction"
        if df.empty:
            raise ValueError(
                "Digital PDF detected, but the transaction columns could not be mapped reliably. "
                "No rows were invented."
            )

    if progress_callback:
        progress_callback(pages, max(pages, 1), "Validating extracted transactions")

    df = df[df["Date"].notna()].copy()
    if df.empty:
        raise ValueError("PDF extraction produced no reliable dated transaction rows.")

    # HARD EVIDENCE BOUNDARY:
    # Bank PDFs often carry a later print/generation date in the footer
    # (this statement was printed on 17-03-2026), while the actual transaction
    # period is explicitly 01-01-2022 to 30-06-2023. Never allow a footer/header
    # date to become a transaction date.
    statement_period = re.search(
        r"period of\\s+(\\d{1,2}[-/.]\\d{1,2}[-/.]\\d{2,4})\\s+to\\s+(\\d{1,2}[-/.]\\d{1,2}[-/.]\\d{2,4})",
        "\\n".join(texts),
        flags=re.I,
    )
    period_start = period_end = None
    if statement_period:
        period_start = pd.to_datetime(statement_period.group(1).replace(".", "-").replace("/", "-"), dayfirst=True, errors="coerce")
        period_end = pd.to_datetime(statement_period.group(2).replace(".", "-").replace("/", "-"), dayfirst=True, errors="coerce")
        if pd.notna(period_start) and pd.notna(period_end):
            outside = (df["Date"] < period_start) | (df["Date"] > period_end)
            outside_count = int(outside.sum())
            df = df.loc[~outside].copy()
        else:
            outside_count = 0
    else:
        outside_count = 0

    if df.empty:
        raise ValueError("PDF extraction produced no transactions inside the statement's declared period.")

    movement_presence = df[["Debit", "Credit"]].notna().any(axis=1).mean()
    balance_presence = df["Balance"].notna().mean()

    if movement_presence < 0.60:
        raise ValueError(
            f"PDF extraction confidence is too low ({movement_presence:.0%} rows contain Debit/Credit; "
            f"{balance_presence:.0%} contain Balance). The transaction amounts could not be "
            "mapped reliably, so the analysis was stopped rather than guessing."
        )

    if "Source_Page" not in df:
        df["Source_Page"] = np.nan

    if progress_callback:
        progress_callback(pages, max(pages, 1), "Running forensic classification and validation")

    df = enrich(df)
    flags = df[df.Priority.isin(["REVIEW", "CRITICAL"])].copy()

    meta = {
        "source_type": "PDF — scanned/OCR" if ratio < 0.5 else "PDF — native/digital",
        "location": f"{pages} pages",
        "layout_confidence": extraction_method,
        "warnings": [
            f"Extraction method: {extraction_method}.",
            "Debit/Credit values are accepted only from detected transaction columns.",
            "Source_Page is retained for evidence tracing.",
            "Blank amount fields are treated as unknown, not zero.",
            f"Statement-period guard removed {outside_count} extracted row(s) outside the declared transaction period." if outside_count else "All extracted transaction dates fall within the statement's declared period.",
        ],
    }

    return {"transactions": df, "flags": flags, "meta": meta}

def analyze_upload(data, name, progress_callback=None):
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        return analyze_pdf(data, name, progress_callback=progress_callback)
    if ext in {".xlsx", ".xls", ".csv"}:
        return analyze_excel(data, name)
    raise ValueError("Unsupported file type. Upload XLSX, XLS, CSV, or PDF.")




def balance_check(df, tolerance=0.01):
    """Reconcile reported balance against previous balance + credit - debit.

    Blank Debit/Credit values remain unknown. Such rows are marked
    INCOMPLETE rather than treating blanks as zero. This prevents extraction
    gaps from creating artificial red balance errors.
    """
    x = df.copy().reset_index(drop=True)

    for c in ["Debit", "Credit", "Balance"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    rows = []
    previous_balance = np.nan

    for i, r in x.iterrows():
        reported = r["Balance"]
        debit = r["Debit"]
        credit = r["Credit"]

        if pd.isna(reported):
            status = "NO REPORTED BALANCE"
            expected = np.nan
            difference = np.nan
        elif pd.isna(previous_balance):
            # First usable balance establishes the opening reference.
            expected = reported
            difference = 0.0
            status = "OPENING / REFERENCE"
        elif pd.isna(debit) and pd.isna(credit):
            # Both movement columns are blank: the source does not provide a
            # usable movement amount for this row, so reconciliation is incomplete.
            expected = np.nan
            difference = np.nan
            status = "INCOMPLETE — DEBIT/CREDIT UNKNOWN"
        else:
            # Bank statements normally leave the opposite side blank:
            # a credit row has Debit blank, and a debit row has Credit blank.
            # For reconciliation only, that structural blank is zero movement.
            debit_value = 0.0 if pd.isna(debit) else float(debit)
            credit_value = 0.0 if pd.isna(credit) else float(credit)
            expected = previous_balance + credit_value - debit_value
            difference = reported - expected
            status = "MATCH" if abs(difference) <= tolerance else "MISMATCH"

        rows.append({
            "Source_Row": r.get("Source_Row", i + 1),
            "Source_Page": r.get("Source_Page", ""),
            "Date": r.get("Date", pd.NaT),
            "Debit": debit,
            "Credit": credit,
            "Reported_Balance": reported,
            "Expected_Balance": expected,
            "Difference": difference,
            "Status": status,
        })

        if pd.notna(reported):
            previous_balance = reported

    return pd.DataFrame(rows)


def balance_mismatches(df, tolerance=0.01):
    check = balance_check(df, tolerance=tolerance)
    if check.empty:
        return 0
    return int((check["Status"] == "MISMATCH").sum())


def build_master_analysis(df):
    """Evidence-led master analytics. Flags are review priorities, never fraud conclusions."""
    x = df.copy()
    debit = x["Debit"].fillna(0)
    credit = x["Credit"].fillna(0)
    amount = debit + credit

    positive = amount[amount > 0]
    q95 = positive.quantile(.95) if len(positive) else np.nan
    q99 = positive.quantile(.99) if len(positive) else np.nan

    round_mask = (amount >= 50000) & (amount.mod(10000).eq(0))
    high_mask = amount >= (q99 if pd.notna(q99) else np.inf)
    cash_mask = x["Payment_Rail"].eq("ATM/CASH")
    unknown_mask = x["Payment_Rail"].eq("OTHER / UNIDENTIFIED")

    day_counts = x.groupby(x["Date"].dt.date)["Narration"].transform("size")
    velocity_mask = day_counts >= 15

    dup_key = (
        x["Date"].astype(str) + "|" +
        x["Narration"].fillna("").astype(str) + "|" +
        debit.round(2).astype(str) + "|" +
        credit.round(2).astype(str)
    )
    duplicate_mask = dup_key.duplicated(keep=False) & x["Date"].notna()

    master_rows = [
        ["Transaction volume", len(x), "Entire evidence set", "Factual scope", "INFORMATIONAL"],
        ["Period covered", f"{x['Date'].min().strftime('%d-%m-%Y') if x['Date'].notna().any() else '-'} to {x['Date'].max().strftime('%d-%m-%Y') if x['Date'].notna().any() else '-'}", "Transaction dates", "Coverage", "INFORMATIONAL"],
        ["Total credits", credit.sum(), "Credit column", "Funds received", "INFORMATIONAL"],
        ["Total debits", debit.sum(), "Debit column", "Funds paid/out", "INFORMATIONAL"],
        ["Net flow", credit.sum() - debit.sum(), "Credits minus debits", "Aggregate flow", "INFORMATIONAL"],
        ["Top 1% amount transactions", int(high_mask.sum()), f"Threshold ₹{q99:,.2f}" if pd.notna(q99) else "Unavailable", "High-value review", "REVIEW"],
        ["Top 5% amount transactions", int((amount >= q95).sum()) if pd.notna(q95) else 0, f"Threshold ₹{q95:,.2f}" if pd.notna(q95) else "Unavailable", "High-value review", "REVIEW"],
        ["Rapid same/next-day onward movements", int(x["Rapid_Movement"].sum()), "Credit followed by >=80% debit within 1 day", "Fund-flow review", "REVIEW"],
        ["Large round-value transactions", int(round_mask.sum()), ">= ₹50,000 and multiple of ₹10,000", "Pattern review", "REVIEW"],
        ["High transaction-velocity rows", int(velocity_mask.sum()), "15+ transactions on same date", "Activity-pattern review", "REVIEW"],
        ["Potential duplicate rows", int(duplicate_mask.sum()), "Same date + narration + debit + credit", "Data-quality review", "REVIEW"],
        ["Cash/ATM activity", int(cash_mask.sum()), "Payment rail classification", "Cash-flow review", "REVIEW"],
        ["Unidentified payment rail", int(unknown_mask.sum()), "Narration did not expose a known rail", "Data-quality only; not suspicious by itself", "DATA QUALITY"],
        ["Balance mismatches", balance_mismatches(x), "Sequential balance reconciliation", "Evidence integrity", "CRITICAL" if balance_mismatches(x) else "PASS"],
    ]
    master = pd.DataFrame(master_rows, columns=["Finding", "Value", "Basis", "Interpretation", "Review_Status"])

    # Fund-flow review: large credits followed by substantial debits within 24 hours.
    ff = []
    credits = x[credit > 0]
    debits = x[debit > 0]
    for ci, cr in credits.iterrows():
        if cr["Credit"] < 100000 or pd.isna(cr["Date"]):
            continue
        candidates = debits[
            (debits["Date"] >= cr["Date"]) &
            (debits["Date"] <= cr["Date"] + pd.Timedelta(days=1)) &
            (debits["Debit"] >= cr["Credit"] * 0.8)
        ]
        for di, dr in candidates.iterrows():
            ff.append({
                "Credit_Date": cr["Date"], "Credit_Amount": cr["Credit"],
                "Credit_Narration": cr["Narration"], "Credit_Source_Row": cr.get("Source_Row", ""),
                "Debit_Date": dr["Date"], "Debit_Amount": dr["Debit"],
                "Debit_Narration": dr["Narration"], "Debit_Source_Row": dr.get("Source_Row", ""),
                "Movement_Ratio": round(dr["Debit"] / cr["Credit"], 4) if cr["Credit"] else np.nan,
                "Review_Reason": "Large credit followed by substantial debit within 1 day"
            })
    fund_flow = pd.DataFrame(ff)

    # Counterparty concentration.
    cp = x[x["Counterparty"].fillna("").ne("")].copy()
    if len(cp):
        concentration = cp.groupby("Counterparty").agg(
            Transactions=("Narration", "size"),
            Credits=("Credit", "sum"),
            Debits=("Debit", "sum"),
            First_Date=("Date", "min"),
            Last_Date=("Date", "max")
        ).reset_index()
        total_credit = concentration["Credits"].fillna(0).sum()
        total_debit = concentration["Debits"].fillna(0).sum()
        concentration["Credit_Share"] = concentration["Credits"].fillna(0) / total_credit if total_credit else 0
        concentration["Debit_Share"] = concentration["Debits"].fillna(0) / total_debit if total_debit else 0
        concentration["Net_Flow"] = concentration["Credits"].fillna(0) - concentration["Debits"].fillna(0)
        concentration = concentration.sort_values(["Credits", "Debits"], ascending=False).reset_index(drop=True)
    else:
        concentration = pd.DataFrame(columns=["Counterparty","Transactions","Credits","Debits","First_Date","Last_Date","Credit_Share","Debit_Share","Net_Flow"])

    # Data quality and provenance.
    dq = pd.DataFrame([
        ["Rows extracted", len(x), "Total normalized transactions"],
        ["Missing transaction date", int(x["Date"].isna().sum()), "Date not extracted"],
        ["Missing both debit and credit", int((debit.eq(0) & credit.eq(0)).sum()), "No monetary movement recorded"],
        ["Missing reported balance", int(x["Balance"].isna().sum()), "Balance unavailable in source"],
        ["Unidentified rail", int(unknown_mask.sum()), "Classification limitation; not a fraud finding"],
        ["Potential duplicate rows", int(duplicate_mask.sum()), "Requires source verification"],
        ["Balance mismatches", balance_mismatches(x), "Sequential reconciliation"],
    ], columns=["Data_Quality_Item","Count","Meaning"])

    provenance_cols = [c for c in ["Source_Sheet","Source_Page","Source_Row"] if c in x.columns]
    if provenance_cols:
        provenance = x.groupby(provenance_cols, dropna=False).size().reset_index(name="Extracted_Transactions")
    else:
        provenance = pd.DataFrame()

    return master, fund_flow, concentration, dq, provenance


def _format_workbook(wb):
    from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
    from openpyxl.utils import get_column_letter
    thin=Side(style="thin",color="B7B7B7")
    border=Border(left=thin,right=thin,top=thin,bottom=thin)
    section_names={"FUND-FLOW REVIEW","MONTHLY FLOW","DATA QUALITY","BALANCE RECONCILIATION"}
    for ws in wb.worksheets:
        ws.freeze_panes=None
        ws.auto_filter.ref=ws.dimensions
        for row in ws.iter_rows():
            for cell in row:
                cell.font=Font(name="Bookman Old Style",size=10,bold=False)
                cell.alignment=Alignment(vertical="top",wrap_text=False)
                cell.border=border
                cell.fill=PatternFill(fill_type=None)
        for cell in ws[1]:
            cell.font=Font(name="Bookman Old Style",size=10,bold=True)
            cell.alignment=Alignment(horizontal="center",vertical="center",wrap_text=False)
            cell.border=border
            cell.fill=PatternFill(fill_type=None)
        headers={str(x.value):x.column for x in ws[1] if x.value is not None}
        for h,col_idx in headers.items():
            hn=norm(h)
            if "date" in hn:
                for cells in ws.iter_cols(min_col=col_idx,max_col=col_idx,min_row=2):
                    for x in cells:
                        if x.value is not None: x.number_format="dd-mm-yyyy"
            if hn in {"debit","credit","balance","expected","reported","difference","credits","debits","net flow"} or "amount" in hn:
                for cells in ws.iter_cols(min_col=col_idx,max_col=col_idx,min_row=2):
                    for x in cells:
                        if isinstance(x.value,(int,float,np.integer,np.floating)) and not pd.isna(x.value):
                            x.number_format='#,##0.00;[Red]-#,##0.00;-'
            if any(k in hn for k in ["narration","description","remarks","reason","category","rail","reference"]):
                for cells in ws.iter_cols(min_col=col_idx,max_col=col_idx,min_row=2):
                    for x in cells:
                        if x.value is None or (isinstance(x.value,str) and not x.value.strip()): x.value="-"
        for col_cells in ws.columns:
            vals=[str(x.value) if x.value is not None else "" for x in col_cells[:250]]
            ws.column_dimensions[get_column_letter(col_cells[0].column)].width=min(max(max([len(v) for v in vals]+[10])+2,12),70)
        for row in ws.iter_rows():
            if row and str(row[0].value or "") in section_names:
                for cell in row:
                    cell.font=Font(name="Bookman Old Style",size=10,bold=True)
                    cell.fill=PatternFill(fill_type=None)
                    cell.border=border

def build_workbook(df, flags, meta):
    out=io.BytesIO()
    export_df=df.copy()
    # Source_Page is retained internally for evidence tracing, but it is not
    # part of the user's requested transaction export.
    if "Source_Page" in export_df.columns:
        export_df=export_df.drop(columns=["Source_Page"])
    credits=export_df["Credit"].fillna(0); debits=export_df["Debit"].fillna(0)
    master,fund_flow,concentration,dq,_=build_master_analysis(df)
    summary=pd.DataFrame({"Metric":["Source","Period","Transactions","Total Credits","Total Debits","Net Flow","Review","Critical","Balance Mismatches"],"Value":[meta.get("source_type","-"),meta.get("location","-"),len(export_df),credits.sum(),debits.sum(),credits.sum()-debits.sum(),int((export_df.Priority=="REVIEW").sum()),int((export_df.Priority=="CRITICAL").sum()),balance_mismatches(df)]})
    rail=df.groupby("Payment_Rail",dropna=False).agg(Transactions=("Narration","size"),Credits=("Credit","sum"),Debits=("Debit","sum")).reset_index().rename(columns={"Payment_Rail":"Analysis"}); rail.insert(0,"Section","Payment Rail")
    cat=df.groupby("Category",dropna=False).agg(Transactions=("Narration","size"),Credits=("Credit","sum"),Debits=("Debit","sum")).reset_index().rename(columns={"Category":"Analysis"}); cat.insert(0,"Section","Category")
    flow=pd.concat([rail,cat],ignore_index=True)
    with pd.ExcelWriter(out,engine="openpyxl") as w:
        summary.to_excel(w,index=False,sheet_name="01_Executive_Summary")
        export_df.to_excel(w,index=False,sheet_name="02_Transactions")
        flow.to_excel(w,index=False,sheet_name="03_Flow_Analysis")
        master.to_excel(w,index=False,sheet_name="04_Master_Analysis")
        row=len(master)+3
        pd.DataFrame({"Section":["FUND-FLOW REVIEW"]}).to_excel(w,index=False,sheet_name="04_Master_Analysis",startrow=row-1)
        if fund_flow.empty:
            pd.DataFrame({"Result":["No configured large-credit / onward-debit pattern identified."]}).to_excel(w,index=False,sheet_name="04_Master_Analysis",startrow=row); row+=4
        else:
            fund_flow.head(60).to_excel(w,index=False,sheet_name="04_Master_Analysis",startrow=row); row+=min(len(fund_flow),60)+3
        monthly=df.copy(); monthly["Month"]=monthly["Date"].dt.to_period("M").astype(str)
        monthly=monthly.groupby("Month",dropna=False).agg(Transactions=("Narration","size"),Credits=("Credit","sum"),Debits=("Debit","sum")).reset_index()
        pd.DataFrame({"Section":["MONTHLY FLOW"]}).to_excel(w,index=False,sheet_name="04_Master_Analysis",startrow=row-1)
        monthly.to_excel(w,index=False,sheet_name="04_Master_Analysis",startrow=row); row+=len(monthly)+3
        pd.DataFrame({"Section":["DATA QUALITY"]}).to_excel(w,index=False,sheet_name="04_Master_Analysis",startrow=row-1)
        dq.to_excel(w,index=False,sheet_name="04_Master_Analysis",startrow=row); row+=len(dq)+3
        pd.DataFrame({"Section":["BALANCE RECONCILIATION"]}).to_excel(w,index=False,sheet_name="04_Master_Analysis",startrow=row-1)
        balance_check(df).to_excel(w,index=False,sheet_name="04_Master_Analysis",startrow=row)
        _format_workbook(w.book)
    out.seek(0); return out.getvalue()


def build_pdf_report(df, flags, meta, filename):
    """Create a professional, evidence-led forensic PDF report.

    The report is intentionally conservative: amounts are shown as plain numeric
    values (no currency glyphs that may render as squares), dates are formatted
    consistently, and the report clearly distinguishes extracted facts from
    review priorities.
    """
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    )

    master, fund_flow, concentration, dq, _ = build_master_analysis(df)

    def fmt_amount(value):
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value):,.2f}"

    def fmt_date(value):
        if value is None or pd.isna(value):
            return "-"
        try:
            return pd.Timestamp(value).strftime("%d-%m-%Y")
        except Exception:
            return "-"

    credits = pd.to_numeric(df["Credit"], errors="coerce").fillna(0)
    debits = pd.to_numeric(df["Debit"], errors="coerce").fillna(0)
    net = credits.sum() - debits.sum()

    # Keep the report factual. Do not present a future/garbled date as the
    # evidence period if the parser produced one; use the normalized dates.
    valid_dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
    if len(valid_dates):
        period_text = f"{valid_dates.min().strftime('%d-%m-%Y')} to {valid_dates.max().strftime('%d-%m-%Y')}"
    else:
        period_text = "-"

    movement_presence = df[["Debit", "Credit"]].notna().any(axis=1).mean() if len(df) else 0
    balance_presence = df["Balance"].notna().mean() if len(df) else 0
    mismatch_count = balance_mismatches(df)

    rails = df.groupby("Payment_Rail", dropna=False).agg(
        Transactions=("Narration", "size"),
        Credits=("Credit", "sum"),
        Debits=("Debit", "sum")
    ).reset_index().sort_values("Transactions", ascending=False)

    cats = df.groupby("Category", dropna=False).agg(
        Transactions=("Narration", "size"),
        Credits=("Credit", "sum"),
        Debits=("Debit", "sum")
    ).reset_index().sort_values("Transactions", ascending=False)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        rightMargin=12*mm,
        leftMargin=12*mm,
        topMargin=11*mm,
        bottomMargin=11*mm,
        title="FORENSIC INTELLIGENCE - Executive Forensic Analysis Report",
        author="FORENSIC INTELLIGENCE",
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "FITitle", parent=styles["Title"], alignment=TA_CENTER,
        fontName="Helvetica-Bold", fontSize=19, leading=22, spaceAfter=7
    )
    subtitle = ParagraphStyle(
        "FISubtitle", parent=styles["Heading2"], alignment=TA_LEFT,
        fontName="Helvetica-Bold", fontSize=12, leading=15, spaceAfter=5
    )
    section = ParagraphStyle(
        "FISection", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=12, leading=15, spaceBefore=5, spaceAfter=6
    )
    body = ParagraphStyle(
        "FIBody", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=8.5, leading=11, spaceAfter=3
    )
    cell = ParagraphStyle(
        "FICell", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=7.2, leading=8.8, alignment=TA_LEFT
    )
    cell_bold = ParagraphStyle(
        "FICellBold", parent=cell, fontName="Helvetica-Bold"
    )

    story = [
        Paragraph("FORENSIC INTELLIGENCE", title),
        Paragraph("Executive Forensic Analysis Report", subtitle),
        Paragraph(f"<b>Evidence:</b> {filename}", body),
        Paragraph(
            f"<b>Source:</b> {meta.get('source_type','-')} &nbsp;&nbsp; "
            f"<b>Pages:</b> {meta.get('location','-')} &nbsp;&nbsp; "
            f"<b>Evidence Period:</b> {period_text}",
            body
        ),
        Spacer(1, 4),
    ]

    # Executive summary.
    summary_data = [
        ["Metric", "Result"],
        ["Transactions", f"{len(df):,}"],
        ["Total Credits", fmt_amount(credits.sum())],
        ["Total Debits", fmt_amount(debits.sum())],
        ["Net Flow", fmt_amount(net)],
        ["Review Priorities", f"{int((df.Priority == 'REVIEW').sum()):,}"],
        ["Critical Priorities", f"{int((df.Priority == 'CRITICAL').sum()):,}"],
        ["Debit/Credit Recovery", f"{movement_presence:.1%} of rows"],
        ["Balance Recovery", f"{balance_presence:.1%} of rows"],
        ["Balance Reconciliation", f"{mismatch_count:,} mismatch(es)"],
    ]

    summary_table = Table(summary_data, colWidths=[55*mm, 48*mm], repeatRows=1)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17324d")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#9a9a9a")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))

    story += [
        summary_table,
        Spacer(1, 7),
        Paragraph("Extraction & Evidence Quality", section),
    ]

    quality_status = "PASS" if movement_presence >= 0.60 and mismatch_count == 0 else "REVIEW REQUIRED"
    quality_text = (
        f"<b>Status:</b> {quality_status}<br/>"
        f"Debit/Credit amounts recovered on {movement_presence:.1%} of transaction rows. "
        f"Balance values recovered on {balance_presence:.1%} of rows. "
        f"Sequential balance reconciliation reports {mismatch_count:,} mismatch(es)."
    )
    if movement_presence < 0.60:
        quality_text += (
            "<br/><b>Important:</b> Debit/Credit extraction confidence is below the "
            "required threshold. The source statement must be re-checked before relying "
            "on monetary conclusions."
        )
    if mismatch_count:
        quality_text += (
            "<br/><b>Important:</b> Balance mismatches are an evidence-integrity review "
            "item and must not be treated as proof of irregularity."
        )

    story += [Paragraph(quality_text, body), PageBreak()]

    # Key findings table with clean numeric formatting.
    story += [Paragraph("Key Review Findings", section)]
    findings = [[
        Paragraph("Finding", cell_bold),
        Paragraph("Value", cell_bold),
        Paragraph("Basis", cell_bold),
        Paragraph("Interpretation", cell_bold),
        Paragraph("Status", cell_bold),
    ]]
    for _, r in master.iterrows():
        value = r["Value"]
        finding = str(r["Finding"])
        if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
            if any(k in finding.lower() for k in ["credit", "debit", "flow", "amount"]):
                value_text = fmt_amount(value)
            else:
                value_text = f"{int(value):,}" if float(value).is_integer() else f"{float(value):,.2f}"
        else:
            value_text = str(value)
        findings.append([
            Paragraph(finding, cell),
            Paragraph(value_text, cell),
            Paragraph(str(r["Basis"]), cell),
            Paragraph(str(r["Interpretation"]), cell),
            Paragraph(str(r["Review_Status"]), cell),
        ])

    findings_table = Table(
        findings,
        repeatRows=1,
        colWidths=[47*mm, 34*mm, 58*mm, 78*mm, 28*mm]
    )
    findings_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17324d")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#9a9a9a")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    story += [findings_table, PageBreak()]

    def profile_table(title_text, table_df):
        story_local = [Paragraph(title_text, section)]
        data = [[
            Paragraph(str(c), cell_bold) for c in table_df.columns
        ]]
        for _, r in table_df.head(20).iterrows():
            row = []
            for c in table_df.columns:
                value = r[c]
                if c in {"Credits", "Debits"}:
                    txt = fmt_amount(value)
                elif c == "Transactions":
                    txt = f"{int(value):,}" if pd.notna(value) else "-"
                else:
                    txt = "-" if pd.isna(value) else str(value)
                row.append(Paragraph(txt, cell))
            data.append(row)

        widths = [63*mm, 31*mm, 42*mm, 42*mm]
        tb = Table(data, repeatRows=1, colWidths=widths)
        tb.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17324d")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#9a9a9a")),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ]))
        story_local.append(tb)
        return story_local

    story += profile_table("Payment Rail Profile", rails)
    story += [Spacer(1, 8)]
    story += profile_table("Category Profile", cats)
    story += [PageBreak(), Paragraph("Fund-Flow Review", section)]

    if fund_flow.empty:
        story.append(Paragraph(
            "No configured large-credit / onward-debit pattern identified under the current review rule.",
            body
        ))
    else:
        ff = fund_flow.head(40).copy()
        cols = [
            "Credit_Date", "Credit_Amount", "Credit_Narration",
            "Debit_Date", "Debit_Amount", "Debit_Narration",
            "Movement_Ratio", "Review_Reason"
        ]
        ff = ff[[c for c in cols if c in ff.columns]]
        data = [[Paragraph(str(c), cell_bold) for c in ff.columns]]
        for _, r in ff.iterrows():
            row = []
            for c in ff.columns:
                v = r[c]
                if c.endswith("_Date"):
                    txt = fmt_date(v)
                elif "Amount" in c:
                    txt = fmt_amount(v)
                elif c == "Movement_Ratio":
                    txt = f"{float(v):.2%}" if pd.notna(v) else "-"
                else:
                    txt = "-" if pd.isna(v) or str(v).strip() == "" else str(v)
                row.append(Paragraph(txt, cell))
            data.append(row)

        ff_widths = [23*mm, 28*mm, 58*mm, 23*mm, 28*mm, 58*mm, 25*mm, 48*mm]
        tb = Table(data, repeatRows=1, colWidths=ff_widths)
        tb.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17324d")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#9a9a9a")),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("FONTSIZE", (0,0), (-1,-1), 6.5),
        ]))
        story.append(tb)

    story += [
        Spacer(1, 9),
        Paragraph("Conclusion & Use of Report", section),
        Paragraph(
            "This report presents extracted facts, analytical observations and review priorities. "
            "A REVIEW or CRITICAL status is not a finding of fraud, illegality, intent or guilt. "
            "All material observations should be verified against the original bank statement and "
            "supporting evidence before being used in an investigation, audit or legal proceeding.",
            body
        ),
        Paragraph(
            f"<b>Generated by:</b> FORENSIC INTELLIGENCE &nbsp;&nbsp; "
            f"<b>Extraction method:</b> {meta.get('layout_confidence','-')}",
            body
        ),
    ]

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


