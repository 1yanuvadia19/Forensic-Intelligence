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
    t = str(t).strip()
    if not t or t.upper() in {"CASH DEPOSIT", "CASH WITHDRAWAL", "SELF", "BANK CHARGES", "INTEREST CREDIT"}:
        return ""
    return t[:160]


def enrich(df):
    x = df.copy()
    x["Counterparty"] = x["Narration"].map(counterparty)
    x["Payment_Rail"] = [rail(v) for v in x["Narration"]]
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


def _native_pdf_text_rows(data):
    import fitz
    doc = fitz.open(stream=data, filetype="pdf")
    records = []
    for page_no, page in enumerate(doc, 1):
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



def _native_pdf_position_rows(data):
    import fitz
    doc = fitz.open(stream=data, filetype="pdf")
    frames = []

    def ykey(w):
        return round(w[1] / 2) * 2

    for page_no, page in enumerate(doc, 1):
        words = page.get_text("words")
        if not words:
            continue

        lines = {}
        for w in words:
            lines.setdefault(ykey(w), []).append(w)

        header = None
        header_score = -1
        for y, ws in lines.items():
            text = " ".join(w[4] for w in sorted(ws, key=lambda z: z[0]))
            n = norm(text)
            score = sum(k in n for k in [
                "date", "value date", "description", "narration",
                "particular", "reference", "debit", "credit",
                "withdrawal", "deposit", "balance"
            ])
            if score > header_score and score >= 3:
                header_score = score
                header = (y, ws)

        if header is None:
            continue

        hws = sorted(header[1], key=lambda z: z[0])
        htext = " ".join(w[4] for w in hws)
        hn = norm(htext)

        def center(w):
            return (w[0] + w[2]) / 2

        def find_x(patterns):
            candidates = []
            for w in hws:
                n = norm(w[4])
                if any(p in n for p in patterns):
                    candidates.append(center(w))
            return min(candidates) if candidates else None

        x_date = find_x(["post date", "transaction date", "txn date", "date"])
        x_value = find_x(["value date"])
        x_narr = find_x(["description", "narration", "particular", "details", "remarks"])
        x_ref = find_x(["reference", "ref no", "cheque no", "utr", "txn id", "transaction id"])
        x_debit = find_x(["debit", "withdrawal", "withdraw"])
        x_credit = find_x(["credit", "deposit"])
        x_balance = find_x(["balance", "closing balance"])

        if x_date is None or x_narr is None or x_balance is None:
            continue

        columns = [
            ("date", x_date), ("value", x_value), ("narr", x_narr),
            ("ref", x_ref), ("debit", x_debit), ("credit", x_credit),
            ("balance", x_balance)
        ]
        columns = [(k, x) for k, x in columns if x is not None]

        def nearest_column(x):
            return min(columns, key=lambda item: abs(item[1] - x))[0]

        for y, ws in sorted(lines.items()):
            if y <= header[0] + 2:
                continue
            ws = sorted(ws, key=lambda z: z[0])
            line_text = " ".join(w[4] for w in ws).strip()
            if not line_text:
                continue

            dm = re.search(r"\\b(\\d{1,2}[-/.]\\d{1,2}[-/.]\\d{2,4})\\b", line_text)
            if not dm:
                continue

            buckets = {k: [] for k, _ in columns}
            for w in ws:
                key = nearest_column(center(w))
                buckets[key].append(w[4])

            date_text = " ".join(buckets.get("date", []))
            date_match = re.search(r"\\d{1,2}[-/.]\\d{1,2}[-/.]\\d{2,4}", date_text)
            if not date_match:
                date_match = dm
            date_value = pd.to_datetime(
                date_match.group(0).replace(".", "-").replace("/", "-"),
                dayfirst=True, errors="coerce"
            )
            if pd.isna(date_value):
                continue

            def amount_from(key):
                vals = buckets.get(key, [])
                for v in reversed(vals):
                    z = money(v)
                    if pd.notna(z):
                        return abs(z)
                return np.nan

            debit_value = amount_from("debit")
            credit_value = amount_from("credit")
            balance_value = amount_from("balance")

            narr_value = " ".join(buckets.get("narr", [])).strip()
            ref_value = " ".join(buckets.get("ref", [])).strip()
            value_date = pd.NaT
            if x_value is not None:
                vt = " ".join(buckets.get("value", []))
                vm = re.search(r"\\d{1,2}[-/.]\\d{1,2}[-/.]\\d{2,4}", vt)
                if vm:
                    value_date = pd.to_datetime(vm.group(0).replace(".", "-").replace("/", "-"), dayfirst=True, errors="coerce")

            if not narr_value:
                narr_value = line_text

            if pd.isna(debit_value) and pd.isna(credit_value):
                # Some statements use a single amount column with CR/DR suffix.
                upper = line_text.upper()
                amt = amount_from("balance")
                if amt and re.search(r"\\bDR\\b", upper):
                    debit_value = amt
                elif amt and re.search(r"\\bCR\\b", upper):
                    credit_value = amt

            frames.append(pd.DataFrame([{
                "Date": date_value,
                "Value_Date": value_date,
                "Narration": narr_value,
                "Reference": ref_value,
                "Debit": debit_value,
                "Credit": credit_value,
                "Balance": balance_value,
                "Source_Page": page_no
            }]))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def analyze_pdf(data, name):
    import fitz
    doc = fitz.open(stream=data, filetype="pdf")
    pages = len(doc)
    texts = [p.get_text("text") for p in doc]
    nonempty = sum(bool(t.strip()) for t in texts)
    ratio = nonempty / pages if pages else 0

    if ratio < 0.5:
        raise ValueError(
            f"{pages}-page PDF appears scanned/image-based ({nonempty} pages contain extractable text). "
            "This version accepts native/digital PDFs; scanned PDFs require OCR before reliable forensic extraction."
        )

    df = _native_pdf_tables(data)
    extraction_method = "Native PDF table extraction"

    if df.empty:
        df = _native_pdf_position_rows(data)
        extraction_method = "Native PDF column-position extraction"

    if df.empty:
        raise ValueError(
            "Digital PDF detected, but the transaction columns could not be mapped reliably. "
            "No rows were invented."
        )

    # Remove obvious repeated headers and rows without a usable date.
    df = df[df["Date"].notna()].copy()
    if df.empty:
        raise ValueError("PDF extraction produced no reliable dated transaction rows.")

    amount_presence = df[["Debit", "Credit", "Balance"]].notna().any(axis=1).mean()
    if amount_presence < 0.60:
        raise ValueError(
            f"PDF extraction confidence is too low ({amount_presence:.0%} of rows contain a monetary field). "
            "The source layout needs a bank-specific parser; analysis was stopped to protect evidence integrity."
        )

    if "Source_Page" not in df:
        df["Source_Page"] = np.nan

    df = enrich(df)
    flags = df[df.Priority.isin(["REVIEW", "CRITICAL"])].copy()

    meta = {
        "source_type": "PDF — native/digital",
        "location": f"{pages} pages",
        "layout_confidence": extraction_method,
        "warnings": [
            "Native PDF rows retain Source_Page for evidence tracing.",
            "Scanned/image PDFs are stopped rather than converted into guessed transactions."
        ]
    }
    return {"transactions": df, "flags": flags, "meta": meta}


def analyze_upload(data, name):
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        return analyze_pdf(data, name)
    if ext in {".xlsx", ".xls", ".csv"}:
        return analyze_excel(data, name)
    raise ValueError("Unsupported file type. Upload XLSX, XLS, CSV, or PDF.")



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
                cell.alignment=Alignment(vertical="top",wrap_text=True)
                cell.border=border
                cell.fill=PatternFill(fill_type=None)
        for cell in ws[1]:
            cell.font=Font(name="Bookman Old Style",size=10,bold=True)
            cell.alignment=Alignment(horizontal="center",vertical="center",wrap_text=True)
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
            ws.column_dimensions[get_column_letter(col_cells[0].column)].width=min(max(max([len(v) for v in vals]+[10])+2,12),48)
        for row in ws.iter_rows():
            if row and str(row[0].value or "") in section_names:
                for cell in row:
                    cell.font=Font(name="Bookman Old Style",size=10,bold=True)
                    cell.fill=PatternFill(fill_type=None)
                    cell.border=border

def build_workbook(df, flags, meta):
    out=io.BytesIO()
    export_df=df.drop(columns=["Counterparty"],errors="ignore").copy()
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
    """Create a concise human-review PDF report from the analysed evidence."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

    master, fund_flow, concentration, dq, _ = build_master_analysis(df)
    rails = df.groupby("Payment_Rail", dropna=False).agg(
        Transactions=("Narration", "size"), Credits=("Credit", "sum"), Debits=("Debit", "sum")
    ).reset_index().sort_values("Transactions", ascending=False)

    cats = df.groupby("Category", dropna=False).agg(
        Transactions=("Narration", "size"), Credits=("Credit", "sum"), Debits=("Debit", "sum")
    ).reset_index().sort_values("Transactions", ascending=False)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, rightMargin=14*mm, leftMargin=14*mm,
        topMargin=14*mm, bottomMargin=14*mm
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("FT", parent=styles["Title"], alignment=TA_CENTER, fontName="Helvetica-Bold", fontSize=18, spaceAfter=8)
    small = ParagraphStyle("FS", parent=styles["BodyText"], fontSize=8.5, leading=11)
    head = ParagraphStyle("FH", parent=styles["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=5)

    story = [
        Paragraph("FORENSIC INTELLIGENCE 360°", title),
        Paragraph("Executive Forensic Analysis Report", styles["Heading2"]),
        Paragraph(f"Evidence: {filename}", small),
        Paragraph(f"Source: {meta.get('source_type','-')} • {meta.get('location','-')}", small),
        Spacer(1, 6),
    ]

    summary_data = [
        ["Metric", "Result"],
        ["Transactions", f"{len(df):,}"],
        ["Credits", f"₹{df.Credit.fillna(0).sum():,.2f}"],
        ["Debits", f"₹{df.Debit.fillna(0).sum():,.2f}"],
        ["Net Flow", f"₹{(df.Credit.fillna(0).sum()-df.Debit.fillna(0).sum()):,.2f}"],
        ["Review", f"{int((df.Priority=='REVIEW').sum()):,}"],
        ["Critical", f"{int((df.Priority=='CRITICAL').sum()):,}"],
        ["Balance mismatches", f"{balance_mismatches(df):,}"],
    ]
    t = Table(summary_data, colWidths=[60*mm, 80*mm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17324d")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),0.4,colors.grey),("FONTNAME",(0,1),(-1,-1),"Helvetica"),("FONTSIZE",(0,0),(-1,-1),8)]))
    story += [t, Spacer(1, 8), Paragraph("Key Review Findings", head)]

    findings = [["Finding", "Value", "Interpretation", "Status"]]
    for _, r in master.iterrows():
        findings.append([str(r["Finding"]), str(r["Value"]), str(r["Interpretation"]), str(r["Review_Status"])])
    ft = Table(findings, repeatRows=1, colWidths=[48*mm, 28*mm, 70*mm, 25*mm])
    ft.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17324d")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),6.5),("VALIGN",(0,0),(-1,-1),"TOP")]))
    story += [ft, PageBreak(), Paragraph("Payment & Category Profile", head)]

    for title_text, table_df in [("Payment Rails", rails.head(20)), ("Categories", cats.head(20))]:
        story.append(Paragraph(title_text, styles["Heading3"]))
        data = [list(table_df.columns)] + table_df.fillna("-").astype(str).values.tolist()
        tb = Table(data, repeatRows=1)
        tb.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17324d")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),6.5)]))
        story += [tb, Spacer(1, 8)]

    story += [PageBreak(), Paragraph("Fund-Flow Review", head)]
    if fund_flow.empty:
        story.append(Paragraph("No configured large-credit / onward-debit pattern identified.", small))
    else:
        ff = fund_flow.head(40).fillna("-").astype(str)
        data = [list(ff.columns)] + ff.values.tolist()
        tb = Table(data, repeatRows=1, repeatCols=1)
        tb.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17324d")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.25,colors.grey),("FONTSIZE",(0,0),(-1,-1),5.5),("VALIGN",(0,0),(-1,-1),"TOP")]))
        story.append(tb)

    story += [Spacer(1, 8), Paragraph("Conclusion", head),
              Paragraph(
                  "This report identifies evidence-led patterns and review priorities. "
                  "It does not conclude fraud, illegality, intent or guilt. Material findings must be verified against the original bank statement.",
                  small
              )]
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


def balance_check(df):
    rows = []
    for i in range(1, len(df)):
        p = df.iloc[i - 1]
        c = df.iloc[i]
        if pd.notna(p.Balance) and pd.notna(c.Balance):
            credit = 0 if pd.isna(c.Credit) else c.Credit
            debit = 0 if pd.isna(c.Debit) else c.Debit
            expected = p.Balance + credit - debit
            if abs(expected - c.Balance) > 0.01:
                rows.append({
                    "Source_Row": c.get("Source_Row", ""),
                    "Source_Page": c.get("Source_Page", ""),
                    "Date": c.Date,
                    "Narration": c.Narration,
                    "Expected": expected,
                    "Reported": c.Balance,
                    "Difference": c.Balance - expected
                })
    return pd.DataFrame(rows, columns=["Source_Row", "Source_Page", "Date", "Narration", "Expected", "Reported", "Difference"])


def balance_mismatches(df):
    return len(balance_check(df))


def _ocr_pdf_position_rows(data):
    import fitz
    import pytesseract
    from PIL import Image
    doc=fitz.open(stream=data,filetype="pdf")
    frames=[]
    for page_no,page in enumerate(doc,1):
        pix=page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
        img=Image.frombytes("RGB",[pix.width,pix.height],pix.samples)
        od=pytesseract.image_to_data(img,config="--psm 6",output_type=pytesseract.Output.DICT)
        words=[]
        for i,txt in enumerate(od["text"]):
            txt=(txt or "").strip()
            try: conf=float(od["conf"][i])
            except: conf=-1
            if txt and conf>=25:
                words.append((od["left"][i],od["top"][i],od["left"][i]+od["width"][i],od["top"][i]+od["height"][i],txt))
        lines={}
        for w in words: lines.setdefault(round(w[1]/8)*8,[]).append(w)
        header=None; score=-1
        for y,ws in lines.items():
            txt=" ".join(w[4] for w in sorted(ws,key=lambda z:z[0])); nrm=norm(txt)
            sc=sum(k in nrm for k in ["date","description","narration","particular","debit","credit","balance","withdrawal","deposit"])
            if sc>score and sc>=3: score=sc; header=(y,ws)
        if header is None: continue
        hws=sorted(header[1],key=lambda z:z[0])
        cx=lambda w:(w[0]+w[2])/2
        def fx(patterns):
            xs=[cx(w) for w in hws if any(p in norm(w[4]) for p in patterns)]
            return min(xs) if xs else None
        cols=[("date",fx(["post date","transaction date","txn date","date"])),("value",fx(["value date"])),("narr",fx(["description","narration","particular","details","remarks"])),("ref",fx(["reference","ref no","cheque no","utr","txn id","transaction id"])),("debit",fx(["debit","withdrawal","withdraw"])),("credit",fx(["credit","deposit"])),("balance",fx(["balance","closing balance"]))]
        cols=[x for x in cols if x[1] is not None]
        if not any(k=="date" for k,_ in cols) or not any(k=="balance" for k,_ in cols): continue
        nearest=lambda x:min(cols,key=lambda z:abs(z[1]-x))[0]
        for y,ws in sorted(lines.items()):
            if y<=header[0]+4: continue
            ws=sorted(ws,key=lambda z:z[0]); text=" ".join(w[4] for w in ws)
            dm=re.search(r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b",text)
            if not dm: continue
            b={k:[] for k,_ in cols}
            for w in ws: b[nearest(cx(w))].append(w[4])
            dt=" ".join(b.get("date",[])); dm2=re.search(r"\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}",dt) or dm
            datev=pd.to_datetime(dm2.group(0).replace(".","-").replace("/","-"),dayfirst=True,errors="coerce")
            if pd.isna(datev): continue
            def amt(k):
                for v in reversed(b.get(k,[])):
                    z=money(v)
                    if pd.notna(z): return abs(z)
                return np.nan
            narr=" ".join(b.get("narr",[])).strip() or text
            value=pd.NaT
            vm=re.search(r"\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"," ".join(b.get("value",[])))
            if vm: value=pd.to_datetime(vm.group(0).replace(".","-").replace("/","-"),dayfirst=True,errors="coerce")
            frames.append(pd.DataFrame([{"Date":datev,"Value_Date":value,"Narration":narr,"Reference":" ".join(b.get("ref",[])).strip(),"Debit":amt("debit"),"Credit":amt("credit"),"Balance":amt("balance"),"Source_Page":page_no}]))
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def analyze_pdf(data,name):
    import fitz
    doc=fitz.open(stream=data,filetype="pdf")
    pages=len(doc); texts=[p.get_text("text") for p in doc]
    nonempty=sum(bool(t.strip()) for t in texts); ratio=nonempty/pages if pages else 0
    extraction="Native PDF table extraction"
    df=_native_pdf_tables(data) if ratio>=0.5 else pd.DataFrame()
    if df.empty and ratio>=0.5:
        df=_native_pdf_position_rows(data); extraction="Native PDF column-position extraction"
    if df.empty:
        extraction="OCR scanned-PDF extraction"
        try: df=_ocr_pdf_position_rows(data)
        except Exception as exc: raise ValueError("Scanned PDF detected, but OCR could not be completed. Check statement readability.") from exc
    if df.empty: raise ValueError("No reliable transaction table could be reconstructed from this PDF.")
    df=df[df["Date"].notna()].copy()
    if df.empty: raise ValueError("PDF extraction produced no reliable dated transaction rows.")
    presence=df[["Debit","Credit","Balance"]].notna().any(axis=1).mean()
    if presence<0.60: raise ValueError(f"PDF extraction confidence is too low ({presence:.0%}). No values were invented.")
    if "Source_Page" not in df: df["Source_Page"]=np.nan
    df=enrich(df); flags=df[df.Priority.isin(["REVIEW","CRITICAL"])].copy()
    return {"transactions":df,"flags":flags,"meta":{"source_type":"PDF — scanned/OCR" if ratio<0.5 else "PDF — native/digital","location":f"{pages} pages","layout_confidence":extraction,"warnings":[f"Extraction method: {extraction}. Source_Page retained."]}}



