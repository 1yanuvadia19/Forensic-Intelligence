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
            "OCR + table reconstruction is required; the app will not invent rows from images."
        )

    df = _native_pdf_tables(data)
    extraction_method = "Native PDF table extraction"

    if df.empty:
        df = _native_pdf_text_rows(data)
        extraction_method = "Native PDF text/position extraction"

    if df.empty:
        raise ValueError(
            "This PDF contains digital text, but no reliable transaction table could be reconstructed. "
            "The statement layout needs a bank-specific parser rather than inventing debit/credit values."
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
            "Original evidence is preserved through Source_Page.",
            "Scanned/image PDFs are intentionally blocked until OCR/table reconstruction is available."
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
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    text_cols = {
        "Narration", "Reference", "Counterparty", "Payment_Rail", "Category",
        "Flag_Reason", "Priority", "Source_Sheet", "Source_Page", "Source_Row", "Note"
    }

    for ws in wb.worksheets:
        # Header must be bold but NOT frozen.
        ws.freeze_panes = None
        ws.auto_filter.ref = ws.dimensions

        for row in ws.iter_rows():
            for cell in row:
                cell.font = Font(name="Bookman Old Style", size=10, bold=cell.row == 1)
                cell.alignment = Alignment(
                    vertical="top",
                    wrap_text=True if cell.column >= 2 else False
                )

        for cell in ws[1]:
            cell.font = Font(name="Bookman Old Style", size=10, bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        headers = {str(c.value): c.column for c in ws[1] if c.value is not None}
        for h, col_idx in headers.items():
            col = get_column_letter(col_idx)
            hnorm = norm(h)
            if hnorm in {"date", "value date"} or "date" in hnorm:
                for c in ws.iter_cols(min_col=col_idx, max_col=col_idx, min_row=2):
                    for cell in c:
                        if cell.value is not None:
                            cell.number_format = "dd-mm-yyyy"

            if hnorm in {"debit", "credit", "balance", "expected", "reported", "difference", "credits", "debits", "net flow"} or any(k in hnorm for k in ["amount", "value"]):
                for c in ws.iter_cols(min_col=col_idx, max_col=col_idx, min_row=2):
                    for cell in c:
                        if isinstance(cell.value, (int, float, np.integer, np.floating)) and not pd.isna(cell.value):
                            cell.number_format = '₹ #,##0.00;[Red]-₹ #,##0.00;₹ -'

            if h in text_cols or any(k in hnorm for k in ["narration", "description", "remarks", "reason", "counterparty", "category", "rail"]):
                for c in ws.iter_cols(min_col=col_idx, max_col=col_idx, min_row=2):
                    for cell in c:
                        if cell.value is None or (isinstance(cell.value, str) and not cell.value.strip()):
                            cell.value = "-"

        # Professional widths.
        for col_cells in ws.columns:
            values = [str(c.value) if c.value is not None else "" for c in col_cells[:250]]
            width = max([len(v) for v in values] + [10])
            width = min(max(width + 2, 12), 48)
            ws.column_dimensions[get_column_letter(col_cells[0].column)].width = width

        # Add a filter/table without freezing panes.
        if ws.max_row >= 2 and ws.max_column >= 1:
            ref = f"A1:{get_column_letter(ws.max_column)}{ws.max_row}"
            try:
                tab = Table(displayName=f"T_{re.sub(r'[^A-Za-z0-9]', '', ws.title)[:20]}", ref=ref)
                tab.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False, showRowStripes=True, showColumnStripes=False)
                ws.add_table(tab)
            except Exception:
                pass


def build_workbook(df, flags, meta):
    out = io.BytesIO()

    credits = df["Credit"].fillna(0)
    debits = df["Debit"].fillna(0)
    master, fund_flow, concentration, dq, provenance = build_master_analysis(df)

    summary = pd.DataFrame({
        "Metric": [
            "Source type", "Location", "Transactions", "Credits", "Debits",
            "Net Flow", "Review", "Critical", "Balance mismatches"
        ],
        "Value": [
            meta["source_type"], meta["location"], len(df), credits.sum(), debits.sum(),
            credits.sum() - debits.sum(),
            int((df.Priority == "REVIEW").sum()),
            int((df.Priority == "CRITICAL").sum()),
            balance_mismatches(df)
        ]
    })

    rails = df.groupby("Payment_Rail", dropna=False).agg(
        Transactions=("Narration", "size"),
        Credits=("Credit", "sum"),
        Debits=("Debit", "sum")
    ).reset_index()

    cats = df.groupby("Category", dropna=False).agg(
        Transactions=("Narration", "size"),
        Credits=("Credit", "sum"),
        Debits=("Debit", "sum")
    ).reset_index()

    with pd.ExcelWriter(out, engine="openpyxl") as w:
        summary.to_excel(w, index=False, sheet_name="01_Summary")
        df.to_excel(w, index=False, sheet_name="02_Normalised_Transactions")
        flags.to_excel(w, index=False, sheet_name="03_Review_Queue")
        rails.to_excel(w, index=False, sheet_name="04_Payment_Rails")
        cats.to_excel(w, index=False, sheet_name="05_Categories")

        if "Counterparty" in df:
            cp_old = df[df.Counterparty.fillna("") != ""].groupby("Counterparty").agg(
                Transactions=("Narration", "size"),
                Credits=("Credit", "sum"),
                Debits=("Debit", "sum")
            ).sort_values("Credits", ascending=False).reset_index()
            cp_old.to_excel(w, index=False, sheet_name="06_Counterparties")

        monthly = df.copy()
        monthly["Month"] = monthly["Date"].dt.to_period("M").astype(str)
        monthly.groupby("Month", dropna=False).agg(
            Transactions=("Narration", "size"),
            Credits=("Credit", "sum"),
            Debits=("Debit", "sum")
        ).reset_index().to_excel(w, index=False, sheet_name="07_Monthly_Flow")

        balance_check(df).to_excel(w, index=False, sheet_name="08_Balance_Check")
        master.to_excel(w, index=False, sheet_name="09_Master_Analysis")
        fund_flow.to_excel(w, index=False, sheet_name="10_Fund_Flow_Review")
        concentration.to_excel(w, index=False, sheet_name="11_Counterparty_Concentration")
        dq.to_excel(w, index=False, sheet_name="12_Data_Quality")
        provenance.to_excel(w, index=False, sheet_name="13_Evidence_Provenance")
        pd.DataFrame({
            "Note": [
                "MASTER FORENSIC ANALYSIS: findings are evidence-led review priorities, not conclusions of fraud or illegality.",
                f"Layout confidence: {meta.get('layout_confidence', '-')}",
                "Missing numeric values are preserved as blank/NaN; they are never silently converted to zero.",
                "Header rows are bold and intentionally unfrozen.",
                "Workbook font: Bookman Old Style. Dates: dd-mm-yyyy. Monetary fields: Indian ₹ accounting.",
                "Unknown payment rail is a data-quality limitation and is not treated as suspicious by itself.",
                "Every material review finding should be verified against the original bank statement before investigative use."
            ]
        }).to_excel(w, index=False, sheet_name="14_Notes")

        _format_workbook(w.book)

    out.seek(0)
    return out.getvalue()


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
