import io
import re
from pathlib import Path

import numpy as np
import pandas as pd


CANON = [
    "Date",
    "Value_Date",
    "Narration",
    "Reference",
    "Debit",
    "Credit",
    "Balance",
]


# ============================================================
# BASIC UTILITIES
# ============================================================

def norm(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def money(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    s = str(value).strip()
    s = s.replace(",", "").replace("₹", "").replace("INR", "").strip()

    if s in {"", "-", "—", "nan", "None"}:
        return np.nan

    # Remove common CR/DR markers
    s = re.sub(r"\b(CR|DR)\b", "", s, flags=re.I).strip()

    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else np.nan


def clean_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


# ============================================================
# EXCEL / CSV
# ============================================================

def detect_header(raw):
    targets = [
        "post date",
        "transaction date",
        "txn date",
        "date",
        "value date",
        "description",
        "narration",
        "particular",
        "remarks",
        "cheque",
        "reference",
        "debit",
        "withdrawal",
        "credit",
        "deposit",
        "balance",
    ]

    best_score = -1
    best_row = None

    for i in range(min(len(raw), 50)):
        values = [norm(v) for v in raw.iloc[i].tolist()]

        score = 0
        for value in values:
            for target in targets:
                if target in value:
                    score += 1
                    break

        if score > best_score:
            best_score = score
            best_row = i

    if best_score < 3:
        raise ValueError("Could not confidently locate transaction header.")

    return best_row, best_score


def pick_column(columns, patterns):
    best = None
    best_score = 0

    for column in columns:
        n = norm(column)

        for pattern in patterns:
            if pattern in n and len(pattern) > best_score:
                best = column
                best_score = len(pattern)

    return best


def standardize_excel(raw):
    raw = raw.dropna(how="all").dropna(axis=1, how="all").copy()

    if raw.empty:
        raise ValueError("Empty worksheet.")

    header_row, confidence = detect_header(raw)

    headers = [
        clean_text(x) if clean_text(x) else f"Unnamed_{i}"
        for i, x in enumerate(raw.iloc[header_row].tolist())
    ]

    df = raw.iloc[header_row + 1:].copy()
    df.columns = headers
    df = df.dropna(how="all")

    cols = list(df.columns)

    date_col = pick_column(
        cols,
        ["post date", "transaction date", "txn date", "date"]
    )

    value_col = pick_column(
        cols,
        ["value date"]
    )

    narration_col = pick_column(
        cols,
        ["description", "narration", "particular", "remarks", "details"]
    )

    reference_col = pick_column(
        cols,
        ["cheque no", "reference", "ref no", "utr", "transaction id", "txn id"]
    )

    debit_col = pick_column(
        cols,
        ["debit rs", "debit", "withdrawal", "withdraw"]
    )

    credit_col = pick_column(
        cols,
        ["credit rs", "credit", "deposit"]
    )

    balance_col = pick_column(
        cols,
        ["balance rs", "balance", "closing balance"]
    )

    if narration_col is None:
        raise ValueError("Narration/Description column not found.")

    out = pd.DataFrame()

    out["Date"] = (
        pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        if date_col else pd.NaT
    )

    out["Value_Date"] = (
        pd.to_datetime(df[value_col], errors="coerce", dayfirst=True)
        if value_col else pd.NaT
    )

    out["Narration"] = (
        df[narration_col].map(clean_text)
    )

    out["Reference"] = (
        df[reference_col].map(clean_text)
        if reference_col else ""
    )

    # IMPORTANT:
    # Missing numeric information remains NaN.
    # We do NOT silently convert missing data to zero.
    out["Debit"] = (
        df[debit_col].map(money)
        if debit_col else np.nan
    )

    out["Credit"] = (
        df[credit_col].map(money)
        if credit_col else np.nan
    )

    out["Balance"] = (
        df[balance_col].map(money)
        if balance_col else np.nan
    )

    out["Source_Row"] = range(
        header_row + 2,
        header_row + 2 + len(out)
    )

    valid = (
        out["Date"].notna()
        | out["Narration"].ne("")
        | out["Debit"].notna()
        | out["Credit"].notna()
        | out["Balance"].notna()
    )

    out = out[valid].reset_index(drop=True)

    if out.empty:
        raise ValueError("No transaction rows detected.")

    return out, header_row, confidence


# ============================================================
# PAYMENT RAIL INTELLIGENCE
# ============================================================

def payment_rail(narration):
    text = str(narration).upper()

    rules = [
        ("UPI", r"\bUPI\b|UPI/|BHIM|PHONEPE|GPAY|GOOGLEPAY|PAYTM|VPA"),
        ("IMPS", r"\bIMPS\b"),
        ("NEFT", r"\bNEFT\b"),
        ("RTGS", r"\bRTGS\b"),
        (
            "NACH/ECS/ACH",
            r"\bNACH\b|\bECS\b|\bACH\b|ACHDR|NACHDR"
        ),
        (
            "ATM/CASH",
            r"\bATM\b|CASH WITHDRAWAL|CASH DEPOSIT"
        ),
        (
            "CHEQUE",
            r"\bCHQ\b|\bCHEQUE\b|\bCTS\b"
        ),
        (
            "CARD",
            r"CREDIT CARD|DEBIT CARD|CARD PAYMENT"
        ),
        (
            "BROKING/INVESTMENT",
            r"BROKING|BROKER|ZERODHA|ANGEL ONE|ANGELONE|SECURITIES"
        ),
        (
            "INSURANCE/PREMIUM",
            r"PREMIUM|INSURANCE|LIC"
        ),
        (
            "LOAN/FINANCE",
            r"LOAN|HOUSINGFINA|PIRAMAL|EMI"
        ),
        (
            "BANK CHARGES",
            r"BANK CHARGES|CHARGES|GST ON"
        ),
        (
            "INTEREST",
            r"INTEREST"
        ),
        (
            "RECHARGE/UTILITY",
            r"RECHARGE|JIO|VODAFONE|ELECTRIC"
        ),
        (
            "TRANSFER",
            r"\bTFR\b|TRANSFER|TRF"
        ),
    ]

    for name, pattern in rules:
        if re.search(pattern, text):
            return name

    return "OTHER / UNIDENTIFIED"


# ============================================================
# CATEGORY
# ============================================================

def category(narration, debit, credit):
    text = str(narration).upper()

    if "CASH DEPOSIT" in text:
        return "Cash Deposit"

    if "CASH WITHDRAWAL" in text:
        return "Cash Withdrawal"

    if "BANK CHARGES" in text:
        return "Bank Charges"

    if "INTEREST" in text:
        return "Interest"

    if "PREMIUM" in text or "LIC" in text:
        return "Insurance / Premium"

    if any(x in text for x in ["LOAN", "HOUSINGFINA", "PIRAMAL", "EMI"]):
        return "Loan / Finance"

    if any(x in text for x in ["BROKING", "ANGEL ONE", "ZERODHA", "SECURITIES"]):
        return "Investment / Broking"

    if "CREDIT CARD" in text:
        return "Credit Card"

    if any(x in text for x in ["RECHARGE", "JIO", "VODAFONE"]):
        return "Recharge / Utility"

    if any(x in text for x in ["AMAZON", "FLIPKART", "SHOPSY", "ONLINE"]):
        return "Online / Merchant"

    if "SELF" in text:
        return "Self / Internal"

    if pd.notna(credit) and pd.isna(debit):
        return "Other Credit / Receipt"

    if pd.notna(debit) and pd.isna(credit):
        return "Other Debit / Expense"

    return "Other"


# ============================================================
# COUNTERPARTY
# ============================================================

def counterparty(narration):
    text = clean_text(narration)

    if not text:
        return ""

    if text.upper() in {
        "CASH DEPOSIT",
        "CASH WITHDRAWAL",
        "SELF",
        "BANK CHARGES",
        "INTEREST CREDIT",
    }:
        return ""

    return text[:160]


# ============================================================
# ENRICHMENT
# ============================================================

def enrich(df):
    x = df.copy()

    x["Counterparty"] = x["Narration"].map(counterparty)

    x["Payment_Rail"] = [
        payment_rail(v)
        for v in x["Narration"]
    ]

    x["Category"] = [
        category(t, d, c)
        for t, d, c in zip(
            x["Narration"],
            x["Debit"],
            x["Credit"]
        )
    ]

    amount = (
        x["Debit"].fillna(0)
        + x["Credit"].fillna(0)
    )

    positive_amounts = amount[amount > 0]

    q95 = (
        positive_amounts.quantile(0.95)
        if len(positive_amounts)
        else np.nan
    )

    q99 = (
        positive_amounts.quantile(0.99)
        if len(positive_amounts)
        else np.nan
    )

    reasons = []
    priorities = []

    for _, row in x.iterrows():

        amt = (
            (row["Debit"] if pd.notna(row["Debit"]) else 0)
            + (row["Credit"] if pd.notna(row["Credit"]) else 0)
        )

        reason = []
        priority = "NORMAL"

        if pd.notna(q99) and amt >= q99 and amt > 0:
            reason.append("Top 1% by transaction amount")
            priority = "REVIEW"

        elif pd.notna(q95) and amt >= q95 and amt > 0:
            reason.append("Top 5% by transaction amount")
            priority = "REVIEW"

        # Unknown rail alone is NOT treated as suspicious.
        # It is only recorded for information.
        if row["Payment_Rail"] == "OTHER / UNIDENTIFIED":
            reason.append("Payment rail not identified")

        reasons.append("; ".join(reason))
        priorities.append(priority)

    x["Flag_Reason"] = reasons
    x["Priority"] = priorities
    x["Rapid_Movement"] = False

    # Same-day / next-day onward movement
    credits = x[x["Credit"].fillna(0) > 0]
    debits = x[x["Debit"].fillna(0) > 0]

    for credit_index, credit_row in credits.iterrows():

        credit_amount = credit_row["Credit"]

        if pd.isna(credit_amount) or credit_amount < 100000:
            continue

        if pd.isna(credit_row["Date"]):
            continue

        candidates = debits[
            (debits["Date"] >= credit_row["Date"])
            & (
                debits["Date"]
                <= credit_row["Date"] + pd.Timedelta(days=1)
            )
        ]

        hits = candidates[
            candidates["Debit"].fillna(0)
            >= credit_amount * 0.80
        ]

        if len(hits):

            x.loc[credit_index, "Rapid_Movement"] = True

            for hit_index in hits.index:
                x.loc[hit_index, "Rapid_Movement"] = True

    mask = x["Rapid_Movement"]

    x.loc[mask, "Flag_Reason"] = (
        x.loc[mask, "Flag_Reason"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "Rapid onward movement")
        .apply(
            lambda s:
            s
            if "Rapid onward movement" in s
            else s + "; Same/next-day large onward movement"
        )
    )

    x.loc[mask, "Priority"] = "REVIEW"

    return x


# ============================================================
# PDF ENGINE
# ============================================================

DATE_PATTERN = re.compile(
    r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b"
)


def detect_pdf_type(data):
    """
    Returns:
        NATIVE  -> text layer exists
        SCANNED -> no usable text layer
        HYBRID  -> mixed pages
    """

    import fitz

    document = fitz.open(
        stream=data,
        filetype="pdf"
    )

    total = len(document)
    text_pages = 0

    for page in document:
        text = page.get_text("text") or ""

        if len(text.strip()) >= 30:
            text_pages += 1

    if text_pages == 0:
        return "SCANNED", total, text_pages

    if text_pages == total:
        return "NATIVE", total, text_pages

    return "HYBRID", total, text_pages


def parse_pdf_table(data):
    """
    Extract transaction tables from native bank PDFs using pdfplumber.
    """

    import pdfplumber

    transactions = []

    with pdfplumber.open(io.BytesIO(data)) as pdf:

        for page_number, page in enumerate(
            pdf.pages,
            start=1
        ):

            tables = page.extract_tables()

            for table in tables:

                if not table:
                    continue

                for row in table:

                    if not row:
                        continue

                    values = [
                        clean_text(v)
                        for v in row
                    ]

                    joined = " ".join(values)

                    # Must contain a date to be considered
                    # a possible transaction row.
                    date_match = DATE_PATTERN.search(joined)

                    if not date_match:
                        continue

                    transactions.append(
                        {
                            "page": page_number,
                            "values": values,
                        }
                    )

    return transactions


def map_pdf_row(values):
    """
    Conservative bank-statement row mapping.

    We use:
      - date detection
      - amount detection
      - header-independent position heuristics

    We do NOT manufacture missing values.
    """

    if not values:
        return None

    cells = [
        clean_text(v)
        for v in values
    ]

    date_index = None

    for i, cell in enumerate(cells):
        if DATE_PATTERN.search(cell):
            date_index = i
            break

    if date_index is None:
        return None

    date_match = DATE_PATTERN.search(
        cells[date_index]
    )

    date_value = pd.to_datetime(
        date_match.group(),
        errors="coerce",
        dayfirst=True,
    )

    if pd.isna(date_value):
        return None

    numeric_cells = []

    for i, cell in enumerate(cells):
        if i == date_index:
            continue

        value = money(cell)

        if pd.notna(value):
            numeric_cells.append(
                (i, value)
            )

    if not numeric_cells:
        return None

    # Most bank statements have the final numeric
    # column as Balance.
    balance_index, balance_value = numeric_cells[-1]

    amount_candidates = numeric_cells[:-1]

    debit = np.nan
    credit = np.nan

    # Conservative assignment:
    # when two amount columns exist, preserve both.
    if len(amount_candidates) >= 2:

        debit = amount_candidates[-2][1]
        credit = amount_candidates[-1][1]

        # Last numeric is probably balance in normal
        # bank statements, so shift accordingly.
        balance_index, balance_value = numeric_cells[-1]

        debit = amount_candidates[-2][1]
        credit = amount_candidates[-1][1]

    elif len(amount_candidates) == 1:

        # One amount + balance.
        # Do NOT guess debit vs credit.
        debit = np.nan
        credit = amount_candidates[0][1]

    narration_parts = []

    for i, cell in enumerate(cells):

        if i == date_index:
            continue

        if i == balance_index:
            continue

        if money(cell) is not None and pd.notna(money(cell)):
            continue

        if cell:
            narration_parts.append(cell)

    narration = " ".join(narration_parts).strip()

    return {
        "Date": date_value,
        "Value_Date": date_value,
        "Narration": narration,
        "Reference": "",
        "Debit": debit,
        "Credit": credit,
        "Balance": balance_value,
    }


def analyze_pdf(data, name):

    pdf_type, page_count, text_pages = detect_pdf_type(data)

    # --------------------------------------------------------
    # SCANNED PDF
    # --------------------------------------------------------

    if pdf_type == "SCANNED":

        raise ValueError(
            f"This PDF is image/scanned based "
            f"({page_count} pages, 0 usable text pages). "
            f"OCR + table reconstruction is required before "
            f"forensic analysis. No rows were invented."
        )

    # --------------------------------------------------------
    # NATIVE / HYBRID
    # --------------------------------------------------------

    rows = parse_pdf_table(data)

    if not rows:

        raise ValueError(
            "The PDF contains extractable text, but no reliable "
            "transaction table could be reconstructed. "
            "The analysis has been stopped to protect evidence integrity."
        )

    mapped = []

    for item in rows:

        result = map_pdf_row(
            item["values"]
        )

        if result:

            result["Source_Page"] = item["page"]

            mapped.append(result)

    if not mapped:

        raise ValueError(
            "PDF table detected, but transaction-column mapping "
            "could not be validated. Analysis stopped."
        )

    df = pd.DataFrame(mapped)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    valid_dates = df["Date"].notna().mean()

    valid_balance = (
        df["Balance"].notna().mean()
    )

    valid_narration = (
        df["Narration"]
        .fillna("")
        .astype(str)
        .str.len()
        .gt(2)
        .mean()
    )

    confidence = (
        valid_dates * 0.35
        + valid_balance * 0.35
        + valid_narration * 0.30
    )

    if confidence < 0.60:

        raise ValueError(
            f"PDF extraction confidence is too low "
            f"({confidence:.0%}). "
            f"Please review the original PDF before analysis."
        )

    df = enrich(df)

    flags = df[
        df["Priority"].isin(
            ["REVIEW", "CRITICAL"]
        )
    ].copy()

    meta = {
        "source_type": f"PDF ({pdf_type})",
        "location": f"{page_count} pages",
        "layout_confidence": f"{confidence:.0%}",
        "warnings": [],
    }

    if pdf_type == "HYBRID":
        meta["warnings"].append(
            "Hybrid PDF detected. Some pages contained text "
            "while others may require OCR."
        )

    return {
        "transactions": df,
        "flags": flags,
        "meta": meta,
    }


# ============================================================
# EXCEL / CSV ANALYSIS
# ============================================================

def analyze_excel(data, name):

    extension = Path(name).suffix.lower()

    if extension == ".csv":

        raw = pd.read_csv(
            io.BytesIO(data),
            header=None
        )

        standardized, header_row, confidence = (
            standardize_excel(raw)
        )

        standardized["Source_Sheet"] = "CSV"

        df = enrich(standardized)

        flags = df[
            df["Priority"].isin(
                ["REVIEW", "CRITICAL"]
            )
        ].copy()

        return {
            "transactions": df,
            "flags": flags,
            "meta": {
                "source_type": "CSV",
                "location": "CSV file",
                "layout_confidence":
                    f"{confidence} matched header fields",
                "warnings": [],
            },
        }

    book = pd.ExcelFile(
        io.BytesIO(data)
    )

    frames = []
    details = []

    for sheet in book.sheet_names:

        raw = pd.read_excel(
            io.BytesIO(data),
            sheet_name=sheet,
            header=None
        )

        try:

            standardized, header_row, confidence = (
                standardize_excel(raw)
            )

            standardized["Source_Sheet"] = sheet

            frames.append(
                standardized
            )

            details.append(
                (
                    sheet,
                    header_row + 1,
                    confidence,
                    len(standardized),
                )
            )

        except Exception:
            continue

    if not frames:

        raise ValueError(
            "No sheet contained a confident transaction table."
        )

    df = pd.concat(
        frames,
        ignore_index=True
    )

    df = enrich(df)

    flags = df[
        df["Priority"].isin(
            ["REVIEW", "CRITICAL"]
        )
    ].copy()

    meta = {
        "source_type": "Excel",
        "location":
            f"{len(book.sheet_names)} sheet(s); "
            + ", ".join(
                x[0] for x in details
            ),
        "layout_confidence":
            f"{max(x[2] for x in details)} matched header fields",
        "warnings": [],
    }

    return {
        "transactions": df,
        "flags": flags,
        "meta": meta,
    }


# ============================================================
# UNIVERSAL UPLOAD ROUTER
# ============================================================

def analyze_upload(data, name):

    extension = Path(name).suffix.lower()

    if extension == ".pdf":
        return analyze_pdf(data, name)

    if extension == ".csv":
        return analyze_excel(data, name)

    if extension in {".xlsx", ".xls"}:
        return analyze_excel(data, name)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )


# ============================================================
# BALANCE RECONCILIATION
# ============================================================

def balance_check(df):

    rows = []

    for i in range(1, len(df)):

        previous = df.iloc[i - 1]
        current = df.iloc[i]

        if (
            pd.notna(previous["Balance"])
            and pd.notna(current["Balance"])
        ):

            debit = (
                current["Debit"]
                if pd.notna(current["Debit"])
                else 0
            )

            credit = (
                current["Credit"]
                if pd.notna(current["Credit"])
                else 0
            )

            expected = (
                previous["Balance"]
                + credit
                - debit
            )

            difference = (
                current["Balance"]
                - expected
            )

            if abs(difference) > 0.01:

                rows.append(
                    {
                        "Source_Row":
                            current.get(
                                "Source_Row",
                                ""
                            ),
                        "Date":
                            current["Date"],
                        "Narration":
                            current["Narration"],
                        "Expected":
                            expected,
                        "Reported":
                            current["Balance"],
                        "Difference":
                            difference,
                    }
                )

    return pd.DataFrame(rows)


def balance_mismatches(df):
    return len(balance_check(df))


# ============================================================
# FORENSIC EXCEL EXPORT
# ============================================================

def build_workbook(df, flags, meta):

    from openpyxl.styles import Font, Alignment
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    output = io.BytesIO()

    credits = df["Credit"].fillna(0).sum()
    debits = df["Debit"].fillna(0).sum()

    summary = pd.DataFrame(
        {
            "Metric": [
                "Source Type",
                "Location",
                "Transactions",
                "Credits",
                "Debits",
                "Net Flow",
                "Review",
                "Critical",
                "Balance Mismatches",
            ],
            "Value": [
                meta.get("source_type", ""),
                meta.get("location", ""),
                len(df),
                credits,
                debits,
                credits - debits,
                (df["Priority"] == "REVIEW").sum(),
                (df["Priority"] == "CRITICAL").sum(),
                balance_mismatches(df),
            ],
        }
    )

    rails = (
        df.groupby("Payment_Rail")
        .agg(
            Transactions=("Narration", "size"),
            Credits=("Credit", "sum"),
            Debits=("Debit", "sum"),
        )
        .reset_index()
    )

    categories = (
        df.groupby("Category")
        .agg(
            Transactions=("Narration", "size"),
            Credits=("Credit", "sum"),
            Debits=("Debit", "sum"),
        )
        .reset_index()
    )

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        summary.to_excel(
            writer,
            index=False,
            sheet_name="01_Summary"
        )

        df.to_excel(
            writer,
            index=False,
            sheet_name="02_Normalised_Transactions"
        )

        flags.to_excel(
            writer,
            index=False,
            sheet_name="03_Review_Queue"
        )

        rails.to_excel(
            writer,
            index=False,
            sheet_name="04_Payment_Rails"
        )

        categories.to_excel(
            writer,
            index=False,
            sheet_name="05_Categories"
        )

        if "Counterparty" in df.columns:

            cp = (
                df[df["Counterparty"] != ""]
                .groupby("Counterparty")
                .agg(
                    Transactions=("Narration", "size"),
                    Credits=("Credit", "sum"),
                    Debits=("Debit", "sum"),
                )
                .sort_values(
                    "Credits",
                    ascending=False
                )
                .reset_index()
            )

            cp.to_excel(
                writer,
                index=False,
                sheet_name="06_Counterparties"
            )

        monthly = (
            df.assign(
                Month=df["Date"]
                .dt.to_period("M")
                .astype(str)
            )
            .groupby("Month")
            .agg(
                Transactions=("Narration", "size"),
                Credits=("Credit", "sum"),
                Debits=("Debit", "sum"),
            )
            .reset_index()
        )

        monthly.to_excel(
            writer,
            index=False,
            sheet_name="07_Monthly_Flow"
        )

        balance_check(df).to_excel(
            writer,
            index=False,
            sheet_name="08_Balance_Check"
        )

        pd.DataFrame(
            {
                "Note": [
                    "Generated from source evidence.",
                    "Review flagged transactions against original evidence.",
                    "Missing numeric information is not treated as zero.",
                ]
            }
        ).to_excel(
            writer,
            index=False,
            sheet_name="09_Notes"
        )

        # ----------------------------------------------------
        # MASTER FORMATTING
        # ----------------------------------------------------

        for worksheet in writer.book.worksheets:

            # IMPORTANT:
            # Header is bold but NOT frozen.
            worksheet.freeze_panes = None

            worksheet.auto_filter.ref = (
                worksheet.dimensions
            )

            for cell in worksheet[1]:

                cell.font = Font(
                    name="Bookman Old Style",
                    size=11,
                    bold=True
                )

                cell.alignment = Alignment(
                    horizontal="center",
                    vertical="center",
                    wrap_text=True
                )

            for row in worksheet.iter_rows():

                for cell in row:

                    cell.font = Font(
                        name="Bookman Old Style",
                        size=11,
                        bold=(
                            cell.row == 1
                        )
                    )

                    cell.alignment = Alignment(
                        vertical="top",
                        wrap_text=True
                    )

                    # Missing text -> "-"
                    if (
                        cell.value is None
                        and cell.column
                    ):
                        cell.value = "-"

            # Date formatting
            for row in worksheet.iter_rows():

                for cell in row:

                    header = (
                        worksheet.cell(
                            1,
                            cell.column
                        ).value
                    )

                    if header in {
                        "Date",
                        "Value_Date"
                    }:

                        cell.number_format = (
                            "dd-mm-yyyy"
                        )

                    if header in {
                        "Debit",
                        "Credit",
                        "Balance",
                        "Expected",
                        "Reported",
                        "Difference",
                        "Net Flow",
                    }:

                        cell.number_format = (
                            '₹#,##,##0.00;'
                            '[Red]-₹#,##,##0.00;'
                            '-'
                        )

            # Professional column widths
            for column_cells in worksheet.columns:

                column_index = (
                    column_cells[0].column
                )

                values = [
                    str(c.value)
                    for c in column_cells[:100]
                    if c.value is not None
                ]

                longest = (
                    max(
                        [len(v) for v in values],
                        default=10
                    )
                )

                width = min(
                    max(longest + 2, 12),
                    45
                )

                worksheet.column_dimensions[
                    get_column_letter(
                        column_index
                    )
                ].width = width

            # Add Excel table where possible
            if (
                worksheet.max_row >= 2
                and worksheet.max_column >= 1
            ):

                ref = (
                    f"A1:"
                    f"{get_column_letter(worksheet.max_column)}"
                    f"{worksheet.max_row}"
                )

                table = Table(
                    displayName=
                    f"Table_{worksheet.title.replace('_', '')}",
                    ref=ref
                )

                style = TableStyleInfo(
                    name="TableStyleMedium2",
                    showFirstColumn=False,
                    showLastColumn=False,
                    showRowStripes=True,
                    showColumnStripes=False,
                )

                table.tableStyleInfo = style

                try:
                    worksheet.add_table(table)
                except Exception:
                    pass

    output.seek(0)

    return output.getvalue()
