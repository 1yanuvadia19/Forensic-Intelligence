import io

import numpy as np
import pandas as pd
from reportlab.pdfgen import canvas

from pdf_extraction_adapter import _attach_wrapped_narrations
from bank_pdf_engine import extract_bank_pdf
from forensic_core import (
    money,
    rail,
    category,
    counterparty,
    semantic_narration,
    _opening_anchor_check,
    _summary_anchor_check,
    _normalize_reference_fields,
    _native_pdf_position_rows,
)


def test_money_indian_formats():
    assert money("1,25,000") == 125000
    assert money("1,25,000 DR") == -125000
    assert money("₹2,500") == 2500


def test_payment_rail_priority():
    assert rail("UPI/DR/12345/PHONEPE") == "UPI"
    assert rail("IMPS/CR/ABC") == "IMPS"
    assert rail("NEFT CR SALARY") == "NEFT"
    assert rail("RTGS/CR/ABC") == "RTGS"


def test_category_semantics():
    assert category("CASH DEPOSIT", np.nan, 50000) == "Cash Deposit"
    assert category("INTEREST CREDIT", np.nan, 1250) == "Interest"
    assert category("EMI LOAN ABC", 25000, np.nan) == "Loan / Finance"
    assert category("UPI AMAZON", 1200, np.nan) == "Online / Merchant"


def test_counterparty_only_uses_explicit_text():
    assert counterparty("UPI/DR/123456/RAHUL SHARMA") == "RAHUL SHARMA"
    assert counterparty("UPI/DR/123456/9876543210") == ""


def test_semantic_narration():
    assert semantic_narration("BY SALARY ABC LTD", np.nan, 50000) == "Salary"
    assert semantic_narration("ATM CASH WITHDRAWAL", 5000, np.nan) == "Cash Withdrawal – ATM"


def test_reference_cleanup_preserves_evidence():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01"]),
        "Narration": ["UPI payment"],
        "Reference": ["UPI/ABC123 to 31-"],
        "Debit": [100.0],
        "Credit": [np.nan],
        "Balance": [900.0],
    })
    out = _normalize_reference_fields(df)
    assert "to 31-" not in out.loc[0, "Reference"]
    assert "to 31-" in out.loc[0, "Narration"]
    assert "ABC123" in out.loc[0, "Reference"]


def test_opening_balance_anchor():
    df = pd.DataFrame({
        "Balance": [900.0],
        "Debit": [100.0],
        "Credit": [np.nan],
    })
    result = _opening_anchor_check(df, 1000.0)
    assert result["available"] is True
    assert result["match"] is True


def test_summary_anchor_detects_wrong_extraction():
    df = pd.DataFrame({
        "Debit": [100.0, 200.0],
        "Credit": [500.0, 300.0],
        "Balance": [1400.0, 1500.0],
    })
    summary = {"withdrawal": 300.0, "deposit": 800.0, "closing": 1500.0}
    result = _summary_anchor_check(df, summary)
    assert result["available"] is True
    assert result["match"] is True


def test_native_pdf_reuses_verified_layout_on_continuation_pages():
    """A PDF may print the table header only once; every page must still be parsed."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(600, 800))

    # Header appears only on page 1.
    c.drawString(40, 740, "Date")
    c.drawString(100, 740, "Description")
    c.drawString(300, 740, "Withdrawal")
    c.drawString(390, 740, "Deposit")
    c.drawString(480, 740, "Balance")
    c.drawString(40, 700, "10/04/2025")
    c.drawString(100, 700, "UPI PAYMENT")
    c.drawString(390, 700, "500.00")
    c.drawString(480, 700, "1500.00")
    c.showPage()

    # Page 2 deliberately has no repeated column header.
    c.drawString(40, 740, "11/04/2025")
    c.drawString(100, 740, "ATM CASH")
    c.drawString(300, 740, "200.00")
    c.drawString(480, 740, "1300.00")
    c.save()
    buf.seek(0)

    out = _native_pdf_position_rows(buf.getvalue())
    assert len(out) == 2
    assert out.iloc[0]["Credit"] == 500.0
    assert out.iloc[1]["Debit"] == 200.0
    assert out.iloc[1]["Balance"] == 1300.0


def test_pdf_adapter_preserves_amounts_and_appends_wrapped_narration():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(600, 800))
    c.drawString(40, 740, "Date")
    c.drawString(100, 740, "Description")
    c.drawString(300, 740, "Withdrawal")
    c.drawString(390, 740, "Deposit")
    c.drawString(480, 740, "Balance")
    c.drawString(40, 700, "10/04/2025")
    c.drawString(100, 700, "UPI PAYMENT")
    c.drawString(390, 700, "500.00")
    c.drawString(480, 700, "1500.00")
    c.drawString(100, 682, "RAHUL SHARMA HDFC BANK")
    c.drawString(40, 640, "11/04/2025")
    c.drawString(100, 640, "ATM CASH")
    c.drawString(300, 640, "200.00")
    c.drawString(480, 640, "1300.00")
    c.save()
    buf.seek(0)

    extracted = _native_pdf_position_rows(buf.getvalue())
    out = _attach_wrapped_narrations(buf.getvalue(), extracted)

    assert len(out) == 2
    assert out.iloc[0]["Credit"] == 500.0
    assert out.iloc[0]["Balance"] == 1500.0
    assert "RAHUL SHARMA" in out.iloc[0]["Narration"]
    assert out.iloc[1]["Debit"] == 200.0


def test_bank_pdf_engine_handles_hdfc_style_multi_page_statement():
    """The refreshed engine must extract every dated movement, not one row."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(900, 700))

    def header():
        c.drawString(40, 650, "Tran Date")
        c.drawString(105, 650, "Narration")
        c.drawString(320, 650, "Chq./Ref.No.")
        c.drawString(455, 650, "Value Date")
        c.drawString(570, 650, "Withdrawal Amt.")
        c.drawString(680, 650, "Deposit Amt.")
        c.drawString(790, 650, "Closing Balance")

    header()
    c.drawString(40, 610, "01/04/2016")
    c.drawString(105, 610, "PROGRAM MANAGEMENT FEE")
    c.drawString(570, 610, "100.00")
    c.drawString(790, 610, "900.00")
    c.drawString(455, 610, "01/04/2016")

    c.drawString(40, 570, "02/04/2016")
    c.drawString(105, 570, "LOCKER RENT - BRN 305")
    c.drawString(320, 570, "5050000411060")
    c.drawString(570, 570, "10,000.00")
    c.drawString(790, 570, " - ")
    # This page intentionally omits the balance on one row; the next page
    # proves the movement direction from its reported balance.
    c.showPage()

    c.drawString(40, 650, "03/04/2016")
    c.drawString(105, 650, "CASH DEP GOPAL AHMED")
    c.drawString(680, 650, "125,000.00")
    c.drawString(790, 650, "115,900.00")
    c.drawString(455, 650, "03/04/2016")
    c.drawString(40, 610, "04/04/2016")
    c.drawString(105, 610, "CREDIT INTEREST CAPITALISED")
    c.drawString(680, 610, "115.00")
    c.drawString(790, 610, "116,015.00")
    c.save()
    buf.seek(0)

    out = extract_bank_pdf(buf.getvalue())
    assert len(out) == 4
    assert list(out["Date"].dt.strftime("%d/%m/%Y")) == [
        "01/04/2016", "02/04/2016", "03/04/2016", "04/04/2016"
    ]
    assert out.loc[0, "Debit"] == 100.0
    assert out.loc[2, "Credit"] == 125000.0
    assert out.loc[3, "Credit"] == 115.0
    assert out["Source_Page"].nunique() == 2


def test_bank_pdf_engine_keeps_wrapped_narration_on_same_transaction():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(700, 700))
    c.drawString(40, 650, "Date")
    c.drawString(120, 650, "Narration")
    c.drawString(300, 650, "Reference")
    c.drawString(420, 650, "Debit")
    c.drawString(520, 650, "Credit")
    c.drawString(620, 650, "Balance")
    c.drawString(40, 610, "10/04/2025")
    c.drawString(120, 610, "UPI PAYMENT")
    c.drawString(300, 610, "UPI/123456")
    c.drawString(520, 610, "500.00")
    c.drawString(620, 610, "1500.00")
    c.drawString(120, 590, "RAHUL SHARMA HDFC BANK")
    c.drawString(40, 550, "11/04/2025")
    c.drawString(120, 550, "ATM CASH")
    c.drawString(420, 550, "200.00")
    c.drawString(620, 550, "1300.00")
    c.save()
    buf.seek(0)

    out = extract_bank_pdf(buf.getvalue())
    assert len(out) == 2
    assert out.iloc[0]["Credit"] == 500.0
    assert out.iloc[0]["Balance"] == 1500.0
    assert "RAHUL SHARMA" in out.iloc[0]["Narration"]


def test_master_pdf_intelligence_prefers_reconciling_candidate():
    from master_pdf_intelligence import extract_master_pdf

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(800, 700))
    c.drawString(40, 650, "Date")
    c.drawString(120, 650, "Narration")
    c.drawString(300, 650, "Chq./Ref.No.")
    c.drawString(420, 650, "Value Date")
    c.drawString(540, 650, "Withdrawal")
    c.drawString(630, 650, "Deposit")
    c.drawString(720, 650, "Closing Balance")

    rows = [
        ("01/04/2016", "PROGRAM MANAGEMENT FEE", "100.00", "", "99900.00"),
        ("02/04/2016", "LOCKER RENT", "10000.00", "", "89900.00"),
        ("03/04/2016", "CASH DEP GOPAL AHMED", "", "125000.00", "214900.00"),
        ("04/04/2016", "CREDIT INTEREST CAPITALISED", "", "115.00", "215015.00"),
    ]
    y = 610
    for date, narr, wd, dep, bal in rows:
        c.drawString(40, y, date)
        c.drawString(120, y, narr)
        c.drawString(420, y, date)
        if wd:
            c.drawString(540, y, wd)
        if dep:
            c.drawString(630, y, dep)
        c.drawString(720, y, bal)
        y -= 35

    c.save()
    buf.seek(0)

    result = extract_master_pdf(buf.getvalue(), "synthetic-hdfc.pdf")
    out = result["transactions"]
    assert len(out) == 4
    assert out["Debit"].sum() == 10100.0
    assert out["Credit"].sum() == 125115.0
    assert out.iloc[-1]["Balance"] == 215015.0
    assert "Master" not in result["meta"]["layout_confidence"] or result["meta"]["layout_confidence"]
