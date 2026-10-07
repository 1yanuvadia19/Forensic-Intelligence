import numpy as np
import pandas as pd

from forensic_core import (
    money,
    rail,
    category,
    counterparty,
    semantic_narration,
    _opening_anchor_check,
    _summary_anchor_check,
    _normalize_reference_fields,
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
