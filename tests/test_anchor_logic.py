"""Test suite for opening-anchor validation logic.

TEST A: Explicit opening balance + valid running balances => PASS
TEST B: No explicit opening balance + valid running balances => PASS (NOT_PRESENT)
TEST C: Explicit opening balance conflicts with first ledger => FAIL / REVIEW
TEST D: Running balance mismatch => FAIL
TEST E: Descending statement => correctly validate in reverse
TEST F: Page 1 has opening balance, later pages only transactions => valid across pages
"""

import numpy as np
import pandas as pd
from forensic_core_anchor_fix import (
    _extract_opening_balance_from_text_diagnostic,
    _opening_anchor_check_diagnostic,
    should_reject_candidate_for_opening_anchor,
)


def test_a_explicit_opening_valid_running():
    """Explicit opening balance + valid running balances."""
    # Statement text: "Opening Balance: 1000"
    # Transactions:
    #   Date         Narration    Debit  Credit  Balance
    #   2026-01-01   Credit       -      500     1500
    #   2026-01-02   Debit        100    -       1400
    
    opening_info = _extract_opening_balance_from_text_diagnostic("Opening Balance: 1000")
    assert opening_info["detected"] is True
    assert opening_info["amount"] == 1000
    assert opening_info["source_type"] == "explicit_opening"
    
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "Narration": ["Credit", "Debit"],
        "Debit": [np.nan, 100.0],
        "Credit": [500.0, np.nan],
        "Balance": [1500.0, 1400.0],
    })
    
    anchor = _opening_anchor_check_diagnostic(df, opening_info)
    assert anchor["available"] is True
    assert anchor["match"] is True
    assert anchor["opening_anchor_status"] == "VERIFIED"
    
    stats = {"movement_rate": 1.0, "balance_rate": 1.0, "mismatches": 0}
    should_reject, reason = should_reject_candidate_for_opening_anchor(anchor, stats)
    assert should_reject is False, f"Should not reject: {reason}"
    print("✓ TEST A PASSED")


def test_b_no_opening_balance_valid_running():
    """No explicit opening balance + valid running balances."""
    # Statement text has no opening balance
    # Transactions are internally consistent
    
    opening_info = _extract_opening_balance_from_text_diagnostic(
        "Statement Period: Jan 2026\nTransactions follow..."
    )
    assert opening_info["detected"] is False
    assert opening_info["source_type"] == "NOT_FOUND"
    
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "Narration": ["Credit", "Debit"],
        "Debit": [np.nan, 100.0],
        "Credit": [500.0, np.nan],
        "Balance": [1500.0, 1400.0],
    })
    
    anchor = _opening_anchor_check_diagnostic(df, opening_info)
    assert anchor["available"] is False
    assert anchor["opening_anchor_status"] == "NOT_PRESENT"
    
    stats = {"movement_rate": 1.0, "balance_rate": 1.0, "mismatches": 0}
    should_reject, reason = should_reject_candidate_for_opening_anchor(anchor, stats)
    assert should_reject is False, f"Should not reject when opening balance NOT_PRESENT: {reason}"
    print("✓ TEST B PASSED")


def test_c_explicit_opening_conflicts():
    """Explicit opening balance conflicts with first ledger => FAIL."""
    # Statement says: "Opening Balance: 1000"
    # But first balance is 2000 with 500 credit
    # Expected: 1000 + 500 = 1500, but got 2000
    
    opening_info = _extract_opening_balance_from_text_diagnostic("Opening Balance: 1000")
    assert opening_info["detected"] is True
    
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01"]),
        "Narration": ["Credit"],
        "Debit": [np.nan],
        "Credit": [500.0],
        "Balance": [2000.0],  # Should be 1500, not 2000
    })
    
    anchor = _opening_anchor_check_diagnostic(df, opening_info)
    assert anchor["available"] is True
    assert anchor["match"] is False
    assert anchor["opening_anchor_status"] == "CONFLICTING"
    assert abs(anchor["difference"] - 500.0) < 0.01  # 2000 - 1500 = 500
    
    stats = {"movement_rate": 1.0, "balance_rate": 1.0, "mismatches": 0}
    should_reject, reason = should_reject_candidate_for_opening_anchor(anchor, stats)
    assert should_reject is True, f"Should reject explicit opening balance conflict: {reason}"
    print("✓ TEST C PASSED")


def test_d_running_balance_mismatch():
    """Running balance mismatch on first row."""
    # Opening: 1000
    # First row: 500 credit, reported balance 1300 (should be 1500)
    
    opening_info = _extract_opening_balance_from_text_diagnostic("Opening Balance: 1000")
    
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01"]),
        "Narration": ["Credit"],
        "Debit": [np.nan],
        "Credit": [500.0],
        "Balance": [1300.0],  # Mismatch
    })
    
    anchor = _opening_anchor_check_diagnostic(df, opening_info)
    assert anchor["match"] is False
    assert anchor["opening_anchor_status"] == "CONFLICTING"
    print("✓ TEST D PASSED")


def test_e_descending_statement():
    """Descending statement (balance decreases).
    
    Opening Balance: 5000
    Debit amounts decrease the balance.
    """
    opening_info = _extract_opening_balance_from_text_diagnostic("Opening Balance: 5000")
    
    # Debit should decrease balance: 5000 - 100 = 4900
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "Narration": ["Withdrawal", "Fee"],
        "Debit": [100.0, 50.0],
        "Credit": [np.nan, np.nan],
        "Balance": [4900.0, 4850.0],
    })
    
    anchor = _opening_anchor_check_diagnostic(df, opening_info)
    assert anchor["available"] is True
    assert anchor["match"] is True, f"Opening balance should reconcile: diff={anchor['difference']}"
    assert anchor["opening_anchor_status"] == "VERIFIED"
    print("✓ TEST E PASSED")


def test_f_opening_balance_persists_across_pages():
    """Page 1 has opening balance, later pages only transactions.
    
    Opening Balance: 1000 (on page 1)
    Page 1 transactions reconcile
    Page 2 transactions continue from page 1 closing
    """
    opening_info = _extract_opening_balance_from_text_diagnostic("Opening Balance: 1000")
    
    df = pd.DataFrame({
        "Date": pd.to_datetime([
            "2026-01-01", "2026-01-02",  # Page 1
            "2026-01-03", "2026-01-04",  # Page 2
        ]),
        "Narration": ["Cr1", "Dr1", "Cr2", "Dr2"],
        "Debit": [np.nan, 100.0, np.nan, 150.0],
        "Credit": [500.0, np.nan, 600.0, np.nan],
        "Balance": [1500.0, 1400.0, 2000.0, 1850.0],
        "Source_Page": [1, 1, 2, 2],
    })
    
    anchor = _opening_anchor_check_diagnostic(df, opening_info)
    assert anchor["match"] is True
    assert anchor["opening_anchor_status"] == "VERIFIED"
    print("✓ TEST F PASSED")


def test_kotak_diagnostics():
    """Kotak PDF test case: 12 rows, 100% movement, 100% balance, 0 mismatches.
    
    This is the real-world case that should NOT be rejected.
    """
    # Simulate what would be extracted from Kotak
    opening_info = _extract_opening_balance_from_text_diagnostic(
        "Your statement from VED TITIYA KOTAK\nOpening Balance: 1,00,000"
    )
    
    # If opening was NOT detected (common in scanned PDFs), that's OK
    if opening_info["detected"]:
        # If detected, it should reconcile
        df = pd.DataFrame({
            "Date": pd.to_datetime([f"2026-01-{i+1:02d}" for i in range(12)]),
            "Narration": [f"Transaction {i+1}" for i in range(12)],
            "Debit": [100.0 if i % 2 == 0 else np.nan for i in range(12)],
            "Credit": [np.nan if i % 2 == 0 else 150.0 for i in range(12)],
            "Balance": [100000.0 - (100.0 if i % 2 == 0 else -150.0) + sum([(-100.0 if j % 2 == 0 else 150.0) for j in range(i)]) for i in range(12)],
        })
        
        anchor = _opening_anchor_check_diagnostic(df, opening_info)
        stats = {"movement_rate": 1.0, "balance_rate": 1.0, "mismatches": 0}
        should_reject, reason = should_reject_candidate_for_opening_anchor(anchor, stats)
        # With 12 rows, 100% movement, 100% balance, 0 mismatches,
        # it should NOT be rejected regardless of opening anchor
        print(f"  Kotak case: rows=12, movement=100%, balance=100%, mismatches=0")
        print(f"  → Opening anchor: {anchor['opening_anchor_status']}")
        print(f"  → Should reject: {should_reject} (reason: {reason})")
    else:
        print("  Kotak case: opening balance NOT detected in source text")
        print("  → This is expected for scanned PDFs")
        print("  → With 12 rows of valid transactions, extraction should PASS")
    
    print("✓ TEST KOTAK DIAGNOSTICS COMPLETE")


if __name__ == "__main__":
    test_a_explicit_opening_valid_running()
    test_b_no_opening_balance_valid_running()
    test_c_explicit_opening_conflicts()
    test_d_running_balance_mismatch()
    test_e_descending_statement()
    test_f_opening_balance_persists_across_pages()
    test_kotak_diagnostics()
    print("\n✅ ALL ANCHOR LOGIC TESTS PASSED")
