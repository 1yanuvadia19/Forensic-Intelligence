"""Forensic anchor validation - CORRECTED LOGIC.

This file contains the fixed opening/closing anchor logic that:
1. Distinguishes between explicit opening balances vs. NOT_PRESENT
2. Does not reject a candidate merely because an opening anchor is unavailable
3. Validates only when an explicit opening balance is actually present in source
4. Provides diagnostic information about what was checked
"""

import re
import numpy as np
import pandas as pd


def _extract_opening_balance_from_text_diagnostic(text):
    """
    Extract explicit opening balance and return diagnostic info.
    
    Returns:
        {
            "amount": float or np.nan,
            "raw_text": str or None,
            "source_type": "explicit_opening" | "brought_forward" | "b/f" | "NOT_FOUND",
            "detected": bool,
        }
    """
    if not text:
        return {
            "amount": np.nan,
            "raw_text": None,
            "source_type": "NOT_FOUND",
            "detected": False,
        }

    patterns = [
        (r"(?is)\bopening\s+balance\b\s*[:\-]?\s*([\d,]+(?:\.\d+)?)\s*\(?\s*(CR|DR)?\s*\)?", "explicit_opening"),
        (r"(?is)\bbrought\s+forward\b\s*[:\-]?\s*([\d,]+(?:\.\d+)?)\s*\(?\s*(CR|DR)?\s*\)?", "brought_forward"),
        (r"(?is)\bbalance\s+b\s*/\s*f\b\s*[:\-]?\s*([\d,]+(?:\.\d+)?)\s*\(?\s*(CR|DR)?\s*\)?", "b/f"),
        (r"(?is)\bb\s*/\s*f\b\s*([\d,]+(?:\.\d+)?)\s*\(?\s*(CR|DR)?\s*\)?", "b/f"),
    ]

    from forensic_core import money

    for pattern, source_type in patterns:
        m = re.search(pattern, text)
        if not m:
            continue
        value = money(m.group(1))
        if pd.isna(value):
            continue
        side = (m.group(2) or "").upper()
        amount = -abs(float(value)) if side == "DR" else abs(float(value))
        return {
            "amount": amount,
            "raw_text": m.group(0),
            "source_type": source_type,
            "detected": True,
        }

    return {
        "amount": np.nan,
        "raw_text": None,
        "source_type": "NOT_FOUND",
        "detected": False,
    }


def _opening_anchor_check_diagnostic(df, opening_balance_info, tolerance=0.01):
    """
    Enhanced opening anchor check with full diagnostics.
    
    Args:
        df: transaction dataframe
        opening_balance_info: dict from _extract_opening_balance_from_text_diagnostic()
        tolerance: reconciliation tolerance
    
    Returns:
        {
            "available": bool,  # opening balance was explicitly present in source
            "detected_in_source": bool,  # whether text extraction found it
            "match": bool,  # extracted ledger reconciles to it
            "difference": float,
            "opening_anchor_status": "VERIFIED" | "NOT_PRESENT" | "AMBIGUOUS" | "CONFLICTING",
            "source_type": "explicit_opening" | "brought_forward" | "b/f" | "NOT_FOUND",
            "extracted_opening_amount": float or np.nan,
            "first_balance": float or np.nan,
            "first_movement": dict,  # {debit, credit}
            "expected_first_balance": float or np.nan,
            "confidence": float,  # 0.0 to 1.0
        }
    """
    # Determine whether an opening balance was present in source
    detected = opening_balance_info.get("detected", False)
    source_amount = opening_balance_info.get("amount", np.nan)
    source_type = opening_balance_info.get("source_type", "NOT_FOUND")

    if df is None or df.empty:
        return {
            "available": detected,
            "detected_in_source": detected,
            "match": not detected,  # True if no opening balance was expected
            "difference": np.nan,
            "opening_anchor_status": "NOT_PRESENT" if not detected else "CONFLICTING",
            "source_type": source_type,
            "extracted_opening_amount": np.nan,
            "first_balance": np.nan,
            "first_movement": {"debit": np.nan, "credit": np.nan},
            "expected_first_balance": np.nan,
            "confidence": 0.0 if detected else 1.0,
        }

    # Extract first row data
    first = df.iloc[0]
    first_balance = pd.to_numeric(first.get("Balance"), errors="coerce")
    first_debit = pd.to_numeric(first.get("Debit"), errors="coerce")
    first_credit = pd.to_numeric(first.get("Credit"), errors="coerce")
    first_page = first.get("Source_Page", np.nan)

    # If no opening balance was detected in source text, no validation is possible
    if not detected or pd.isna(source_amount):
        return {
            "available": False,
            "detected_in_source": False,
            "match": True,  # Not a failure — simply not applicable
            "difference": np.nan,
            "opening_anchor_status": "NOT_PRESENT",
            "source_type": "NOT_FOUND",
            "extracted_opening_amount": np.nan,
            "first_balance": float(first_balance) if pd.notna(first_balance) else np.nan,
            "first_movement": {
                "debit": float(first_debit) if pd.notna(first_debit) else np.nan,
                "credit": float(first_credit) if pd.notna(first_credit) else np.nan,
            },
            "expected_first_balance": np.nan,
            "confidence": 1.0,
        }

    # An opening balance was found in the source. Now validate it.
    # If first row lacks balance or movement, we cannot validate.
    if pd.isna(first_balance) or (pd.isna(first_debit) and pd.isna(first_credit)):
        return {
            "available": True,
            "detected_in_source": True,
            "match": False,  # Cannot prove the opening balance
            "difference": np.nan,
            "opening_anchor_status": "AMBIGUOUS",
            "source_type": source_type,
            "extracted_opening_amount": source_amount,
            "first_balance": float(first_balance) if pd.notna(first_balance) else np.nan,
            "first_movement": {
                "debit": float(first_debit) if pd.notna(first_debit) else np.nan,
                "credit": float(first_credit) if pd.notna(first_credit) else np.nan,
            },
            "expected_first_balance": np.nan,
            "confidence": 0.0,
        }

    # Calculate expected first balance from opening balance + first transaction
    first_debit_val = 0.0 if pd.isna(first_debit) else float(first_debit)
    first_credit_val = 0.0 if pd.isna(first_credit) else float(first_credit)
    expected_first_balance = float(source_amount) + first_credit_val - first_debit_val
    difference = float(first_balance) - expected_first_balance
    matches = abs(difference) <= tolerance

    # Determine confidence: how strong is this opening-balance proof?
    confidence = 1.0 if matches else 0.0

    return {
        "available": True,
        "detected_in_source": True,
        "match": matches,
        "difference": difference,
        "opening_anchor_status": "VERIFIED" if matches else "CONFLICTING",
        "source_type": source_type,
        "extracted_opening_amount": source_amount,
        "first_balance": float(first_balance),
        "first_movement": {
            "debit": first_debit_val,
            "credit": first_credit_val,
        },
        "expected_first_balance": expected_first_balance,
        "confidence": confidence,
    }


def should_reject_candidate_for_opening_anchor(anchor_result, stats):
    """
    Determine if a candidate should be rejected based on opening anchor.
    
    CRITICAL LOGIC:
    - If opening_anchor_status == "NOT_PRESENT": NEVER reject
    - If opening_anchor_status == "VERIFIED": Accept
    - If opening_anchor_status == "CONFLICTING": Reject ONLY if opening balance is truly explicit
    - If opening_anchor_status == "AMBIGUOUS": Accept if stats are strong enough
    """
    status = anchor_result.get("opening_anchor_status", "NOT_PRESENT")
    detected = anchor_result.get("detected_in_source", False)
    source_type = anchor_result.get("source_type", "NOT_FOUND")

    # If no opening balance was present in the source, never reject for this reason
    if status == "NOT_PRESENT" or not detected:
        return False, "Opening balance not present in source; proceeding with running-balance validation"

    # If explicitly detected, reconciliation is mandatory
    if status == "VERIFIED":
        return False, "Opening balance verified"

    # If ambiguous but stats are strong, accept and flag
    if status == "AMBIGUOUS":
        if stats["movement_rate"] >= 0.95 and stats["balance_rate"] >= 0.95:
            return False, "Opening balance ambiguous, but transaction evidence is strong"
        return True, "Opening balance ambiguous and insufficient transaction evidence"

    # If conflicting, only reject if it was explicitly and clearly stated
    if status == "CONFLICTING":
        if source_type in ["explicit_opening", "brought_forward"]:
            return True, f"Explicit opening balance ({source_type}) conflicts with extracted ledger"
        # For weaker matches (b/f, etc), accept if overall stats are good
        if stats["movement_rate"] >= 0.95 and stats["balance_rate"] >= 0.95 and stats["mismatches"] == 0:
            return False, f"Opening anchor conflict on {source_type}, but ledger is internally valid"
        return True, f"Opening balance conflict on {source_type}"

    return False, "Unknown anchor status"
