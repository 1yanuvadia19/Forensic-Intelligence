"""Master PDF intelligence with proper anchor diagnostics.

This is the FIXED version that correctly handles opening-anchor validation.

Key changes:
1. Uses diagnostic opening-anchor check
2. Never rejects a candidate merely because opening balance is NOT_PRESENT
3. Provides candidate evaluation table for UI
4. Logs why each candidate was rejected
"""

import re
import numpy as np
import pandas as pd
from master_ledger_solver import solve_ledger
from forensic_core_anchor_fix import (
    _extract_opening_balance_from_text_diagnostic,
    _opening_anchor_check_diagnostic,
    should_reject_candidate_for_opening_anchor,
)


def _candidate_stats(df):
    if df is None or df.empty:
        return {
            "rows": 0, "movement_rate": 0.0, "balance_rate": 0.0,
            "mismatches": 999999, "duplicates": 0, "date_reversals": 999999,
        }

    x = df.copy().reset_index(drop=True)
    for c in ["Date", "Debit", "Credit", "Balance"]:
        if c == "Date":
            x[c] = pd.to_datetime(x[c], errors="coerce")
        else:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    movement = x[["Debit", "Credit"]].notna().any(axis=1)
    movement_rate = float(movement.mean()) if len(x) else 0.0
    balance_rate = float(x["Balance"].notna().mean()) if len(x) else 0.0

    import forensic_core as core
    check = core.balance_check(x)
    mismatches = int((check["Status"] == "MISMATCH").sum()) if not check.empty else 0
    dates = x["Date"].dropna()
    reversals = int((dates.diff().dt.days < 0).sum()) if len(dates) else 999999

    duplicate_keys = x.duplicated(
        subset=["Date", "Narration", "Debit", "Credit", "Balance"],
        keep=False,
    )
    return {
        "rows": len(x),
        "movement_rate": movement_rate,
        "balance_rate": balance_rate,
        "mismatches": mismatches,
        "duplicates": int(duplicate_keys.sum()),
        "date_reversals": reversals,
    }


def _quality_score(stats):
    return (
        min(stats["rows"], 10000) * 0.25
        + stats["movement_rate"] * 100
        + stats["balance_rate"] * 70
        - stats["mismatches"] * 35
        - stats["duplicates"] * 2
        - stats["date_reversals"] * 10
    )


def extract_master_pdf_with_diagnostics(data, name, progress_callback=None, pdf_mode=None):
    """Extract PDF with proper opening-anchor validation.
    
    Returns:
        {
            "transactions": df,
            "flags": df with priority issues,
            "meta": metadata,
            "candidates_evaluated": list of candidate diagnostics,
        }
    """
    import fitz
    import forensic_core as core
    from bank_pdf_engine import extract_bank_pdf
    from scanned_statement_engine import extract_scanned_statement

    doc = fitz.open(stream=data, filetype="pdf")
    pages = len(doc)
    texts = [p.get_text("text") for p in doc]
    text_ratio = sum(bool(t.strip()) for t in texts) / pages if pages else 0.0
    full_text = "\n".join(texts)

    if pdf_mode is None or "auto" in str(pdf_mode).lower():
        pdf_mode = "📸 Scanned/Photographed" if text_ratio < 0.5 else "🏦 Bank Digital PDF"

    # Extract opening balance with diagnostics
    opening_balance_info = _extract_opening_balance_from_text_diagnostic(full_text)
    source_summary = core._extract_statement_summary(full_text)

    if progress_callback:
        opening_status = opening_balance_info.get("source_type", "NOT_FOUND")
        progress_callback(3, 10, f"🔍 Opening balance status: {opening_status}")

    candidates = []
    candidates_evaluated = []

    def add(label, frame):
        frame = _normalise_candidate(frame)
        if frame.empty:
            return

        frame, ledger_diag = solve_ledger(frame, opening_balance=opening_balance_info.get("amount"), summary=source_summary)
        frame = _dedupe_candidate(frame)
        stats = _candidate_stats(frame)
        candidates.append({
            "method": label,
            "df": frame,
            "stats": stats,
            "score": _quality_score(stats),
            "ledger_diag": ledger_diag,
        })

    # Build ensemble (extract all candidates)
    # ... [extraction code same as before] ...

    # Evaluate candidates with diagnostics
    for item in candidates:
        frame = item["df"].copy()
        frame = core._sort_transaction_evidence(frame)
        frame = core._repair_movement_from_balance(frame)
        frame = core._repair_pdf_side_mapping(frame)

        # Get diagnostic anchor info
        anchor = _opening_anchor_check_diagnostic(frame, opening_balance_info)
        summary_anchor = core._summary_anchor_check(frame, source_summary)

        # Evaluate rejection reason
        should_reject, rejection_reason = should_reject_candidate_for_opening_anchor(anchor, item["stats"])

        candidates_evaluated.append({
            "method": item["method"],
            "rows": item["stats"]["rows"],
            "movement": f"{item['stats']['movement_rate']:.0%}",
            "balance": f"{item['stats']['balance_rate']:.0%}",
            "mismatches": item["stats"]["mismatches"],
            "opening_anchor": anchor["opening_anchor_status"],
            "opening_detected": opening_balance_info.get("detected", False),
            "opening_conflict": anchor["difference"] if pd.notna(anchor.get("difference")) else None,
            "rejected": should_reject,
            "rejection_reason": rejection_reason,
        })

        if should_reject:
            continue
        if summary_anchor.get("available") and not summary_anchor.get("match"):
            continue

        # Candidate passed
        item["df"] = frame
        item["opening_anchor"] = anchor
        item["summary_anchor"] = summary_anchor
        candidates_evaluated[-1]["accepted"] = True

    # Select best accepted candidate
    accepted = [c for c in candidates if c.get("opening_anchor") is not None]

    if not accepted:
        # Report diagnostics instead of generic error
        diag_lines = ["Extraction failed. Candidate evaluation:"]
        for c in candidates_evaluated:
            diag_lines.append(
                f"  {c['method']}: rows={c['rows']}, "
                f"movement={c['movement']}, balance={c['balance']}, "
                f"mismatches={c['mismatches']}, opening={c['opening_anchor']} "
                f"→ {c['rejection_reason']}"
            )
        raise ValueError("\n".join(diag_lines))

    best = max(accepted, key=lambda x: x["score"])
    df = best["df"].copy()

    df = core._enforce_single_date_schema(core.enrich(df))
    flags = df[df.Priority.isin(["REVIEW", "CRITICAL"])].copy()

    return {
        "transactions": df,
        "flags": flags,
        "meta": {
            "source_type": "PDF — Scanned/OCR" if text_ratio < 0.5 else "PDF — Digital native",
            "layout_confidence": best["method"],
            "opening_balance_status": opening_balance_info.get("source_type", "NOT_FOUND"),
            "opening_balance_detected": opening_balance_info.get("detected", False),
        },
        "candidates_evaluated": candidates_evaluated,
    }


def _normalise_candidate(df):
    # [same as before]
    pass

def _dedupe_candidate(df):
    # [same as before]
    pass
