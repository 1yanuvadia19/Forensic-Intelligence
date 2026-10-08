"""Master PDF intelligence orchestrator with dual-mode extraction.

This is an ensemble system that routes to optimal extractors:
1. Bank Digital PDF → native coordinate-aware extraction
2. Scanned/Photographed → OCR + layout reconstruction

Both candidates are scored and validated through forensic gates.
No transaction is output without evidence-backed validation.
"""

import re

import numpy as np
import pandas as pd

from master_ledger_solver import solve_ledger


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

    core = __import__("forensic_core")
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


def _normalise_candidate(df):
    if df is None or df.empty:
        return pd.DataFrame()

    x = df.copy()
    for col in ["Debit", "Credit", "Balance"]:
        if col not in x:
            x[col] = np.nan
        x[col] = pd.to_numeric(x[col], errors="coerce")

    if "Date" not in x:
        return pd.DataFrame()

    x["Date"] = pd.to_datetime(x["Date"], errors="coerce")
    x = x[x["Date"].notna()].copy()

    for col in ["Narration", "Reference", "Source_Text"]:
        if col not in x:
            x[col] = ""
        x[col] = x[col].fillna("").astype(str)

    if "Source_Page" not in x:
        x["Source_Page"] = np.nan

    dual = x["Debit"].notna() & x["Credit"].notna()
    if dual.any():
        x = __import__("forensic_core")._repair_pdf_side_mapping(x)

    return x.reset_index(drop=True)


def _period_filter(df, start, end):
    if df.empty or start is None or end is None:
        return df
    return df[(df["Date"] >= start) & (df["Date"] <= end)].copy()


def _dedupe_candidate(df):
    if df.empty:
        return df
    return df.drop_duplicates(
        subset=["Date", "Narration", "Debit", "Credit", "Balance"],
        keep="first",
    ).reset_index(drop=True)


def _merge_supported_rows(candidates):
    """Conservative union: only rows with strong identity evidence."""
    rows = []
    seen = set()

    for item in sorted(candidates, key=lambda z: z["score"], reverse=True):
        df = item["df"]
        for _, r in df.iterrows():
            date_key = pd.Timestamp(r["Date"]).strftime("%Y-%m-%d")
            debit = "" if pd.isna(r["Debit"]) else f"D{float(r['Debit']):.2f}"
            credit = "" if pd.isna(r["Credit"]) else f"C{float(r['Credit']):.2f}"
            balance = "" if pd.isna(r["Balance"]) else f"B{float(r['Balance']):.2f}"
            key = (date_key, debit, credit, balance)
            if key in seen:
                continue
            seen.add(key)
            rows.append(r.to_dict())

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["Date", "Source_Page"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


def _is_acceptably_valid(stats):
    """Check if extraction meets minimum quality thresholds.
    
    For scanned PDFs, we are lenient on balance but strict on movement.
    For digital PDFs, we expect both.
    """
    if stats["rows"] < 10:
        return False, "Too few rows"

    if stats["movement_rate"] < 0.95:
        return False, f"Only {stats['movement_rate']:.0%} rows have movement"

    if stats["balance_rate"] < 0.40:
        return False, f"Only {stats['balance_rate']:.0%} have balance"

    max_acceptable_mismatches = max(5, int(stats["rows"] * 0.25))
    if stats["mismatches"] > max_acceptable_mismatches:
        return False, f"{stats['mismatches']} balance mismatches (max: {max_acceptable_mismatches})"

    return True, "Acceptable quality thresholds met"


def extract_master_pdf(data, name, progress_callback=None, pdf_mode=None):
    """Extract PDF with dual-mode routing and ensemble validation.
    
    Intelligently routes between:
    - Digital PDF path: native coordinate extraction
    - Scanned PDF path: OCR + layout reconstruction
    
    Both paths converge through forensic validation gates.
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

    if progress_callback:
        mode_label = "Scanned" if "Scanned" in str(pdf_mode) else "Digital"
        progress_callback(3, 10, f"🔍 Auto-detected: {mode_label} PDF")

    statement_period = re.search(
        r"period of\s+(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\s+to\s+"
        r"(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})",
        full_text,
        re.I,
    )
    period_start = period_end = None
    if statement_period:
        period_start = pd.to_datetime(
            statement_period.group(1).replace(".", "-").replace("/", "-"),
            dayfirst=True, errors="coerce"
        )
        period_end = pd.to_datetime(
            statement_period.group(2).replace(".", "-").replace("/", "-"),
            dayfirst=True, errors="coerce"
        )

    source_opening = core._extract_opening_balance_from_text(full_text)
    source_summary = core._extract_statement_summary(full_text)

    candidates = []

    def add(label, frame):
        frame = _normalise_candidate(frame)
        frame = _period_filter(frame, period_start, period_end)
        frame = _dedupe_candidate(frame)
        if frame.empty:
            return

        frame, ledger_diag = solve_ledger(
            frame,
            opening_balance=source_opening,
            summary=source_summary,
        )
        frame = _dedupe_candidate(frame)
        stats = _candidate_stats(frame)
        candidates.append({
            "method": label,
            "df": frame,
            "stats": stats,
            "score": _quality_score(stats),
            "ledger_diag": ledger_diag,
        })

    if progress_callback:
        progress_callback(4, 10, "🔄 Building extraction ensemble...")

    explicit_scanned = "scanned" in str(pdf_mode).lower() or "photographed" in str(pdf_mode).lower()
    explicit_digital = "digital" in str(pdf_mode).lower() or "bank pdf" in str(pdf_mode).lower()

    # Do not run the expensive 3x OCR pass merely because a digital PDF has a
    # weak/partial text layer. Respect the user's explicit mode first. Native
    # extraction is cheaper and more precise for real digital PDFs; OCR becomes
    # a fallback when native candidates cannot be proven.
    run_scanned_first = explicit_scanned or (not explicit_digital and text_ratio < 0.7)

    if run_scanned_first:
        try:
            if progress_callback:
                progress_callback(5, 10, "📸 OCR: Scanning pages...")
            def _scanned_progress(done, total, message):
                # The scanned engine reports its own page progress. Do not pass
                # that raw 0..N value to Streamlit's global pipeline progress:
                # otherwise page 6/6 incorrectly renders as 100% while the
                # master ensemble still has several extraction/validation stages.
                if progress_callback:
                    pages_done = max(float(done), 0.0)
                    pages_total = max(float(total or 1), 1.0)
                    global_done = 5.0 + (pages_done / pages_total)
                    progress_callback(
                        global_done,
                        10,
                        message,
                    )

            scanned_df, scanned_meta = extract_scanned_statement(
                data, progress_callback=_scanned_progress
            )
            add("Scanned statement OCR grid", scanned_df)
        except Exception:
            scanned_meta = []

    try:
        if progress_callback:
            progress_callback(6, 10, "📄 Native: Coordinate extraction...")
        add(
            "Bank-format-flex coordinate engine",
            extract_bank_pdf(data, progress_callback=progress_callback),
        )
    except Exception:
        pass

    try:
        if progress_callback:
            progress_callback(6.5, 10, "🔙 Legacy: Position rows...")
        add(
            "Legacy coordinate engine",
            core._native_pdf_position_rows(data, progress_callback=progress_callback),
        )
    except Exception:
        pass

    try:
        if progress_callback:
            progress_callback(7, 10, "📊 Table: Reconstruction...")
        add("PDF table reconstruction", core._native_pdf_tables(data))
    except Exception:
        pass

    # If the user selected Digital but the native/text layer produced no
    # trustworthy candidate, perform OCR as a deliberate fallback. This avoids
    # paying the OCR cost on every digital PDF while preserving hybrid-PDF support.
    native_proven = any(
        x["stats"]["rows"] >= 10
        and x["stats"]["movement_rate"] >= 0.95
        and x["stats"]["balance_rate"] >= 0.95
        and x["stats"]["mismatches"] == 0
        for x in candidates
    )
    if explicit_digital and not native_proven:
        try:
            if progress_callback:
                progress_callback(5, 10, "📸 Native extraction insufficient — OCR fallback...")
            def _fallback_scanned_progress(done, total, message):
                if progress_callback:
                    pages_done = max(float(done), 0.0)
                    pages_total = max(float(total or 1), 1.0)
                    progress_callback(
                        5.0 + (pages_done / pages_total),
                        10,
                        message,
                    )
            scanned_df, scanned_meta = extract_scanned_statement(
                data, progress_callback=_fallback_scanned_progress
            )
            add("Scanned statement OCR fallback", scanned_df)
        except Exception:
            pass

    # OCR consensus is expensive. For an explicitly digital PDF, once a
    # native candidate has independently proven the ledger (high movement +
    # high balance coverage + zero running-balance mismatches), STOP adding
    # redundant OCR work. This is the production fast path: proof beats
    # unnecessary ensemble fan-out.
    if not (explicit_digital and native_proven):
        try:
            if progress_callback:
                progress_callback(7.5, 10, "🤖 OCR: Consensus fallback...")
            ocr_df, _, ocr_method = core._ocr_consensus_extract(
                data, progress_callback=progress_callback
            )
            add(ocr_method, ocr_df)
        except Exception:
            pass
    elif progress_callback:
        progress_callback(7.5, 10, "✓ Native ledger proven — OCR consensus skipped")

    if not candidates:
        raise ValueError(
            "No extraction candidates found. PDF may be encrypted, unreadable, or lack transaction data."
        )

    validated = []
    for item in candidates:
        try:
            frame = item["df"].copy()
            frame = core._sort_transaction_evidence(frame)
            frame = core._repair_movement_from_balance(frame)
            frame = core._repair_pdf_side_mapping(frame)

            is_acceptable, reason = _is_acceptably_valid(item["stats"])
            if not is_acceptable:
                continue

            integrity = core.validate_transaction_integrity(frame, item["method"])

            opening_anchor = core._opening_anchor_check(frame, source_opening)
            summary_anchor = core._summary_anchor_check(frame, source_summary)

            # An explicit opening-anchor conflict is evidence that must remain
            # visible, but it must not automatically discard a ledger that is
            # independently proven by its own running-balance chain. In older
            # bank PDFs the "opening balance" text can be outside the transaction
            # table or be extracted from a different statement section. The
            # correct forensic behaviour is REVIEW, not silent rejection.
            anchor_conflict = bool(
                opening_anchor.get("available")
                and not opening_anchor.get("match")
            )
            running_chain_proven = bool(
                item["stats"]["rows"] >= 10
                and item["stats"]["movement_rate"] >= 0.95
                and item["stats"]["balance_rate"] >= 0.95
                and item["stats"]["mismatches"] == 0
            )

            if summary_anchor["available"] and not summary_anchor["match"]:
                continue

            if anchor_conflict and not running_chain_proven:
                continue

            item["df"] = frame
            item["integrity"] = integrity
            item["opening_anchor"] = opening_anchor
            item["opening_anchor"]["status"] = (
                "CONFLICTING" if anchor_conflict else
                "VERIFIED" if opening_anchor.get("available") else
                "NOT_PRESENT"
            )
            item["opening_anchor"]["requires_review"] = anchor_conflict
            item["summary_anchor"] = summary_anchor
            validated.append(item)
        except Exception:
            continue

    if validated:
        best = max(validated, key=lambda x: x["score"])
    else:
        merged = _merge_supported_rows(candidates[: min(4, len(candidates))])
        merged = core._sort_transaction_evidence(merged)
        merged = core._repair_movement_from_balance(merged)
        merged = core._repair_pdf_side_mapping(merged)

        try:
            merged_stats = _candidate_stats(merged)
            is_acceptable, reason = _is_acceptably_valid(merged_stats)
            if not is_acceptable:
                raise ValueError(f"Merged candidate: {reason}")

            integrity = core.validate_transaction_integrity(
                merged, "Master ensemble consensus"
            )
            opening_anchor = core._opening_anchor_check(merged, source_opening)
            summary_anchor = core._summary_anchor_check(merged, source_summary)

            opening_conflict = bool(
                opening_anchor.get("available") and not opening_anchor.get("match")
            )
            merged_chain_proven = bool(
                merged_stats["rows"] >= 10
                and merged_stats["movement_rate"] >= 0.95
                and merged_stats["balance_rate"] >= 0.95
                and merged_stats["mismatches"] == 0
            )
            if opening_conflict and not merged_chain_proven:
                raise ValueError("opening anchor conflict without independent running-balance proof")
            if summary_anchor["available"] and not summary_anchor["match"]:
                raise ValueError("summary anchor mismatch")
            opening_anchor["status"] = (
                "CONFLICTING" if opening_conflict else
                "VERIFIED" if opening_anchor.get("available") else
                "NOT_PRESENT"
            )
            opening_anchor["requires_review"] = opening_conflict

            best = {
                "method": "Master ensemble consensus",
                "df": merged,
                "integrity": integrity,
                "opening_anchor": opening_anchor,
                "summary_anchor": summary_anchor,
                "stats": merged_stats,
                "score": _quality_score(merged_stats),
            }
        except Exception as e:
            ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
            diagnostics = "; ".join(
                f"{x['method']}: rows={x['stats']['rows']}, "
                f"movement={x['stats']['movement_rate']:.0%}, "
                f"balance={x['stats']['balance_rate']:.0%}, "
                f"mismatches={x['stats']['mismatches']}, "
                f"opening={x.get('opening_anchor', {}).get('status', 'UNTESTED')}"
                for x in ranked[:4]
            )
            raise ValueError(
                f"Extraction could not produce acceptable results. Candidates: {diagnostics}. Error: {str(e)}"
            )

    df = best["df"].copy()

    source_has_balance = bool(
        re.search(r"\bbalance\b|closing\s+balance", full_text, re.I)
    )
    balance_presence = float(df["Balance"].notna().mean()) if len(df) else 0.0

    is_scanned = text_ratio < 0.5
    min_balance_threshold = 0.40 if is_scanned else 0.90

    if source_has_balance and balance_presence < min_balance_threshold:
        raise ValueError(
            f"Only {balance_presence:.0%} of {len(df)} rows carry a balance. "
            f"For {'scanned' if is_scanned else 'digital'} PDFs, minimum is {min_balance_threshold:.0%}."
        )

    df = core._enforce_single_date_schema(core.enrich(df))
    flags = df[df.Priority.isin(["REVIEW", "CRITICAL"])].copy()

    meta = {
        "source_type": "PDF — Scanned/OCR" if text_ratio < 0.5 else "PDF — Digital native",
        "location": f"{pages} page(s)",
        "layout_confidence": best["method"],
        "pdf_mode_detected": "Scanned" if text_ratio < 0.7 else "Digital",
        "warnings": [
            f"✓ Extraction method: {best['method']}",
            f"✓ Ensemble evaluated {len(candidates)} candidate(s)",
            f"✓ Selected extraction: {len(df):,} transactions",
            f"✓ Balance coverage: {balance_presence:.0%}",
            f"✓ Source pages tracked for every transaction",
        ],
    }

    return {"transactions": df, "flags": flags, "meta": meta}
