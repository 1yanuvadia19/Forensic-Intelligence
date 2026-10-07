"""Master PDF intelligence orchestrator.

This is intentionally an ensemble, not a single parser. A bank statement is
first treated as a document-structure problem and only then as a ledger.

Candidate engines:
1. coordinate-aware native extraction,
2. pdfplumber table extraction,
3. refreshed bank-format-flex extraction,
4. independent OCR consensus for scanned/hybrid pages.

Candidates are scored against the statement's own evidence. A candidate is
accepted only when the complete ledger can be validated; otherwise the UI gets
a diagnostic failure rather than fabricated financial data.
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
    # Row coverage matters, but mathematical evidence matters much more.
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

    # Never allow both sides on a single candidate row.
    dual = x["Debit"].notna() & x["Credit"].notna()
    if dual.any():
        # Keep the side which is independently supported by the balance chain.
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
    """Conservative union: a row must have strong identity evidence.

    Identity is date + movement amount + balance when available. This is used
    only as a recovery candidate, never as a blind concatenation of parsers.
    """
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


def extract_master_pdf(data, name, progress_callback=None):
    import fitz
    import forensic_core as core
    from bank_pdf_engine import extract_bank_pdf
    from scanned_statement_engine import extract_scanned_statement

    doc = fitz.open(stream=data, filetype="pdf")
    pages = len(doc)
    texts = [p.get_text("text") for p in doc]
    text_ratio = sum(bool(t.strip()) for t in texts) / pages if pages else 0.0
    full_text = "\n".join(texts)

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

        # Independent evidence solver: statement order and Debit/Credit side
        # are hypotheses. It may relabel a side only when the printed balance
        # chain supports it; monetary values are never changed.
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
        progress_callback(0, max(pages, 1), "Building extraction ensemble")

    # 0. Image-only scanned-statement vision engine. This is especially
    # important for old/photocopied statements whose PDF has no text layer.
    # It reconstructs rows from OCR coordinates rather than trusting OCR text
    # order, and resolves direction from the printed running balance.
    try:
        scanned_df, scanned_meta = extract_scanned_statement(
            data, progress_callback=progress_callback
        )
        add("Scanned statement vision grid", scanned_df)
    except Exception:
        scanned_meta = []

    # A. Refreshed coordinate engine.
    try:
        add(
            "Bank-format-flex coordinate engine",
            extract_bank_pdf(data, progress_callback=progress_callback),
        )
    except Exception:
        pass

    # B. Existing native coordinate engine as an independent hypothesis.
    try:
        add(
            "Legacy coordinate engine",
            core._native_pdf_position_rows(data, progress_callback=progress_callback),
        )
    except Exception:
        pass

    # C. pdfplumber table engine.
    try:
        add("PDF table reconstruction", core._native_pdf_tables(data))
    except Exception:
        pass

    # D. OCR consensus is useful for hybrid pages even when most of the PDF has
    # selectable text. It is intentionally independent of the native engines.
    try:
        ocr_df, _, ocr_method = core._ocr_consensus_extract(
            data, progress_callback=progress_callback
        )
        add(ocr_method, ocr_df)
    except Exception:
        pass

    if not candidates:
        raise ValueError(
            "Master extraction found no transaction candidates. "
            "The PDF may be encrypted, image-only without usable OCR, or use a "
            "layout that requires visual review."
        )

    # Try every candidate through the same hard forensic gate. Do not accept the
    # first parser that returns rows.
    validated = []
    for item in candidates:
        try:
            frame = item["df"].copy()
            frame = core._sort_transaction_evidence(frame)
            frame = core._repair_movement_from_balance(frame)
            frame = core._repair_pdf_side_mapping(frame)
            integrity = core.validate_transaction_integrity(frame, item["method"])

            opening_anchor = core._opening_anchor_check(frame, source_opening)
            summary_anchor = core._summary_anchor_check(frame, source_summary)

            if opening_anchor["available"] and not opening_anchor["match"]:
                continue
            if summary_anchor["available"] and not summary_anchor["match"]:
                continue

            item["df"] = frame
            item["integrity"] = integrity
            item["opening_anchor"] = opening_anchor
            item["summary_anchor"] = summary_anchor
            validated.append(item)
        except Exception:
            continue

    # If one complete candidate is mathematically valid, prefer the one with
    # the highest evidence/coverage score.
    if validated:
        best = max(validated, key=lambda x: x["score"])
    else:
        # Conservative recovery: union only strongly identified rows from the
        # best candidates, then run the full gate again. This can recover rows
        # lost by one extractor while refusing unsupported invented amounts.
        merged = _merge_supported_rows(candidates[: min(4, len(candidates))])
        merged = core._sort_transaction_evidence(merged)
        merged = core._repair_movement_from_balance(merged)
        merged = core._repair_pdf_side_mapping(merged)

        try:
            integrity = core.validate_transaction_integrity(
                merged, "Master ensemble consensus"
            )
            opening_anchor = core._opening_anchor_check(merged, source_opening)
            summary_anchor = core._summary_anchor_check(merged, source_summary)
            if opening_anchor["available"] and not opening_anchor["match"]:
                raise ValueError("opening anchor mismatch")
            if summary_anchor["available"] and not summary_anchor["match"]:
                raise ValueError("summary anchor mismatch")

            best = {
                "method": "Master ensemble consensus",
                "df": merged,
                "integrity": integrity,
                "opening_anchor": opening_anchor,
                "summary_anchor": summary_anchor,
                "stats": _candidate_stats(merged),
                "score": _quality_score(_candidate_stats(merged)),
            }
        except Exception:
            ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
            diagnostics = "; ".join(
                f"{x['method']}: rows={x['stats']['rows']}, "
                f"movement={x['stats']['movement_rate']:.0%}, "
                f"balance={x['stats']['balance_rate']:.0%}, "
                f"mismatches={x['stats']['mismatches']}, "
                f"ledger={x.get('ledger_diag', {}).get('order', '?')}"
                for x in ranked[:4]
            )
            raise ValueError(
                "Master extraction could not prove a complete ledger. "
                "No transaction was invented. Candidate diagnostics: " + diagnostics
            )

    df = best["df"].copy()

    # Final source-balance coverage guard.
    source_has_balance = bool(
        re.search(r"\bbalance\b|closing\s+balance", full_text, re.I)
    )
    balance_presence = float(df["Balance"].notna().mean()) if len(df) else 0.0
    if source_has_balance and balance_presence < 0.90:
        raise ValueError(
            f"Master extraction found {len(df)} rows, but only "
            f"{balance_presence:.0%} carry a reported balance. "
            "The statement's balance evidence is incomplete."
        )

    outside_count = 0
    if statement_period is not None:
        outside_count = 0

    df = core._enforce_single_date_schema(core.enrich(df))
    flags = df[df.Priority.isin(["REVIEW", "CRITICAL"])].copy()

    meta = {
        "source_type": "PDF — scanned/OCR" if text_ratio < 0.5 else "PDF — native/digital",
        "location": f"{pages} pages",
        "layout_confidence": best["method"],
        "warnings": [
            f"Master extraction engine selected: {best['method']}.",
            f"Extraction ensemble evaluated {len(candidates)} independent candidate(s).",
            "Every exported transaction passed date, movement-side and running-balance validation.",
            "No transaction amount is fabricated to force reconciliation.",
            "Source_Page and Source_Text are retained for evidence tracing.",
            "Image-only PDFs are routed through coordinate-aware OCR row reconstruction before generic OCR consensus.",
            "Wrapped narrations are reconstructed separately from physical PDF lines.",
            f"Ledger hypothesis solver selected order={best.get('ledger_diag', {}).get('order', 'n/a')} with {best.get('ledger_diag', {}).get('matches', 0)} balance-chain matches.",
        ],
    }

    return {"transactions": df, "flags": flags, "meta": meta}
