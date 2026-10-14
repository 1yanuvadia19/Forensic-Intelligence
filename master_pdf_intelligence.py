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



def _extract_pnb_statement_rows(data, progress_callback=None):
    """Coordinate-based parser for PNB's native statement layout.

    PNB prints transaction date and money columns on one baseline while
    narration is a few points above it. Parse by column coordinates rather
    than relying on PDF text-stream order, which interleaves columns.
    """
    import fitz

    doc = fitz.open(stream=data, filetype="pdf")
    rows = []
    money_re = re.compile(r"^[\d,]+\.\d{2}$")
    date_re = re.compile(r"^\d{2}-\d{2}-\d{4}$")

    for page_no, page in enumerate(doc, start=1):
        words = page.get_text("words")
        if not words:
            continue

        # Require the expected PNB column header on at least the first
        # transaction-bearing page; this is not a generic PDF fallback.
        page_text = page.get_text("text")
        has_pnb_columns = all(
            token in page_text.lower()
            for token in ("tran date", "withdrawal", "deposit", "balance", "narration")
        )
        # The first page has a printed column header around y=379. Continuation
        # pages omit the header, so their transactions start near the top margin.
        y_min = 390 if has_pnb_columns else 35
        dates = []
        for word in words:
            x0, y0, x1, y1, token, *_ = word
            if (
                x0 < 90
                and y_min < y0 < page.rect.height - 42
                and date_re.fullmatch(str(token).strip())
            ):
                dates.append((float(y0), str(token).strip()))

        for y, date_text in dates:
            nearby = [
                w for w in words
                if abs(float(w[1]) - y) <= 6.0
                and float(w[1]) > y_min
                and float(w[1]) < page.rect.height - 35
            ]

            debit = credit = balance = None
            cheque_tokens = []
            for word in nearby:
                x0, y0, x1, y1, token, *_ = word
                token = str(token).strip()
                if money_re.fullmatch(token):
                    amount = float(token.replace(",", ""))
                    if 70 <= x0 < 145:
                        debit = amount
                    elif 145 <= x0 < 205:
                        credit = amount
                    elif 205 <= x0 < 260:
                        balance = amount
                elif 250 <= x0 < 320 and token not in {"Alpha", "CHQ.", "NO."}:
                    cheque_tokens.append(token)

            # A transaction must have a printed running balance and one
            # movement amount. Never infer or manufacture a missing amount.
            if balance is None or (debit is None and credit is None):
                continue
            if debit is not None and credit is not None:
                continue

            narration_words = []
            for word in nearby:
                x0, y0, x1, y1, token, *_ = word
                token = str(token).strip()
                if 320 <= x0 < 495 and not money_re.fullmatch(token):
                    if token.lower() not in {"cr.", "dr."}:
                        narration_words.append((float(x0), token))
            narration = " ".join(t for _, t in sorted(narration_words)).strip()

            # Extract a traceable transaction reference where present.
            ref_match = re.search(
                r"\b(?:UPI|IMPS|NEFT|RTGS|ACH|NACH|POS|ATM)[/-][A-Z0-9/-]+",
                narration,
                re.I,
            )
            reference = (
                ref_match.group(0)
                if ref_match
                else " ".join(cheque_tokens).strip()
            )

            rows.append({
                "Date": pd.to_datetime(date_text, dayfirst=True, errors="coerce"),
                "Narration": narration,
                "Reference": reference,
                "Debit": debit,
                "Credit": credit,
                "Balance": balance,
                "Source_Page": page_no,
                "Source_Text": f"PNB native coordinate parser | page {page_no} | {date_text}",
            })

        if progress_callback and page_no % 3 == 0:
            progress_callback(
                5.5 + 0.4 * page_no / max(len(doc), 1),
                10,
                f"PNB native parser: page {page_no}/{len(doc)}",
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result = result[result["Date"].notna()].copy()
    result = result.sort_values(
        ["Source_Page", "Date"], kind="stable"
    ).reset_index(drop=True)

    # Guard the bank-specific route: if the file does not have PNB's printed
    # headers, return no candidate so the existing ensemble can take over.
    first_page_text = doc[0].get_text("text").lower() if len(doc) else ""
    signature = all(x in first_page_text for x in (
        "tran date", "withdrawal", "deposit", "balance", "narration",
        "alpha", "chq.", "no."
    ))
    if not signature or not any(
        all(x in page.get_text("text").lower() for x in ("tran date", "withdrawal", "deposit", "balance", "narration"))
        for page in doc
    ):
        return pd.DataFrame()

    return result


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

        if label == "Legacy coordinate engine" and progress_callback:
            progress_callback(6.98, 10, "Legacy: validating ledger hypotheses...")
        frame, ledger_diag = solve_ledger(
            frame,
            opening_balance=source_opening,
            summary=source_summary,
        )
        if label == "Legacy coordinate engine" and progress_callback:
            progress_callback(7.0, 10, "Legacy: ledger validation complete")
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

    # PNB's electronically generated statement has a stable coordinate layout.
    # Route matching files through a dedicated parser first; the general
    # ensemble remains available whenever the PNB signature does not match.
    pnb_signature = all(token in full_text.lower() for token in (
        "statement of account", "tran date", "withdrawal", "deposit",
        "alpha", "chq.", "no.", "additional", "info"
    ))
    if pnb_signature and not explicit_scanned:
        try:
            if progress_callback:
                progress_callback(5.2, 10, "🏦 PNB layout detected — extracting by source coordinates...")
            pnb_rows = _extract_pnb_statement_rows(data, progress_callback=progress_callback)
            if not pnb_rows.empty:
                add("PNB native coordinate parser", pnb_rows)
                pnb_item = candidates[-1]
                pnb_stats = pnb_item["stats"]
                pnb_proven = (
                    pnb_stats["rows"] >= 10
                    and pnb_stats["movement_rate"] >= 0.99
                    and pnb_stats["balance_rate"] >= 0.99
                    and pnb_stats["mismatches"] == 0
                )
                if pnb_proven and progress_callback:
                    progress_callback(8.2, 10, f"✓ PNB ledger reconciled: {pnb_stats['rows']} rows, zero running-balance mismatches")
        except Exception as exc:
            if progress_callback:
                progress_callback(5.4, 10, f"⚠️ PNB-specific parser failed; trying general engines: {type(exc).__name__}: {exc}")

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
        except Exception as exc:
            scanned_meta = []
            if progress_callback:
                progress_callback(5.9, 10, f"⚠️ Scanned OCR candidate failed: {type(exc).__name__}: {exc}")

    # First native candidate. A PNB-specific candidate that already proves
    # every adjacent balance is stronger evidence than rerunning generic
    # coordinate/table extractors against the same fixed column layout.
    pnb_proven = any(
        x["method"] == "PNB native coordinate parser"
        and x["stats"]["rows"] >= 10
        and x["stats"]["movement_rate"] >= 0.99
        and x["stats"]["balance_rate"] >= 0.99
        and x["stats"]["mismatches"] == 0
        for x in candidates
    )
    try:
        if progress_callback:
            progress_callback(6, 10, "📄 Native: Coordinate extraction...")
        add(
            "Bank-format-flex coordinate engine",
            extract_bank_pdf(data, progress_callback=progress_callback),
        )
    except Exception as exc:
        if progress_callback:
            progress_callback(6.4, 10, f"⚠️ Bank-format coordinate extraction failed: {type(exc).__name__}: {exc}")

    native_proven = any(
        x["stats"]["rows"] >= 10
        and x["stats"]["movement_rate"] >= 0.95
        and x["stats"]["balance_rate"] >= 0.95
        and x["stats"]["mismatches"] == 0
        for x in candidates
    )

    # If the first native/scanned candidate is already mathematically proven,
    # stop the ensemble fan-out. Extra engines are fallbacks, not a requirement.
    if not native_proven:
        try:
            if progress_callback:
                progress_callback(6.5, 10, "🔙 Legacy: Position rows...")
            add(
                "Legacy coordinate engine",
                core._native_pdf_position_rows(data, progress_callback=progress_callback),
            )
        except Exception as exc:
            if progress_callback:
                progress_callback(6.9, 10, f"⚠️ Legacy coordinate extraction failed: {type(exc).__name__}: {exc}")

    native_proven = any(
        x["stats"]["rows"] >= 10
        and x["stats"]["movement_rate"] >= 0.95
        and x["stats"]["balance_rate"] >= 0.95
        and x["stats"]["mismatches"] == 0
        for x in candidates
    )

    if not native_proven:
        try:
            if progress_callback:
                progress_callback(7, 10, "📊 Table: Reconstruction...")
            add(
                "PDF table reconstruction",
                core._native_pdf_tables(data, progress_callback=progress_callback),
            )
        except Exception as exc:
            if progress_callback:
                progress_callback(7.4, 10, f"⚠️ Table reconstruction failed: {type(exc).__name__}: {exc}")

    # Do not run a second, separate scanned-OCR engine here for an explicitly
    # digital PDF. The OCR-consensus fallback below is the single fallback path;
    # running both OCR pipelines serially on a 200-page statement caused long
    # waits without producing a usable result sooner.
    # OCR consensus is extremely expensive on long digital PDFs. Run it when
    # native/table candidates are absent or unusable; do not spend another full
    # multi-pass OCR cycle when an extracted candidate meets the final validator's
    # minimum quality thresholds. Balance conflicts remain visible to validation.
    usable_candidate = any(_is_acceptably_valid(x["stats"])[0] for x in candidates)
    run_ocr_consensus = not explicit_digital or (not native_proven and not usable_candidate)

    if run_ocr_consensus:
        try:
            if progress_callback:
                progress_callback(7.5, 10, "🤖 OCR: Consensus fallback (no usable native candidate)...")
            ocr_df, _, ocr_method = core._ocr_consensus_extract(
                data, progress_callback=progress_callback
            )
            add(ocr_method, ocr_df)
        except Exception as exc:
            if progress_callback:
                progress_callback(7.5, 10, f"⚠️ OCR consensus failed: {type(exc).__name__}: {exc}")
    elif progress_callback:
        reason = "ledger proven" if native_proven else "usable candidate found; integrity checks still apply"
        progress_callback(7.5, 10, f"✓ OCR skipped — {reason}")

    if not candidates:
        raise ValueError(
            "No extraction candidates found. PDF may be encrypted, unreadable, or lack transaction data."
        )

    validated = []
    for item in candidates:
        try:
            frame = item["df"].copy()
            is_pnb_candidate = item["method"] == "PNB native coordinate parser"

            # Preserve PNB physical source order. The bank-specific parser has
            # already mapped amounts by their printed columns; generic sorting
            # and side-repair heuristics can turn its single visible discrepancy
            # into multiple false mismatches.
            if not is_pnb_candidate:
                frame = core._sort_transaction_evidence(frame)
                frame = core._repair_movement_from_balance(frame)
                frame = core._repair_pdf_side_mapping(frame)
            else:
                frame = frame.reset_index(drop=True)

            is_acceptable, reason = _is_acceptably_valid(item["stats"])
            if not is_acceptable:
                continue

            if is_pnb_candidate:
                reconciliation = core.balance_check(frame)
                mismatch_mask = reconciliation["Status"].eq("MISMATCH")
                mismatch_count = int(mismatch_mask.sum())
                if mismatch_count == 0:
                    integrity = core.validate_transaction_integrity(frame, item["method"])
                elif mismatch_count == 1:
                    # Allow a useful evidence-ledger export while explicitly
                    # marking the one unresolved source row for review. Never
                    # modify its amount or represent the ledger as fully reconciled.
                    mismatch = reconciliation.loc[mismatch_mask].iloc[0]
                    source_row = mismatch.get("Source_Row", mismatch.name + 1)
                    source_page = mismatch.get("Source_Page", "")
                    difference = float(mismatch.get("Difference", 0.0))
                    frame["Balance_Validation"] = "MATCH"
                    row_mask = pd.Series(False, index=frame.index)
                    if "Source_Row" in frame.columns:
                        row_mask = pd.to_numeric(frame["Source_Row"], errors="coerce").eq(float(source_row))
                    elif "Source_Page" in frame.columns:
                        row_mask = frame["Source_Page"].eq(source_page) & (frame.index == mismatch.name)
                    else:
                        row_mask.iloc[int(mismatch.name)] = True
                    frame.loc[row_mask, "Balance_Validation"] = "REVIEW — balance mismatch"
                    frame["Balance_Difference"] = np.nan
                    frame.loc[row_mask, "Balance_Difference"] = difference
                    integrity = {
                        "movement_rate": float(frame[["Debit", "Credit"]].notna().any(axis=1).mean()),
                        "date_reversals": 0,
                        "balance_mismatches": 1,
                        "balance_gaps": 0,
                        "review_required": True,
                        "review_reason": f"One running-balance mismatch at source page {source_page}, row {source_row}; difference {difference:,.2f}. Amounts unchanged.",
                    }
                else:
                    continue
            else:
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
                if is_pnb_candidate and integrity.get("review_required"):
                    integrity["review_reason"] += " Statement summary totals also differ; verify against the source PDF."
                else:
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
    if "Balance_Validation" in df.columns:
        review_rows = df["Balance_Validation"].astype(str).str.startswith("REVIEW")
        if "Priority" in df.columns:
            df.loc[review_rows, "Priority"] = "REVIEW"
        if review_rows.any():
            df.loc[review_rows, "Review_Reason"] = df.loc[review_rows, "Balance_Validation"]
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

    if "Balance_Validation" in df.columns and df["Balance_Validation"].astype(str).str.startswith("REVIEW").any():
        meta["warnings"].append("⚠️ PNB ledger exported with one explicitly flagged balance mismatch; verify the marked row against the source PDF before relying on totals.")

    return {"transactions": df, "flags": flags, "meta": meta}
