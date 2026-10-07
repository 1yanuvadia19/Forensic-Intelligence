"""Bank-format-agnostic PDF extraction adapter.

This module keeps the deterministic forensic core intact while adding a
layout-agnostic post-pass for one of the most common bank-PDF variations:
transaction narrations that wrap onto physical PDF lines or continue at the
top of the next page.

The adapter uses PyMuPDF word coordinates (not plain text order), which is
appropriate for PDFs where visual columns are represented by x/y positions.
"""

import re


def _norm(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _line_key(word):
    return round(float(word[1]) / 2) * 2


def _clean_line(words):
    return " ".join(str(w[4]).strip() for w in sorted(words, key=lambda x: x[0])).strip()


def _is_continuation(line):
    text = str(line).strip()
    if not text or not re.search(r"[A-Za-z]{2,}", text):
        return False
    if re.search(r"\b(?:page|statement|opening balance|closing balance|total withdrawal|total deposit)\b", text, re.I):
        return False
    if re.search(r"\b(?:date|narration|description|particular|debit|credit|withdrawal|deposit|balance)\b", text, re.I):
        return False
    if re.search(r"\b(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b", text):
        return False
    # A continuation line is normally narrative text, not a standalone money row.
    if re.fullmatch(r"\s*(?:₹\s*)?[\d,]+(?:\.\d+)?\s*(?:CR|DR)?\s*", text, re.I):
        return False
    return True


def _attach_wrapped_narrations(data, transactions):
    """Attach clearly textual wrapped narration lines to extracted rows.

    We only modify Narration/Source_Text. Amounts, dates, references and
    balances are never inferred here. A maximum of three continuation lines is
    attached to one transaction to avoid swallowing page furniture.
    """
    if transactions is None or transactions.empty:
        return transactions

    import fitz

    result = transactions.copy().reset_index(drop=True)
    if "Source_Page" not in result.columns:
        return result

    doc = fitz.open(stream=data, filetype="pdf")
    page_rows = {}
    for idx, row in result.iterrows():
        page = int(row["Source_Page"])
        page_rows.setdefault(page, []).append(idx)

    for page_no in range(1, len(doc) + 1):
        page = doc[page_no - 1]
        words = page.get_text("words")
        if not words:
            continue

        lines = {}
        for word in words:
            lines.setdefault(_line_key(word), []).append(word)
        ordered = sorted((y, _clean_line(ws)) for y, ws in lines.items())
        tx_indices = page_rows.get(page_no, [])
        if not tx_indices:
            # The top of a page can be a continuation of the last transaction
            # from the preceding page.
            prior = [i for i in result.index if int(result.at[i, "Source_Page"]) < page_no]
            tx_indices = [prior[-1]] if prior else []

        if not tx_indices:
            continue

        date_re = re.compile(r"^\s*(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b", re.I)
        for y, line in ordered:
            if date_re.match(line):
                for idx in tx_indices:
                    src = str(result.at[idx, "Source_Text"])
                    if str(result.at[idx, "Source_Page"]) == str(page_no) and line[:40] in src:
                        break

        # Fallback: source rows are already in statement order. Attach only
        # lines before the next dated transaction on the same page.
        active_idx = None
        used = 0
        same_page = sorted(tx_indices, key=lambda i: i)
        same_page_cursor = 0

        for _, line in ordered:
            if date_re.match(line):
                if same_page_cursor < len(same_page):
                    active_idx = same_page[same_page_cursor]
                    same_page_cursor += 1
                    used = 0
                continue

            if active_idx is None:
                # On a continuation page, use the last prior transaction only
                # for the first few textual lines.
                prior = [i for i in result.index if int(result.at[i, "Source_Page"]) < page_no]
                active_idx = prior[-1] if prior else None

            if active_idx is None or used >= 3 or not _is_continuation(line):
                continue

            extra = re.sub(r"\s+", " ", line).strip()
            old_narr = str(result.at[active_idx, "Narration"]).strip()
            old_src = str(result.at[active_idx, "Source_Text"]).strip()
            if extra and extra not in old_narr:
                result.at[active_idx, "Narration"] = f"{old_narr} {extra}".strip()
                result.at[active_idx, "Source_Text"] = f"{old_src} {extra}".strip()
                used += 1

    return result


def analyze_upload(*args, **kwargs):
    """Run the existing forensic pipeline with the PDF continuation adapter."""
    import forensic_core as core

    original_native = core._native_pdf_position_rows
    original_ocr = core._ocr_pdf_position_rows

    def native_adapter(data, progress_callback=None):
        extracted = original_native(data, progress_callback=progress_callback)
        return _attach_wrapped_narrations(data, extracted)

    def ocr_adapter(data, progress_callback=None, ocr_psm=6):
        extracted = original_ocr(
            data,
            progress_callback=progress_callback,
            ocr_psm=ocr_psm,
        )
        return _attach_wrapped_narrations(data, extracted)

    core._native_pdf_position_rows = native_adapter
    core._ocr_pdf_position_rows = ocr_adapter
    try:
        return core.analyze_upload(*args, **kwargs)
    finally:
        core._native_pdf_position_rows = original_native
        core._ocr_pdf_position_rows = original_ocr


# Re-export the public application API.
from forensic_core import (  # noqa: E402
    build_workbook,
    build_master_analysis,
    build_pdf_report,
    balance_mismatches,
)

__all__ = [
    "analyze_upload",
    "build_workbook",
    "build_master_analysis",
    "build_pdf_report",
    "balance_mismatches",
]
