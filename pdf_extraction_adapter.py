"""PDF extraction orchestration layer.

The adapter keeps the forensic core's validation/classification logic but
replaces its brittle single-layout native parser with the bank-format-flexible
engine in bank_pdf_engine.py. Wrapped narration is then attached without
changing dates or amounts.
"""

import re

from bank_pdf_engine import extract_bank_pdf


def _norm(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _line_key(word):
    return round(float(word[1]) / 2) * 2


def _clean_line(words):
    return " ".join(
        str(w[4]).strip() for w in sorted(words, key=lambda x: x[0])
    ).strip()


def _is_continuation(line):
    text = str(line).strip()
    if not text or not re.search(r"[A-Za-z]{2,}", text):
        return False
    if re.search(
        r"\b(?:page|statement|opening balance|closing balance|total withdrawal|total deposit)\b",
        text,
        re.I,
    ):
        return False
    if re.search(
        r"\b(?:date|narration|description|particular|debit|credit|withdrawal|deposit|balance)\b",
        text,
        re.I,
    ):
        return False
    if re.search(r"\b(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b", text):
        return False
    if re.fullmatch(
        r"\s*(?:₹\s*)?[\d,]+(?:\.\d+)?\s*(?:CR|DR)?\s*",
        text,
        re.I,
    ):
        return False
    return True


def _attach_wrapped_narrations(data, transactions):
    """Attach clearly textual wrapped narration lines to extracted rows."""
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
            prior = [
                i for i in result.index
                if int(result.at[i, "Source_Page"]) < page_no
            ]
            tx_indices = [prior[-1]] if prior else []

        if not tx_indices:
            continue

        date_re = re.compile(
            r"^\s*(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|"
            r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"[a-z]*\s+\d{2,4})\b",
            re.I,
        )

        active_idx = None
        same_page_cursor = 0
        same_page = sorted(tx_indices, key=lambda i: i)
        used = 0

        for _, line in ordered:
            if date_re.match(line):
                if same_page_cursor < len(same_page):
                    active_idx = same_page[same_page_cursor]
                    same_page_cursor += 1
                    used = 0
                continue

            if active_idx is None:
                prior = [
                    i for i in result.index
                    if int(result.at[i, "Source_Page"]) < page_no
                ]
                active_idx = prior[-1] if prior else None

            if active_idx is None or used >= 8 or not _is_continuation(line):
                continue

            extra = re.sub(r"\s+", " ", line).strip()
            old_narr = str(result.at[active_idx, "Narration"]).strip()
            old_src = str(result.at[active_idx, "Source_Text"]).strip()

            if extra and extra not in old_narr:
                result.at[active_idx, "Narration"] = (
                    f"{old_narr} {extra}".strip()
                )
                result.at[active_idx, "Source_Text"] = (
                    f"{old_src} {extra}".strip()
                )
                used += 1

    return result


def analyze_upload(data, name, progress_callback=None):
    """Route PDFs through the master ensemble; keep Excel/CSV on the core path."""
    from pathlib import Path

    if Path(name).suffix.lower() == ".pdf":
        from master_pdf_intelligence import extract_master_pdf
        result = extract_master_pdf(
            data,
            name,
            progress_callback=progress_callback,
        )
        import forensic_core as core
        tx = _attach_wrapped_narrations(data, result["transactions"])
        tx = core._enforce_single_date_schema(core.enrich(tx))
        result["transactions"] = tx
        result["flags"] = tx[tx.Priority.isin(["REVIEW", "CRITICAL"])].copy()
        result["meta"]["warnings"].append(
            "Wrapped narration reconstruction was applied after ledger validation; dates and amounts were not altered."
        )
        return result

    import forensic_core as core
    return core.analyze_upload(
        data,
        name,
        progress_callback=progress_callback,
    )


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
    "_attach_wrapped_narrations",
]
