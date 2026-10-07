"""Bank-statement OCR + format-aware extraction layer.

This module adds a format detection and OCR reconciliation layer for bank
statements with varying layouts. It is intentionally conservative: it only
extracts rows when OCR can identify a date and a movement or balance signal.
"""

from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageFilter, ImageOps


DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4})\b"
)
MONEY_RE = re.compile(
    r"(?:₹\s*)?\(?\d[\d,]*(?:\.\d+)?\)?(?:\s*(?:CR|DR))?",
    re.I,
)


class BankFormatDetector:
    """Heuristic detector for common Indian bank statement styles."""

    PATTERNS = {
        "HDFC": ["hdfc", "hdfc bank", "hdfc bank ltd", "cif", "statement of account"],
        "ICICI": ["icici", "icici bank", "i.c.i.c.i", "customer id", "branch code"],
        "SBI": ["sbi", "state bank of india", "sb account", "sbi card"],
        "AXIS": ["axis bank", "axis", "axis finance"],
        "KOTAK": ["kotak", "kotak mahindra"],
        "YES": ["yes bank", "yes foundation"],
        "BANK_OF_BARODA": ["bank of baroda", "bob"],
        "PNB": ["pnb", "punjab national bank"],
    }

    @staticmethod
    def detect(text: str) -> Dict[str, Any]:
        norm = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()
        best_name = "GENERIC"
        best_score = 0
        matched_keywords: List[str] = []

        for name, keywords in BankFormatDetector.PATTERNS.items():
            score = sum(1 for keyword in keywords if keyword in norm)
            if score > best_score:
                best_score = score
                best_name = name
                matched_keywords = [kw for kw in keywords if kw in norm][:6]

        confidence = min(0.99, 0.45 + 0.12 * best_score)
        return {
            "name": best_name,
            "confidence": round(confidence, 2),
            "keywords": matched_keywords,
        }


def _money_value(raw: str) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip().replace("₹", "").replace("INR", "").replace(" ", "")
    if not text or text in {"-", "—", "nan", "None"}:
        return None
    upper = text.upper()
    sign = -1 if upper.endswith("DR") or text.startswith("-") else 1
    match = re.search(r"\d[\d,]*(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return sign * float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _ocr_words(image: Image.Image) -> List[Dict[str, Any]]:
    try:
        data = pytesseract.image_to_data(
            image,
            config="--oem 3 --psm 6",
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return []

    words: List[Dict[str, Any]] = []
    for idx, text in enumerate(data.get("text", [])):
        cleaned = str(text).strip()
        if not cleaned:
            continue
        conf = int(data.get("conf", [0] * len(data.get("text", [])))[idx]) if idx < len(data.get("conf", [])) else 0
        if conf < 0:
            conf = 0
        left = float(data.get("left", [0] * len(data.get("text", [])))[idx])
        top = float(data.get("top", [0] * len(data.get("text", [])))[idx])
        width = float(data.get("width", [0] * len(data.get("text", [])))[idx])
        height = float(data.get("height", [0] * len(data.get("text", [])))[idx])
        words.append({
            "text": cleaned,
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "conf": conf,
        })
    return words


def _group_words_by_line(words: List[Dict[str, Any]], tolerance: float = 18.0) -> List[List[Dict[str, Any]]]:
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: (float(w["top"]), float(w["left"])))
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    last_top = None

    for word in sorted_words:
        top = float(word["top"])
        if current and last_top is not None and abs(top - last_top) > tolerance:
            groups.append(current)
            current = []
        current.append(word)
        last_top = top

    if current:
        groups.append(current)
    return groups


def _extract_date_from_line(line_text: str) -> Optional[str]:
    match = DATE_RE.search(line_text)
    if not match:
        return None
    return match.group(0)


def _extract_money_tokens(line_text: str):
    matches = []
    for token in re.finditer(r"(?:₹\s*)?\(?\d[\d,]*(?:\.\d+)?\)?(?:\s*(?:CR|DR))?", line_text, flags=re.I):
        value = _money_value(token.group(0))
        if value is not None:
            matches.append({"raw": token.group(0), "value": value})
    return matches


def _remove_date_and_money(line_text: str) -> str:
    text = line_text
    text = DATE_RE.sub(" ", text)
    text = re.sub(r"(?:₹\s*)?\(?\d[\d,]*(?:\.\d+)?\)?(?:\s*(?:CR|DR))?", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -:|")


def _infer_direction(narration: str, amount: float, balance: Optional[float], previous_balance: Optional[float]) -> str:
    upper = (narration or "").upper()
    if balance is not None and previous_balance is not None and not np.isnan(previous_balance):
        delta = balance - previous_balance
        if abs(delta - amount) <= max(0.05, abs(amount) * 0.0001):
            return "Credit"
        if abs(delta + amount) <= max(0.05, abs(amount) * 0.0001):
            return "Debit"

    if re.search(r"\b(?:DR|DEBIT|WITHDRAW|ATM|CHQ|CHEQUE|FEE|CHARGE|GST|SERVICE|CARD)\b", upper):
        return "Debit"
    if re.search(r"\b(?:CR|CREDIT|UPI|NEFT|IMPS|RTGS|SALARY|INCOME|REVERSAL|BY)\b", upper):
        return "Credit"
    return "Credit" if amount >= 0 else "Debit"


def _page_to_image(page) -> Image.Image:
    pix = page.get_pixmap(matrix=__import__("fitz").Matrix(2, 2), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
    image = ImageOps.autocontrast(image)
    image = image.filter(ImageFilter.SHARPEN)
    return image


def extract_bank_statement_ocr(data: bytes, progress_callback=None) -> pd.DataFrame:
    """OCR-first candidate extraction for scanned or hybrid bank statements.

    The output is intentionally shaped like the existing forensic ledger schema so
    that it can be fed into the validation and analytics pipeline.
    """
    import fitz

    doc = fitz.open(stream=data, filetype="pdf")
    rows: List[Dict[str, Any]] = []
    previous_balance: Optional[float] = None
    seq = 0

    for page_no, page in enumerate(doc, 1):
        if progress_callback:
            progress_callback(page_no, max(len(doc), 1), f"OCR extraction page {page_no}/{len(doc)}")

        image = _page_to_image(page)
        words = _ocr_words(image)
        if not words:
            continue

        page_text = " ".join(w["text"] for w in words)
        bank_format = BankFormatDetector.detect(page_text)

        groups = _group_words_by_line(words)
        for group in groups:
            line_text = " ".join(w["text"] for w in group)
            date_raw = _extract_date_from_line(line_text)
            if not date_raw:
                continue

            money_tokens = _extract_money_tokens(line_text)
            if not money_tokens:
                continue

            amount = max(abs(item["value"]) for item in money_tokens)
            balance_candidates = [item["value"] for item in money_tokens if item["raw"].upper().endswith("CR") or item["raw"].upper().endswith("DR") or "BAL" in line_text.upper()]
            balance = max(balance_candidates) if balance_candidates else None

            narration = _remove_date_and_money(line_text)
            narration = re.sub(r"\s+", " ", narration).strip()
            if not narration:
                narration = line_text.strip()

            date_value = pd.to_datetime(date_raw, errors="coerce", dayfirst=True)
            if pd.isna(date_value):
                continue

            if balance is not None and previous_balance is not None:
                direction = _infer_direction(narration, float(amount), float(balance), previous_balance)
            else:
                direction = _infer_direction(narration, float(amount), None, previous_balance)

            debit = amount if direction == "Debit" else np.nan
            credit = amount if direction == "Credit" else np.nan

            row = {
                "Date": date_value,
                "Narration": narration[:500],
                "Reference": "",
                "Debit": debit,
                "Credit": credit,
                "Balance": float(balance) if balance is not None else np.nan,
                "Source_Page": page_no,
                "Source_Seq": seq,
                "Source_Text": line_text[:1500],
                "OCR_Format": bank_format["name"],
                "OCR_Confidence": bank_format["confidence"],
            }
            rows.append(row)
            seq += 1

            if row.get("Balance") is not None and not np.isnan(row["Balance"]):
                previous_balance = float(row["Balance"])

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result = result.sort_values(["Source_Page", "Source_Seq"], kind="stable").reset_index(drop=True)
    return result


__all__ = ["BankFormatDetector", "extract_bank_statement_ocr"]
