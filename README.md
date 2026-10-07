# FORENSIC INTELLIGENCE — MASTER BLASTER v14.3

This build is the refreshed forensic extraction engine for bank-statement evidence with independent data-entry proof.

## v13 foundation

The extraction/normalization layer is based on the supplied Claude-style forensic extraction prompt, while preserving the project's later evidence-first rules:

- Native/digital PDF coordinate-aware extraction
- Scanned/hybrid PDF OCR with independent passes and deterministic reconciliation
- Excel / XLS / CSV multi-sheet header detection
- Exactly one Date column in the normalized output
- Human-style Particulars instead of bracketed labels
- Explicit counterparty extraction only from readable source evidence
- Payment-rail classification: UPI, IMPS, NEFT, RTGS, NACH/ECS/ACH, ATM/Cash, cheque, card/POS, transfer, etc.
- Reference cleanup and preservation of useful identifiers
- Running-balance validation
- Missing-source-row recovery through a GAP RESET anchor when subsequent bank-reported balances independently reconcile
- No fabricated transactions or amounts
- Review/Critical are review priorities, not conclusions of fraud or illegality
- Evidence provenance through source-page/source-row fields
- Investigation-ready Excel and PDF exports

## Particulars philosophy

Particulars are normalized to look like a human-entered forensic ledger, for example:

- Interest
- Bank Charge
- Cash Withdrawal – ATM
- Cheque Withdrawal
- Net Banking - JITENDRA
- TRANSFER OUT - HDFC
- Loan / Finance - BAJAJ FINANCE

The original transaction evidence remains available to the analysis engine; normalization does not invent names or amounts.

## Run without Command Prompt

Double-click START_FORENSIC_TOOL.vbs

## Inputs

- .xlsx
- .xls
- .csv
- native/digital .pdf
- scanned/image/hybrid .pdf

## Forensic caution

This is an analytical extraction and screening tool, not a certified forensic extraction or legal-conclusion engine. Material findings must always be checked against the original statement and supporting evidence.
NaN