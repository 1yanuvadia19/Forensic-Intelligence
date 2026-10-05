# FORENSIC INTELLIGENCE 360° — MASTER BLASTER v2

This build was created from the supplied benchmark evidence:
- DHIREN AJAKIYA SBI SUMMARY - done.xlsx (8,416 transaction rows)
- RAHUL SBI.pdf (120-page statement; scanned/image-based behaviour observed)
- DHIREN AJAKIYA SBI MASTER BLASTER forensic result
- previous successful Sandip Patel SBI forensic result (when included)

## What was learned from the DHIREN workbook
The transaction table begins after two title/metadata rows. The header contains:
Sr. No., Post Date, Cheque No., Description, Remarks, Debit (Rs.), Credit (Rs.), Balance (Rs.)

The parser now searches for the header instead of assuming row 1 is the header, supports multiple worksheets, and preserves source sheet/source row.

## Run without Command Prompt
Double-click:
START_FORENSIC_TOOL.vbs

It uses the folder containing the launcher, installs requirements if needed, starts Streamlit, and opens the browser.

## Inputs
.xlsx / .xls / .csv / .pdf

## PDF policy
The supplied 120-page PDF is treated conservatively. If it is scanned/image-only, the app stops rather than inventing transaction rows. A production OCR/table-reconstruction layer is the next required module.

## Forensic caution
This is an analytical screening tool, not a certified forensic extraction or legal conclusion engine. Every flagged transaction must be verified against the source evidence.
