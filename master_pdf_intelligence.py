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

    # OCR + format detection: a second independent pass for hybrid or low-text
    # PDFs with uncertain native extraction. This pass is intentionally
    # evidence-aware and supplements, rather than replacing, the ledger solver.
    if extract_bank_statement_ocr is not None:
        try:
            ocr_df = extract_bank_statement_ocr(data, progress_callback=progress_callback)
            if not ocr_df.empty:
                add("OCR + format-aware bank extraction", ocr_df)
        except Exception:
            pass
