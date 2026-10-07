import io
import os
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from pdf_extraction_adapter import analyze_upload, build_workbook, build_master_analysis, build_pdf_report, balance_mismatches

# Bump this whenever extraction/export logic changes. It prevents Streamlit from
# reusing a stale in-memory result after a forensic-engine update.
ANALYSIS_ENGINE_VERSION = "2026-10-07-master-forensic-intelligence-v20.0-ocr-premium"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Forensic Intelligence | Bank Statement Analysis",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# UI STYLE & THEME
# ============================================================

st.markdown(
    """
    <style>
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    .hero {
        padding: 28px 36px;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #0f3552,
            #1a4a70,
            #265a88
        );
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 12px 28px rgba(15, 53, 90, 0.22);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .hero p {
        margin: 8px 0 0;
        color: #d4e5f7;
        font-size: 1.1rem;
        font-weight: 400;
    }

    .hero .badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        color: #d4e5f7;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin-top: 12px;
        font-weight: 500;
    }

    .status-box {
        padding: 16px 20px;
        border-radius: 12px;
        border: 1px solid #d9e2ec;
        background: #f8fafc;
        margin-bottom: 14px;
    }

    .mode-selector {
        display: flex;
        gap: 12px;
        margin: 16px 0;
    }

    .mode-btn {
        flex: 1;
        padding: 12px 16px;
        border-radius: 10px;
        border: 2px solid #d9e2ec;
        background: white;
        cursor: pointer;
        font-size: 0.95rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .mode-btn:hover {
        border-color: #265a88;
        background: #f0f5fa;
    }

    .mode-btn.active {
        border-color: #265a88;
        background: linear-gradient(135deg, #1a4a70, #265a88);
        color: white;
    }

    .info-card {
        background: linear-gradient(135deg, #f0f5fa, #e8eef8);
        padding: 16px;
        border-radius: 10px;
        border-left: 4px solid #265a88;
        margin: 12px 0;
    }

    .info-card strong {
        color: #0f3552;
    }

    .extraction-step {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 0;
        font-size: 0.95rem;
    }

    .step-badge {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: #265a88;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .quality-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .quality-high {
        background: #e8f5e9;
        color: #2e7d32;
    }

    .quality-medium {
        background: #fff3e0;
        color: #e65100;
    }

    .quality-low {
        background: #ffebee;
        color: #c62828;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🔎 FORENSIC INTELLIGENCE</h1>
        <p>Premium bank-statement extraction • OCR + native PDF • payment-rail parsing • forensic evidence</p>
        <span class="badge">✓ Dual-mode extraction (scanned + digital PDF)</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR — EVIDENCE INTAKE & MODE SELECTION
# ============================================================

with st.sidebar:

    st.header("📂 Evidence Intake")

    # Mode selection
    st.subheader("📋 PDF Source Mode")
    mode = st.radio(
        "Select your bank statement source:",
        ("🏦 Bank Digital PDF", "📸 Scanned/Photographed"),
        help="Choose based on how you received your statement. The system automatically optimizes extraction.",
        key="pdf_mode"
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload bank statement",
        type=["xlsx", "xls", "csv", "pdf"],
        help=(
            "✓ Excel (.xlsx, .xls)\n"
            "✓ CSV (.csv)\n"
            "✓ Bank PDF (native digital)\n"
            "✓ Scanned PDF (photographed, image-based)"
        ),
    )

    st.divider()

    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []

    st.markdown("### 🕐 Recent Analysis")
    if st.session_state.analysis_history:
        st.caption(f"**{len(st.session_state.analysis_history)}** statement(s) in session")
        for item in st.session_state.analysis_history[-6:][::-1]:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.caption(f"📄 {item['name'][:28]}")
            with col2:
                st.caption(f"{item['transactions']:,} txn")
    else:
        st.caption("No statements yet.")

    st.divider()

    st.markdown("### ⚙️ Extraction Pipeline")
    with st.expander("View pipeline steps", expanded=False):
        steps = [
            ("1", "Format detection", "Bank + PDF type identification"),
            ("2", "Native extraction", "Digital PDF text layer processing"),
            ("3", "OCR fallback", "Scanned/hybrid page reconstruction"),
            ("4", "Ledger validation", "Balance + transaction integrity check"),
            ("5", "Payment rail ID", "UPI, NEFT, IMPS, ATM, Cheque, etc."),
            ("6", "Anomaly screen", "Review + Critical priority flags"),
            ("7", "Evidence export", "Excel + PDF with source tracking"),
        ]
        for badge, title, desc in steps:
            st.markdown(
                f"""<div class="extraction-step">
                <span class="step-badge">{badge}</span>
                <div><strong>{title}</strong><br/><small>{desc}</small></div>
                </div>""",
                unsafe_allow_html=True,
            )


# ============================================================
# NO FILE
# ============================================================

if uploaded_file is None:

    st.markdown(
        """
        <div class="info-card">
        <strong>🚀 Ready to analyze.</strong> Upload a bank statement from the left panel.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✓ What we accept")
        st.markdown(
            """
            - **Excel files** (.xlsx, .xls)
            - **CSV exports** (.csv)
            - **Bank PDFs** (native digital)
            - **Scanned PDFs** (photographed statements)
            - **Hybrid PDFs** (mixed text + images)
            """
        )

    with col2:
        st.markdown("### 🎯 What we do")
        st.markdown(
            """
            - Extract with **OCR + format awareness**
            - Validate all **transactions** against running balance
            - Classify **payment rails** (UPI, NEFT, IMPS, Cheque, ATM, etc.)
            - Screen for **anomalies** (review/critical priorities)
            - Export **evidence-ready** Excel + PDF
            """
        )

    st.divider()

    st.markdown("### 📖 Evidence-First Principle")
    st.info(
        "**We do not invent data.** If extraction fails or cannot be validated against "
        "the statement's own balance evidence, analysis stops. This ensures every exported "
        "transaction is traceable to the original source."
    )

    st.stop()


# ============================================================
# ANALYSE UPLOAD
# ============================================================

file_bytes = uploaded_file.getvalue()
file_name = uploaded_file.name
file_hash = hashlib.sha256(file_bytes).hexdigest()

col_name, col_bytes = st.columns([2, 1])
with col_name:
    st.caption(f"📄 **{file_name}**")
with col_bytes:
    st.caption(f"{len(file_bytes):,} bytes")

col_mode, col_engine = st.columns([1, 1])
with col_mode:
    mode_label = "🏦 Bank PDF" if mode == "🏦 Bank Digital PDF" else "📸 Scanned PDF"
    st.caption(f"**Mode:** {mode_label}")
with col_engine:
    st.caption(f"**Engine:** v{ANALYSIS_ENGINE_VERSION.split('-')[-1]}")

# Reuse the result for the same uploaded evidence instead of re-running
# the entire PDF pipeline on every Streamlit rerun.
cached_result = st.session_state.get("analysis_result")
cached_hash = st.session_state.get("analysis_hash")
cached_engine_version = st.session_state.get("analysis_engine_version")

analysis_cache_key = f"{file_hash}:{ANALYSIS_ENGINE_VERSION}:{mode}"

if (
    cached_result is not None
    and cached_hash == analysis_cache_key
    and cached_engine_version == ANALYSIS_ENGINE_VERSION
):
    result = cached_result
    st.success("✓ Analysis reused — extraction already complete.")
else:
    progress = st.progress(0, text="Initializing forensic analysis…")
    status = st.empty()

    def show_progress(done, total, message):
        total = max(int(total or 1), 1)
        pct = max(0.0, min(float(done) / total, 1.0))
        progress.progress(pct, text=f"{int(pct * 100)}% — {message}")
        status.caption(f"**Progress:** {int(pct * 100)}% | {message}")

    try:
        result = analyze_upload(
            file_bytes,
            file_name,
            progress_callback=show_progress,
            pdf_mode=mode,
        )
        progress.progress(1.0, text="100% — Extraction complete")
        status.success("✓ Forensic analysis completed successfully.")

        st.session_state.analysis_result = result
        st.session_state.analysis_hash = analysis_cache_key
        st.session_state.analysis_engine_version = ANALYSIS_ENGINE_VERSION
        st.session_state.derived_hash = None
        st.session_state.derived_analysis = None
        st.session_state.export_hash = None
        st.session_state.export_engine_version = ANALYSIS_ENGINE_VERSION
        st.session_state.export_workbook = None
        st.session_state.export_pdf = None

        # Store a compact session history record.
        history = st.session_state.setdefault("analysis_history", [])
        history = [h for h in history if h["hash"] != file_hash]
        history.append({
            "hash": file_hash,
            "name": file_name,
            "transactions": len(result["transactions"]),
            "source": result["meta"].get("source_type", "-"),
            "location": result["meta"].get("location", "-"),
            "pdf_mode": mode,
        })
        st.session_state.analysis_history = history[-25:]

    except Exception as exc:
        progress.empty()
        status.empty()

        st.error("⚠️ **Evidence extraction / validation failed**")
        st.warning(str(exc))

        st.markdown(
            """
            **Analysis stopped — this is intentional.**

            The system refuses to continue with unvalidated bank-statement data.
            This ensures every transaction is evidence-backed.

            **Next steps:**
            - Verify the PDF is readable and contains transaction data
            - Try the other PDF mode (scanned vs. digital)
            - Check that the statement includes running balances
            """
        )

        st.stop()


# ============================================================
# RESULTS
# ============================================================

df = result["transactions"].copy()
flags = result["flags"].copy()
meta = result["meta"].copy()

# Cache derived analytics for the current evidence hash.
if st.session_state.get("derived_hash") != file_hash:
    with st.spinner("Preparing forensic analytics…"):
        st.session_state.derived_analysis = build_master_analysis(df)
        st.session_state.derived_hash = file_hash

master_cached, fund_flow_cached, concentration_cached, dq_cached, provenance_cached = st.session_state.derived_analysis


# ============================================================
# COMMAND CENTER
# ============================================================

tabs = st.tabs(
    [
        "🎯 Command Center",
        "📋 Transactions",
        "🚨 Risk & Anomalies",
        "📑 Annexures",
        "🧠 Master Analysis",
        "🤖 AI Assistant",
        "📤 Export",
        "🗂️ History",
    ],
)


# ============================================================
# COMMAND CENTER TAB
# ============================================================

with tabs[0]:

    st.subheader("🎯 Command Center")

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Transactions",
        f"{len(df):,}",
        delta=None,
    )

    c2.metric(
        "Total Credits",
        f"₹{df['Credit'].fillna(0).sum():,.2f}",
    )

    c3.metric(
        "Total Debits",
        f"₹{df['Debit'].fillna(0).sum():,.2f}",
    )

    c4.metric(
        "Review Queue",
        f"{len(flags):,}",
        delta=f"{int((len(flags) / len(df) * 100)) if len(df) > 0 else 0}% flagged",
    )

    st.divider()

    # Extraction quality & source info
    st.subheader("📊 Extraction Quality & Evidence")

    e1, e2, e3 = st.columns(3)

    source_type = meta.get("source_type", "Unknown")
    if "scanned" in source_type.lower():
        quality_class = "quality-medium"
        quality_emoji = "⚠️"
    else:
        quality_class = "quality-high"
        quality_emoji = "✓"

    with e1:
        st.markdown(
            f"""<div class="quality-indicator {quality_class}">
            {quality_emoji} <strong>{source_type}</strong>
            </div>""",
            unsafe_allow_html=True,
        )

    with e2:
        st.markdown(f"**Layout:** {meta.get('layout_confidence', 'N/A')}")

    with e3:
        st.markdown(f"**Pages:** {meta.get('location', 'N/A')}")

    st.divider()

    # Warnings & flags
    warnings = meta.get("warnings", [])
    if warnings:
        st.markdown("### ⚠️ Extraction Notes")
        for i, warning in enumerate(warnings[:5], 1):
            st.caption(f"**[{i}]** {warning}")
        if len(warnings) > 5:
            st.caption(f"*... and {len(warnings) - 5} more.*")

    st.divider()

    # Transaction overview
    st.markdown("### 📋 Transaction Sample (First 100)")
    st.dataframe(
        df.head(100),
        use_container_width=True,
        height=420,
    )


# ============================================================
# TRANSACTIONS TAB
# ============================================================

with tabs[1]:

    st.subheader("📋 Normalised Transactions")

    search = st.text_input(
        "🔍 Search narration / counterparty",
        placeholder="e.g. UPI JOHN, NEFT HDFC, ATM WITHDRAWAL, SALARY...",
    )

    view = df.copy()

    if search:
        mask = (
            view.astype(str)
            .apply(
                lambda column:
                column.str.contains(
                    search,
                    case=False,
                    na=False,
                )
            )
            .any(axis=1)
        )
        view = view[mask]

    st.caption(
        f"Showing **{len(view):,}** transaction(s) of **{len(df):,}** total"
    )

    display_view = view.head(5000)
    if len(view) > 5000:
        st.caption("*Showing first 5,000 rows for performance. Export contains full evidence set.*")

    st.dataframe(
        display_view,
        use_container_width=True,
        height=600,
    )


# ============================================================
# RISK & ANOMALIES TAB
# ============================================================

with tabs[2]:

    st.subheader("🚨 Review & Risk Queue")

    review_count = (df["Priority"] == "REVIEW").sum()
    critical_count = (df["Priority"] == "CRITICAL").sum()

    r1, r2, r3 = st.columns(3)

    r1.metric("Review", f"{review_count:,}")
    r2.metric("Critical", f"{critical_count:,}")
    r3.metric("Total", f"{review_count + critical_count:,}")

    st.info(
        "**A Review/Critical flag is a priority for human review**, not a conclusion "
        "of fraud or illegality. Evidence-based anomaly detection only."
    )

    if flags.empty:
        st.success(
            "✓ No transactions meet the configured review rules. "
            "Statement appears normal."
        )
    else:
        st.markdown(f"### Top {min(len(flags), 5000)} Review Priorities")
        display_flags = flags.head(5000)
        if len(flags) > 5000:
            st.caption("*Showing first 5,000 rows for performance.*")
        st.dataframe(
            display_flags,
            use_container_width=True,
            height=600,
        )


# ============================================================
# ANNEXURES TAB
# ============================================================

with tabs[3]:

    st.subheader("📑 Investigation Annexures")

    st.markdown(
        """
        The exported evidence workbook contains the following investigation-ready schedules:

        - **Executive Summary** — Key metrics, opening/closing balance, net cash flow
        - **Normalised Transactions** — Full ledger with payment-rail classification
        - **Flow Analysis** — Large-credit / onward-debit patterns
        - **Master Analysis** — Forensic findings, concentration, data quality
        - **Source Tracking** — Every transaction linked to PDF page + position
        """
    )

    st.info(
        "Annexures are generated from extracted evidence. "
        "**Always verify material findings against the original statement.**"
    )


# ============================================================
# MASTER ANALYSIS TAB
# ============================================================

with tabs[4]:

    st.subheader("🧠 Master Forensic Analysis")
    st.caption("Concise evidence-led findings for investigative review.")

    master, fund_flow, concentration, dq, provenance = master_cached, fund_flow_cached, concentration_cached, dq_cached, provenance_cached

    st.markdown("#### Summary Findings")
    st.dataframe(master, use_container_width=True, height=500)

    st.markdown("#### Fund-Flow Review")
    if fund_flow.empty:
        st.success("✓ No configured large-credit / onward-debit pattern identified.")
    else:
        st.dataframe(fund_flow.head(60), use_container_width=True, height=360)

    st.markdown("#### Data Quality Report")
    st.dataframe(dq, use_container_width=True, height=280)


# ============================================================
# AI ASSISTANT TAB
# ============================================================

with tabs[5]:

    st.subheader("🤖 Forensic AI Assistant")
    st.caption("Ask questions about the analysed evidence, patterns and review priorities.")

    master_ai, fund_ai, cp_ai, dq_ai, prov_ai = master_cached, fund_flow_cached, concentration_cached, dq_cached, provenance_cached

    question = st.text_area(
        "Ask the forensic assistant",
        placeholder="Example: What are the most important findings? / Explain the largest transaction. / Are there rapid same-day patterns?",
        height=90,
    )

    if st.button("🔍 Analyse with AI", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            try:
                if not api_key:
                    try:
                        api_key = st.secrets.get("OPENAI_API_KEY")
                    except Exception:
                        api_key = None

                context = {
                    "source": meta,
                    "summary": master_ai.to_dict("records"),
                    "top_review_rows": flags.head(60).fillna("-").to_dict("records"),
                    "fund_flow": fund_ai.head(40).fillna("-").to_dict("records"),
                    "data_quality": dq_ai.fillna("-").to_dict("records"),
                }

                if api_key:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)
                    response = client.responses.create(
                        model="gpt-6-luna",
                        input=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a forensic financial analysis copilot. "
                                    "Answer only from the supplied evidence context. "
                                    "Distinguish facts, observations and review priorities. "
                                    "Never conclude fraud, guilt, illegality or intent. "
                                    "If evidence is insufficient, say so. Be concise and professional."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    "Evidence context:\n" + str(context) +
                                    "\n\nInvestigator question: " + question
                                ),
                            },
                        ],
                    )
                    st.markdown(response.output_text)
                else:
                    review_n = int((df["Priority"] == "REVIEW").sum())
                    critical_n = int((df["Priority"] == "CRITICAL").sum())
                    rapid_n = int(df["Rapid_Movement"].sum())
                    mismatch_n = int(balance_mismatches(df))
                    st.markdown(
                        "**Quick Evidence Summary**\n\n"
                        "- Transactions analysed: **" + f"{len(df):,}" + "**\n"
                        "- Review priorities: **" + f"{review_n:,}" + "**\n"
                        "- Critical priorities: **" + f"{critical_n:,}" + "**\n"
                        "- Rapid same/next-day movements: **" + f"{rapid_n:,}" + "**\n"
                        "- Balance mismatches: **" + f"{mismatch_n:,}" + "**\n\n"
                        "*For conversational AI reasoning, add an OPENAI_API_KEY to Streamlit app secrets.*"
                    )
            except Exception as exc:
                st.error(f"AI Assistant error: {exc}")


# ============================================================
# EXPORT TAB
# ============================================================

with tabs[6]:

    st.subheader("📤 Evidence-Ready Export")
    st.caption("Download investigation workbook (Excel) and forensic report (PDF).")

    export_hash = st.session_state.get("export_hash")
    export_engine_version = st.session_state.get("export_engine_version")
    export_ready = (
        export_hash == file_hash
        and export_engine_version == ANALYSIS_ENGINE_VERSION
        and st.session_state.get("export_workbook") is not None
    )

    if not export_ready:
        st.info("Exports are generated on-demand to keep the app responsive.")
        if st.button("⚙️ Prepare Evidence Exports", type="primary", use_container_width=True):
            try:
                with st.spinner("Generating Excel analysis and forensic PDF…"):
                    st.session_state.export_workbook = build_workbook(df, flags, meta)
                    st.session_state.export_pdf = build_pdf_report(df, flags, meta, file_name)
                    st.session_state.export_hash = file_hash
                    st.session_state.export_engine_version = ANALYSIS_ENGINE_VERSION
                st.success("✓ Exports prepared. Download below.")
                st.rerun()
            except Exception as exc:
                st.error(f"Export generation failed: {exc}")
    else:
        workbook = st.session_state.export_workbook
        pdf_report = st.session_state.export_pdf
        base = Path(file_name).stem

        st.markdown("### ⬇️ Download")
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="📊 Excel Analysis",
                data=workbook,
                file_name=base + "_FORENSIC_ANALYSIS.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        with col2:
            st.download_button(
                label="📄 PDF Forensic Report",
                data=pdf_report,
                file_name=base + "_FORENSIC_REPORT.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        st.divider()
        if st.button("🔄 Rebuild Exports", use_container_width=True):
            st.session_state.export_hash = None
            st.session_state.export_workbook = None
            st.session_state.export_pdf = None
            st.rerun()


# ============================================================
# HISTORY TAB
# ============================================================

with tabs[7]:

    st.subheader("🗂️ Analysis History")
    st.caption(
        "Statements analysed in this browser session. Original evidence is never altered."
    )

    history = st.session_state.get("analysis_history", [])

    if not history:
        st.info("No statement has been analysed yet.")
    else:
        history_df = pd.DataFrame(history)
        history_df = history_df.rename(columns={
            "name": "Statement",
            "transactions": "Transactions",
            "source": "Source Type",
            "location": "Pages",
            "pdf_mode": "Mode",
        })
        history_df = history_df.drop(columns=["hash"], errors="ignore")
        st.dataframe(history_df.iloc[::-1], use_container_width=True, hide_index=True)

        st.divider()
        st.caption(
            "✓ Identical statements are automatically reused in this session. "
            "Tab navigation or widget interaction will not re-run extraction."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

footer_cols = st.columns(3)
with footer_cols[0]:
    st.caption("🔎 **FORENSIC INTELLIGENCE**")
with footer_cols[1]:
    st.caption("Evidence-first • OCR-aware • Payment-rail classified")
with footer_cols[2]:
    st.caption("*Human review required for investigative conclusions*")
