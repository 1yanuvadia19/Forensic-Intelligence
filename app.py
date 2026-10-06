import io
import os
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from forensic_core import analyze_upload, build_workbook, build_master_analysis, build_pdf_report, balance_mismatches

# Bump this whenever extraction/export logic changes. It prevents Streamlit from
# reusing a stale in-memory result after a forensic-engine update.
ANALYSIS_ENGINE_VERSION = "2026-10-06-master-blaster-v4"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Forensic Intelligence",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# UI STYLE
# ============================================================

st.markdown(
    """
    <style>

    .hero {
        padding: 24px 30px;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #15243a,
            #24476f
        );
        color: white;
        margin-bottom: 18px;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.25rem;
    }

    .hero p {
        margin: .45rem 0 0;
        color: #d9e7f8;
        font-size: 1.05rem;
    }

    .status-box {
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid #d9e2ec;
        background: #f8fafc;
        margin-bottom: 12px;
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
        <p>
            Bank-statement intelligence • payment-rail parsing •
            anomaly screening • evidence-ready exports
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR — EVIDENCE INTAKE
# ============================================================

with st.sidebar:

    st.header("📂 Evidence Intake")

    uploaded_file = st.file_uploader(
        "Upload bank statement",
        type=["xlsx", "xls", "csv", "pdf"],
        help=(
            "Supported formats: Excel, CSV and bank-generated PDF. "
            "Scanned PDFs are automatically processed through OCR table reconstruction."
        ),
    )

    st.divider()

    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []

    st.markdown("### History")
    if st.session_state.analysis_history:
        st.caption(f"{len(st.session_state.analysis_history)} statement(s) analysed in this session")
        for item in st.session_state.analysis_history[-8:][::-1]:
            st.write(f"• {item['name']} — {item['transactions']:,} transactions")
    else:
        st.caption("No statements analysed yet in this session.")

    st.divider()

    st.markdown("### Pipeline")

    pipeline = [
        "1. Extraction reliability",
        "2. Extract transactions",
        "3. Identify payment rails",
        "4. Screen anomalies",
        "5. Reconcile balances",
        "6. Export evidence",
    ]

    for item in pipeline:
        st.write(item)


# ============================================================
# NO FILE
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload a bank statement from the left panel to begin "
        "the forensic analysis."
    )

    st.markdown(
        """
        ### Supported evidence

        - Excel `.xlsx`
        - Excel `.xls`
        - CSV `.csv`
        - Native bank-generated PDF `.pdf`
        - Scanned/image PDFs are automatically reconstructed using OCR with
          page-level evidence tracing and validation.

        ### Evidence-first principle

        The system does not treat extraction failure as valid data.
        If PDF column mapping cannot be validated, analysis is stopped
        rather than inventing transactions.
        """
    )

    st.stop()


# ============================================================
# ANALYSE UPLOAD
# ============================================================

file_bytes = uploaded_file.getvalue()
file_name = uploaded_file.name
file_hash = hashlib.sha256(file_bytes).hexdigest()

st.caption(
    f"Evidence loaded: **{file_name}** • "
    f"{len(file_bytes):,} bytes"
)

# Reuse the result for the same uploaded evidence instead of re-running
# the entire PDF pipeline on every Streamlit rerun.
cached_result = st.session_state.get("analysis_result")
cached_hash = st.session_state.get("analysis_hash")
cached_engine_version = st.session_state.get("analysis_engine_version")

analysis_cache_key = f"{file_hash}:{ANALYSIS_ENGINE_VERSION}"

if (
    cached_result is not None
    and cached_hash == analysis_cache_key
    and cached_engine_version == ANALYSIS_ENGINE_VERSION
):
    result = cached_result
    st.success("✓ Existing analysis reused — no re-processing required.")
else:
    progress = st.progress(0, text="Starting forensic analysis…")
    status = st.empty()

    def show_progress(done, total, message):
        total = max(int(total or 1), 1)
        pct = max(0.0, min(float(done) / total, 1.0))
        progress.progress(pct, text=f"{int(pct * 100)}% — {message}")
        status.caption(f"Analysis progress: **{int(pct * 100)}%**")

    try:
        result = analyze_upload(
            file_bytes,
            file_name,
            progress_callback=show_progress,
        )
        progress.progress(1.0, text="100% — Analysis complete")
        status.success("Forensic analysis completed successfully.")

        st.session_state.analysis_result = result
        st.session_state.analysis_hash = analysis_cache_key
        st.session_state.analysis_engine_version = ANALYSIS_ENGINE_VERSION
        st.session_state.derived_hash = None
        st.session_state.derived_analysis = None
        st.session_state.export_hash = None
        st.session_state.export_engine_version = ANALYSIS_ENGINE_VERSION
        st.session_state.export_workbook = None
        st.session_state.export_pdf = None

        # Store a compact session history record. Full transaction data stays
        # in analysis_result; history is intentionally lightweight.
        history = st.session_state.setdefault("analysis_history", [])
        history = [h for h in history if h["hash"] != file_hash]
        history.append({
            "hash": file_hash,
            "name": file_name,
            "transactions": len(result["transactions"]),
            "source": result["meta"].get("source_type", "-"),
            "location": result["meta"].get("location", "-"),
        })
        st.session_state.analysis_history = history[-25:]

    except Exception as exc:
        progress.empty()
        status.empty()

        st.error(
            "⚠️ **Evidence extraction / validation failed**"
        )

        st.warning(str(exc))

        st.markdown(
            """
            **Analysis has been stopped.**

            This is intentional: the system must not continue with
            incorrectly mapped bank-statement data.
            """
        )

        st.stop()


# ============================================================
# RESULTS
# ============================================================

df = result["transactions"].copy()
flags = result["flags"].copy()
meta = result["meta"].copy()

# Cache derived analytics for the current evidence hash. This prevents
# expensive pandas analysis from running again on every widget interaction.
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
        "🚨 Risk",
        "📑 Annexures",
        "🧠 Master Analysis",
        "🤖 AI Assistant",
        "📤 Export",
        "🗂️ History",
    ],
    on_change="rerun",
    key="main_tabs",
)


# ============================================================
# COMMAND CENTER
# ============================================================

if tabs[0].open:
    with tabs[0]:

        st.subheader("Command Center")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Transactions",
            f"{len(df):,}",
        )

        c2.metric(
            "Credits",
            f"₹{df['Credit'].fillna(0).sum():,.2f}",
        )

        c3.metric(
            "Debits",
            f"₹{df['Debit'].fillna(0).sum():,.2f}",
        )

        c4.metric(
            "Review Queue",
            f"{len(flags):,}",
        )

        st.divider()

        st.subheader("Evidence Status")

        m1, m2, m3 = st.columns(3)

        m1.write(
            f"**Source:** {meta.get('source_type', '-')}"
        )

        m2.write(
            f"**Location:** {meta.get('location', '-')}"
        )

        m3.write(
            f"**Extraction confidence:** "
            f"{meta.get('layout_confidence', '-')}"
        )

        warnings = meta.get("warnings", [])

        if warnings:

            for warning in warnings:
                st.warning(warning)

        st.subheader("Transaction Overview")

        st.dataframe(
            df.head(100),
            use_container_width=True,
            height=420,
        )


    # ============================================================
    # TRANSACTIONS
    # ============================================================

if tabs[1].open:
    with tabs[1]:

        st.subheader("Normalised Transactions")

        search = st.text_input(
            "Search narration / counterparty",
            placeholder="e.g. UPI, NEFT, SBI, LIC...",
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
            f"Showing {len(view):,} transaction(s)"
        )

        display_view = view.head(5000)
        if len(view) > 5000:
            st.caption("Showing the first 5,000 matching rows for performance. Export contains the full evidence set.")
        st.dataframe(
            display_view,
            use_container_width=True,
            height=600,
        )


    # ============================================================
    # RISK
    # ============================================================

if tabs[2].open:
    with tabs[2]:

        st.subheader("Review & Risk Queue")

        review_count = (
            (df["Priority"] == "REVIEW")
            .sum()
        )

        critical_count = (
            (df["Priority"] == "CRITICAL")
            .sum()
        )

        r1, r2 = st.columns(2)

        r1.metric(
            "Review",
            f"{review_count:,}",
        )

        r2.metric(
            "Critical",
            f"{critical_count:,}",
        )

        st.info(
            "A Review/Critical flag is a review priority, "
            "not a conclusion of fraud or illegality."
        )

        if flags.empty:

            st.success(
                "No transactions currently meet the configured "
                "review rules."
            )

        else:

            display_flags = flags.head(5000)
            if len(flags) > 5000:
                st.caption("Showing the first 5,000 review rows for performance. Export contains the full evidence set.")
            st.dataframe(
                display_flags,
                use_container_width=True,
                height=600,
            )


    # ============================================================
    # ANNEXURES
    # ============================================================

if tabs[3].open:
    with tabs[3]:

        st.subheader("Investigation Annexures")

        annexure_options = [
            "Executive Summary",
            "Normalised Transactions",
            "Flow Analysis",
            "Master Analysis",
        ]

        st.write(
            "The export workbook contains the following "
            "investigation-ready schedules:"
        )

        for item in annexure_options:
            st.write(f"✓ {item}")

        st.info(
            "Annexures are generated from extracted evidence. "
            "Always verify material findings against the original statement."
        )



    # ============================================================
    # MASTER ANALYSIS
    # ============================================================

if tabs[4].open:
    with tabs[4]:

        st.subheader("🧠 Master Forensic Analysis")
        st.caption("Concise evidence-led findings for investigative review.")

        master, fund_flow, concentration, dq, provenance = master_cached, fund_flow_cached, concentration_cached, dq_cached, provenance_cached

        st.dataframe(master, use_container_width=True, height=500)

        st.markdown("#### Fund-Flow Review")
        if fund_flow.empty:
            st.success("No configured large-credit / onward-debit pattern identified.")
        else:
            st.dataframe(fund_flow.head(60), use_container_width=True, height=360)

        st.markdown("#### Data Quality")
        st.dataframe(dq, use_container_width=True, height=280)


    # ============================================================
    # AI ASSISTANT
    # ============================================================

if tabs[5].open:
    with tabs[5]:

        st.subheader("🤖 Forensic AI Assistant")
        st.caption("Ask questions about the analysed evidence, patterns and review priorities.")

        master_ai, fund_ai, cp_ai, dq_ai, prov_ai = master_cached, fund_flow_cached, concentration_cached, dq_cached, provenance_cached

        question = st.text_area(
            "Ask the forensic assistant",
            placeholder="Example: What are the most important findings in this statement?",
            height=90,
        )

        if st.button("Analyse with AI", type="primary", use_container_width=True):
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
                            "**Quick evidence answer**\n\n"
                            "- Transactions analysed: **" + f"{len(df):,}" + "**\n"
                            "- Review priorities: **" + f"{review_n:,}" + "**\n"
                            "- Critical priorities: **" + f"{critical_n:,}" + "**\n"
                            "- Rapid same/next-day movements: **" + f"{rapid_n:,}" + "**\n"
                            "- Balance mismatches: **" + f"{mismatch_n:,}" + "**\n\n"
                            "For conversational evidence reasoning, add an OPENAI_API_KEY "
                            "to the Streamlit app secrets."
                        )
                except Exception as exc:
                    st.error(f"AI Assistant could not respond: {exc}")


    # ============================================================
    # EXPORT
    # ============================================================

if tabs[6].open:
    with tabs[6]:

        st.subheader("Evidence-Ready Export")
        st.caption("Download the concise investigation workbook or executive forensic report.")

        export_hash = st.session_state.get("export_hash")
        export_engine_version = st.session_state.get("export_engine_version")
        export_ready = (
            export_hash == file_hash
            and export_engine_version == ANALYSIS_ENGINE_VERSION
            and st.session_state.get("export_workbook") is not None
        )

        if not export_ready:
            st.info("Exports are generated only when requested, which keeps the Community Cloud app responsive.")
            if st.button("⚙️ Prepare Evidence Exports", type="primary", use_container_width=True):
                try:
                    with st.spinner("Preparing Excel analysis and forensic PDF…"):
                        st.session_state.export_workbook = build_workbook(df, flags, meta)
                        st.session_state.export_pdf = build_pdf_report(df, flags, meta, file_name)
                        st.session_state.export_hash = file_hash
                    st.success("Exports prepared. You can now download them below.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not generate the export: {exc}")
        else:
            workbook = st.session_state.export_workbook
            pdf_report = st.session_state.export_pdf
            base = Path(file_name).stem

            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    label="⬇️ Excel Analysis",
                    data=workbook,
                    file_name=base + "_FORENSIC_INTELLIGENCE.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

            with col2:
                st.download_button(
                    label="⬇️ PDF Forensic Report",
                    data=pdf_report,
                    file_name=base + "_FORENSIC_REPORT.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            if st.button("♻️ Rebuild Exports", use_container_width=True):
                st.session_state.export_hash = None
                st.session_state.export_workbook = None
                st.session_state.export_pdf = None
                st.rerun()


    # ============================================================
    # HISTORY
    # ============================================================

if tabs[7].open:
    with tabs[7]:

        st.subheader("🗂️ Analysis History")
        st.caption(
            "Statements analysed during this browser session. "
            "The original evidence is not altered."
        )

        history = st.session_state.get("analysis_history", [])

        if not history:
            st.info("No statement has been analysed in this session yet.")
        else:
            history_df = pd.DataFrame(history)
            history_df = history_df.rename(columns={
                "name": "Statement",
                "transactions": "Transactions",
                "source": "Source",
                "location": "Location",
            })
            history_df = history_df.drop(columns=["hash"], errors="ignore")
            st.dataframe(history_df.iloc[::-1], use_container_width=True, hide_index=True)

            st.info(
                "The same uploaded statement is automatically reused during this session, "
                "so changing tabs or interacting with the app does not re-run PDF extraction."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "FORENSIC INTELLIGENCE • "
    "Evidence-first analysis • "
    "Human review required for investigative conclusions"
)
