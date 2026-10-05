import io
from pathlib import Path

import pandas as pd
import streamlit as st

from forensic_core import analyze_upload, build_workbook, build_master_analysis, build_pdf_report


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Forensic Intelligence 360°",
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
        <h1>🔎 FORENSIC INTELLIGENCE 360°</h1>
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
            "Scanned PDFs are detected separately and require OCR."
        ),
    )

    st.divider()

    st.markdown("### Pipeline")

    pipeline = [
        "1. Detect layout",
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
        - Scanned/image PDFs are detected and blocked until OCR
          reconstruction is available.

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

st.caption(
    f"Evidence loaded: **{file_name}** • "
    f"{len(file_bytes):,} bytes"
)


with st.spinner("Analysing evidence..."):

    try:

        result = analyze_upload(
            file_bytes,
            file_name,
        )

    except Exception as exc:

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
        "📤 Export",
    ]
)


# ============================================================
# COMMAND CENTER
# ============================================================

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

    st.dataframe(
        view,
        use_container_width=True,
        height=600,
    )


# ============================================================
# RISK
# ============================================================

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

        st.dataframe(
            flags,
            use_container_width=True,
            height=600,
        )


# ============================================================
# ANNEXURES
# ============================================================

with tabs[3]:

    st.subheader("Investigation Annexures")

    annexure_options = [
        "Normalised Transactions",
        "Review Queue",
        "Payment Rails",
        "Categories",
        "Counterparties",
        "Monthly Flow",
        "Balance Check",
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
# EXPORT
# ============================================================

with tabs[5]:

    st.subheader("Evidence-Ready Export")
    st.caption("Download the concise investigation workbook or executive forensic report.")

    try:
        workbook = build_workbook(df, flags, meta)
        pdf_report = build_pdf_report(df, flags, meta, file_name)
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

    except Exception as exc:
        st.error(f"Could not generate the export: {exc}")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "FORENSIC INTELLIGENCE 360° • "
    "Evidence-first analysis • "
    "Human review required for investigative conclusions"
)
