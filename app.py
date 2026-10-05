from pathlib import Path
import streamlit as st
import pandas as pd
from forensic_core import analyze_upload, build_workbook

st.set_page_config(page_title="Forensic Intelligence 360°", page_icon="🔎", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1.1rem}
.hero{padding:24px 30px;border-radius:20px;background:linear-gradient(135deg,#15243a,#24476f);color:#fff;margin-bottom:18px}
.hero h1{margin:0;font-size:2.25rem}.hero p{margin:.45rem 0 0;color:#d9e7f8}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><h1>🔎 FORENSIC INTELLIGENCE 360°</h1>
<p>Bank-statement intelligence • payment-rail parsing • anomaly screening • evidence-ready exports</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📁 Evidence Intake")
    up = st.file_uploader("Upload bank statement", type=["xlsx","xls","csv","pdf"])
    st.divider()
    st.markdown("### Pipeline")
    for x in ["1. Detect layout","2. Extract transactions","3. Identify payment rails",
              "4. Screen anomalies","5. Reconcile balances","6. Export evidence"]:
        st.markdown(x)
    st.divider()
    st.caption("Deterministic analysis works without an API key.")

if not up:
    st.info("Upload an Excel/CSV/PDF bank statement to begin.")
    st.stop()

try:
    with st.spinner("Reading evidence and building the forensic dataset..."):
        result = analyze_upload(up.getvalue(), up.name)

    df, flags, meta = result["transactions"], result["flags"], result["meta"]

    st.success(f"Evidence loaded: {len(df):,} transactions • {meta['source_type']} • {meta['location']}")

    a,b,c,d,e,f = st.columns(6)
    a.metric("Transactions", f"{len(df):,}")
    b.metric("Credits", f"₹{df['Credit'].sum():,.0f}")
    c.metric("Debits", f"₹{df['Debit'].sum():,.0f}")
    d.metric("Net Flow", f"₹{df['Credit'].sum()-df['Debit'].sum():,.0f}")
    e.metric("Review", f"{(df['Priority']=='REVIEW').sum():,}")
    f.metric("Critical", f"{(df['Priority']=='CRITICAL').sum():,}")

    tabs=st.tabs(["📊 Command Center","🔎 Transactions","🚨 Risk","🏦 Counterparties","📑 Annexures","⬇️ Export"])
    with tabs[0]:
        st.subheader("Payment rails")
        st.bar_chart(df["Payment_Rail"].value_counts())
        st.subheader("Categories")
        st.bar_chart(df["Category"].value_counts())
        st.caption(f"Layout detection confidence: {meta.get('layout_confidence','N/A')}")
        if meta.get("warnings"):
            for w in meta["warnings"]: st.warning(w)

    with tabs[1]:
        q=st.text_input("Search narration / counterparty / reference")
        v=df
        if q:
            mask=v.astype(str).apply(lambda s:s.str.contains(q,case=False,na=False)).any(axis=1)
            v=v[mask]
        st.dataframe(v,use_container_width=True,height=560)

    with tabs[2]:
        st.dataframe(flags,use_container_width=True,height=560)

    with tabs[3]:
        cp=(df[df["Counterparty"].ne("")]
            .groupby("Counterparty")
            .agg(Transactions=("Counterparty","size"),Credits=("Credit","sum"),Debits=("Debit","sum"))
            .sort_values(["Credits","Debits"],ascending=False))
        st.dataframe(cp,use_container_width=True,height=500)

    with tabs[4]:
        for cat in sorted(df["Category"].dropna().unique()):
            sub=df[df["Category"]==cat]
            with st.expander(f"{cat} — {len(sub):,}"):
                st.dataframe(sub,use_container_width=True,height=250)

    with tabs[5]:
        wb=build_workbook(df,flags,meta)
        st.download_button("⬇️ Download Master Forensic Workbook",wb,
            file_name=f"Forensic_Master_{Path(up.name).stem}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
        st.download_button("⬇️ Download Normalised CSV",df.to_csv(index=False).encode(),
            file_name=f"Normalised_{Path(up.name).stem}.csv",mime="text/csv",
            use_container_width=True)

except Exception as ex:
    st.error(f"Processing stopped safely: {type(ex).__name__}: {ex}")
    st.info("No partial forensic output is presented when the input layout cannot be read confidently.")
