"""Forensic Bridge Engine
Deterministic investigation layer built on verified bank transactions.

This module does not decide fraud. It reconstructs observable relationships:
source-to-destination candidates, flow chains, circularity, dormant activation,
counterparty concentration and behavioural deviations. Every finding keeps
source rows/dates/amounts so an investigator can trace it back to evidence.
"""
import re
import numpy as np
import pandas as pd


def _num(v):
    try:
        return float(v) if pd.notna(v) else 0.0
    except Exception:
        return 0.0


def _party(v):
    s = re.sub(r"[^A-Z0-9 ]+", " ", str(v or "").upper())
    s = re.sub(r"\s+", " ", s).strip()
    return s if s and s not in {"-", "UNKNOWN", "N A", "NA"} else ""


def _fingerprint(row):
    date = pd.Timestamp(row["Date"]).strftime("%Y%m%d") if pd.notna(row.get("Date")) else "NA"
    amount = round(_num(row.get("Debit")) + _num(row.get("Credit")), 2)
    rail = str(row.get("Payment_Rail") or "UNKNOWN").upper()
    party = _party(row.get("Counterparty"))
    ref = re.sub(r"\s+", "", str(row.get("Reference") or ""))[:32].upper()
    return f"{date}|{amount:.2f}|{rail}|{party}|{ref}"


def _source_row(row, fallback):
    return row.get("Source_Row", fallback)


def build_forensic_bridge(df, max_days=3, tolerance=0.03, min_amount=50000):
    """Return evidence-linked investigative tables.

    tolerance is relative amount tolerance (3% default). It is a candidate
    matching tolerance only; it never changes source amounts.
    """
    if df is None or df.empty:
        empty = pd.DataFrame()
        return {
            "bridges": empty, "chains": empty, "circles": empty,
            "dormant": empty, "behaviour": empty, "fingerprints": empty,
            "questions": empty,
        }

    x = df.copy().reset_index(drop=True)
    x["_debit"] = pd.to_numeric(x["Debit"], errors="coerce").fillna(0.0)
    x["_credit"] = pd.to_numeric(x["Credit"], errors="coerce").fillna(0.0)
    x["_date"] = pd.to_datetime(x["Date"], errors="coerce")
    x["_party"] = x.get("Counterparty", pd.Series([""] * len(x))).map(_party)
    x["_fp"] = [_fingerprint(r) for _, r in x.iterrows()]

    # ------------------------------------------------------------------
    # 1. SOURCE -> DESTINATION BRIDGE
    # A credit is linked to later debits of similar magnitude. This is a
    # candidate trail, not a claim that the money is the same legal funds.
    # ------------------------------------------------------------------
    credits = x[x["_credit"] >= min_amount]
    debits = x[x["_debit"] >= min_amount]
    bridges = []
    for ci, cr in credits.iterrows():
        if pd.isna(cr["_date"]):
            continue
        amount = cr["_credit"]
        lo, hi = amount * (1 - tolerance), amount * (1 + tolerance)
        candidates = debits[
            (debits["_date"] >= cr["_date"]) &
            (debits["_date"] <= cr["_date"] + pd.Timedelta(days=max_days)) &
            (debits["_debit"] >= lo) &
            (debits["_debit"] <= hi)
        ]
        for di, dr in candidates.iterrows():
            days = (dr["_date"] - cr["_date"]).total_seconds() / 86400
            ratio = dr["_debit"] / amount if amount else np.nan
            party_link = bool(cr["_party"] and dr["_party"] and cr["_party"] == dr["_party"])
            score = 0
            if abs(ratio - 1) <= 0.01: score += 40
            elif abs(ratio - 1) <= 0.03: score += 25
            if days <= 1: score += 30
            else: score += 15
            if party_link: score += 20
            bridges.append({
                "Source_Date": cr["_date"], "Source_Amount": amount,
                "Source_Party": cr["_party"] or "-",
                "Source_Reference": str(cr.get("Reference", "")),
                "Source_Row": _source_row(cr, ci + 1),
                "Destination_Date": dr["_date"], "Destination_Amount": dr["_debit"],
                "Destination_Party": dr["_party"] or "-",
                "Destination_Reference": str(dr.get("Reference", "")),
                "Destination_Row": _source_row(dr, di + 1),
                "Days_Gap": round(days, 3),
                "Amount_Ratio": round(ratio, 4),
                "Bridge_Score": score,
                "Bridge_Type": "NEAR-EXACT ONWARD MOVEMENT" if abs(ratio - 1) <= 0.03 else "SIMILAR-VALUE ONWARD MOVEMENT",
            })
    bridge_df = pd.DataFrame(bridges)
    if not bridge_df.empty:
        bridge_df = bridge_df.sort_values(["Bridge_Score", "Source_Date"], ascending=[False, True]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. COUNTERPARTY FLOW CHAINS
    # Build compact chains from bridge candidates: A credit followed by B debit.
    # ------------------------------------------------------------------
    chains = []
    if not bridge_df.empty:
        for _, b in bridge_df.head(250).iterrows():
            chains.append({
                "Chain": f"CREDIT [{b.Source_Row}] -> DEBIT [{b.Destination_Row}]",
                "Start_Date": b.Source_Date,
                "Start_Amount": b.Source_Amount,
                "End_Date": b.Destination_Date,
                "End_Amount": b.Destination_Amount,
                "Days": b.Days_Gap,
                "Source_Row": b.Source_Row,
                "Destination_Row": b.Destination_Row,
                "Interpretation": "Potential onward movement; trace source documents/counterparty relationship",
                "Confidence": "HIGH" if b.Bridge_Score >= 70 else "MEDIUM",
            })
    chain_df = pd.DataFrame(chains)

    # ------------------------------------------------------------------
    # 3. CIRCULAR / ROUND-TRIP COUNTERPARTY FLOWS
    # Same counterparties appearing on both sides with temporal proximity.
    # ------------------------------------------------------------------
    circles = []
    cp = x[x["_party"] != ""]
    for party, g in cp.groupby("_party"):
        ins = g[g["_credit"] >= min_amount]
        outs = g[g["_debit"] >= min_amount]
        for _, inc in ins.iterrows():
            for _, out in outs.iterrows():
                if pd.isna(inc["_date"]) or pd.isna(out["_date"]) or out["_date"] < inc["_date"]:
                    continue
                days = (out["_date"] - inc["_date"]).total_seconds() / 86400
                ratio = out["_debit"] / inc["_credit"] if inc["_credit"] else 0
                if days <= max_days and 0.90 <= ratio <= 1.10:
                    circles.append({
                        "Counterparty": party,
                        "Credit_Date": inc["_date"], "Credit_Amount": inc["_credit"],
                        "Credit_Source_Row": _source_row(inc, 0),
                        "Debit_Date": out["_date"], "Debit_Amount": out["_debit"],
                        "Debit_Source_Row": _source_row(out, 0),
                        "Days_Gap": round(days, 3),
                        "Amount_Ratio": round(ratio, 4),
                        "Finding": "Same-counterparty near-equal inflow/outflow; investigate round-trip possibility",
                    })
    circle_df = pd.DataFrame(circles).drop_duplicates() if circles else pd.DataFrame()

    # ------------------------------------------------------------------
    # 4. DORMANT ACTIVATION
    # Detect long inactivity followed by unusually large movement.
    # ------------------------------------------------------------------
    dormant = []
    ordered = x.sort_values("_date")
    valid = ordered[ordered["_date"].notna()]
    if len(valid):
        previous_date = None
        baseline = []
        for idx, r in valid.iterrows():
            if previous_date is not None:
                gap = (r["_date"] - previous_date).days
                amount = r["_debit"] + r["_credit"]
                if gap >= 90 and amount >= max(min_amount, 100000):
                    dormant.append({
                        "Activation_Date": r["_date"],
                        "Dormancy_Days": gap,
                        "Amount": amount,
                        "Direction": "CREDIT" if r["_credit"] > 0 else "DEBIT",
                        "Source_Row": _source_row(r, idx + 1),
                        "Reference": str(r.get("Reference", "")),
                        "Finding": "Large movement after >=90-day transaction inactivity",
                    })
            previous_date = r["_date"]
    dormant_df = pd.DataFrame(dormant)

    # ------------------------------------------------------------------
    # 5. BEHAVIOURAL BASELINE
    # Counterparty-specific median vs current amount.
    # ------------------------------------------------------------------
    behaviour = []
    for party, g in cp.groupby("_party"):
        amounts = (g["_debit"] + g["_credit"]).replace(0, np.nan).dropna()
        if len(amounts) < 3:
            continue
        median = float(amounts.median())
        for idx, r in g.iterrows():
            amount = r["_debit"] + r["_credit"]
            if amount >= min_amount and median > 0 and amount >= median * 3:
                behaviour.append({
                    "Counterparty": party,
                    "Date": r["_date"],
                    "Amount": amount,
                    "Typical_Median": round(median, 2),
                    "Deviation_x": round(amount / median, 2),
                    "Source_Row": _source_row(r, idx + 1),
                    "Finding": "Transaction materially exceeds this counterparty's observed baseline",
                })
    behaviour_df = pd.DataFrame(behaviour)

    # ------------------------------------------------------------------
    # 6. TRANSACTION FINGERPRINTS
    # Duplicate fingerprints are a data-quality / repeated-entry signal.
    # ------------------------------------------------------------------
    fp_counts = x["_fp"].value_counts()
    fp = x[x["_fp"].isin(fp_counts[fp_counts > 1].index)].copy()
    fingerprints = []
    for _, r in fp.iterrows():
        fingerprints.append({
            "Fingerprint": r["_fp"],
            "Date": r["_date"],
            "Amount": r["_debit"] + r["_credit"],
            "Counterparty": r["_party"] or "-",
            "Source_Row": _source_row(r, 0),
            "Occurrence_Count": int(fp_counts[r["_fp"]]),
            "Finding": "Repeated transaction fingerprint; verify whether genuine recurrence or duplicate entry",
        })
    fingerprint_df = pd.DataFrame(fingerprints)

    # ------------------------------------------------------------------
    # 7. INVESTIGATION QUESTIONS — bridge from analytics to actual forensic work.
    # ------------------------------------------------------------------
    questions = []
    if not bridge_df.empty:
        questions.append(["Fund trail", len(bridge_df), "Which source credit is followed by a near-equal debit, and what supporting invoice/loan/asset document explains the movement?"])
    if not circle_df.empty:
        questions.append(["Circularity", len(circle_df), "Do the near-equal inflow/outflow pairs represent genuine consideration or circular movement between related parties?"])
    if not dormant_df.empty:
        questions.append(["Dormant activation", len(dormant_df), "What changed immediately before the account became active after the long inactivity period?"])
    if not behaviour_df.empty:
        questions.append(["Behaviour deviation", len(behaviour_df), "Why do these transactions materially exceed the counterparty's normal observed transaction size?"])
    if not fingerprint_df.empty:
        questions.append(["Repeated fingerprint", len(fingerprint_df), "Are repeated transaction fingerprints genuine recurring payments or duplicate/constructed entries?"])
    if not questions:
        questions.append(["No bridge anomaly", 0, "No configured bridge anomaly was identified; absence of a flag is not proof of absence of irregularity."])
    question_df = pd.DataFrame(questions, columns=["Investigation_Area", "Cases", "Question_for_Investigator"])

    return {
        "bridges": bridge_df,
        "chains": chain_df,
        "circles": circle_df,
        "dormant": dormant_df,
        "behaviour": behaviour_df,
        "fingerprints": fingerprint_df,
        "questions": question_df,
    }
