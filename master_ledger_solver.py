"""Evidence-constrained ledger direction/order solver.

This module does not invent transaction amounts. It treats Debit/Credit direction
and statement order as hypotheses and selects the hypothesis that best agrees
with the balances printed in the source document.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def _num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _delta(row, swapped=False):
    d = _num(row.get("Debit"))
    c = _num(row.get("Credit"))
    if pd.isna(d) and pd.isna(c):
        return np.nan
    if pd.notna(d) and pd.notna(c):
        return d - c if swapped else c - d
    if pd.notna(d):
        return d if swapped else -d
    return -c if swapped else c


def _apply_side(row, swapped):
    d = _num(row.get("Debit"))
    c = _num(row.get("Credit"))
    if not swapped:
        return d, c
    return c, d


def _order_variants(df):
    x = df.copy().reset_index(drop=True)
    variants = []
    if "Source_Seq" in x.columns:
        variants.append(("source", x.sort_values("Source_Seq", kind="stable").reset_index(drop=True)))
        variants.append(("source-reverse", x.sort_values("Source_Seq", kind="stable", ascending=False).reset_index(drop=True)))
    if "Date" in x.columns:
        d = pd.to_datetime(x["Date"], errors="coerce")
        x2 = x.assign(_d=d)
        variants.append(("date-ascending", x2.sort_values("_d", kind="stable").drop(columns=["_d"]).reset_index(drop=True)))
        variants.append(("date-descending", x2.sort_values("_d", kind="stable", ascending=False).drop(columns=["_d"]).reset_index(drop=True)))
    out, seen = [], set()
    for name, frame in variants:
        if "Source_Seq" in frame.columns:
            key = tuple(frame["Source_Seq"].tolist())
        elif "Date" in frame.columns:
            key = tuple(pd.to_datetime(frame["Date"], errors="coerce").astype("int64").tolist())
        else:
            key = tuple(range(len(frame)))
        if key not in seen:
            seen.add(key)
            out.append((name, frame))
    return out or [("input", x)]


def _score_order(df, opening_balance=None, summary=None):
    n = len(df)
    if n == 0:
        return df.copy(), {"matches": 0, "known": 0, "side_swaps": 0, "score": -1e9}

    if pd.notna(opening_balance):
        current = {round(float(opening_balance), 2): (0.0, 0, 0, [])}
    else:
        current = {None: (0.0, 0, 0, [])}

    for i in range(n):
        row = df.iloc[i]
        bal = _num(row.get("Balance"))
        hypotheses = [False]
        if pd.notna(_num(row.get("Debit"))) or pd.notna(_num(row.get("Credit"))):
            hypotheses.append(True)

        next_states = {}
        for prev_balance, (score, matches, swaps, path) in current.items():
            for swapped in hypotheses:
                delta = _delta(row, swapped)
                local = 0.0
                local_match = 0
                local_swap = 1 if swapped else 0

                if pd.notna(bal) and prev_balance is not None and pd.notna(delta):
                    expected = prev_balance + delta
                    err = abs(expected - float(bal))
                    if err <= 0.01:
                        local += 45.0
                        local_match = 1
                    else:
                        local -= min(80.0, 5.0 + err / 1000.0)
                elif pd.isna(bal):
                    local += 1.0
                else:
                    local -= 1.0

                if swapped:
                    local -= 4.0

                new_balance = round(float(bal), 2) if pd.notna(bal) else (
                    round(prev_balance + delta, 2)
                    if prev_balance is not None and pd.notna(delta)
                    else prev_balance
                )
                key = new_balance
                value = (
                    score + local,
                    matches + local_match,
                    swaps + local_swap,
                    path + [swapped],
                )
                old = next_states.get(key)
                if old is None or value[0] > old[0]:
                    next_states[key] = value

        current = next_states or current

    best = max(current.values(), key=lambda v: v[0])
    path = best[3]
    out = df.copy().reset_index(drop=True)

    for i, swapped in enumerate(path[:len(out)]):
        if not swapped:
            continue
        d, c = _apply_side(out.iloc[i], True)
        out.at[i, "Debit"] = d
        out.at[i, "Credit"] = c
        out.at[i, "Direction_Resolved_By"] = "balance-chain side hypothesis"

    known = int(
        pd.to_numeric(out["Balance"], errors="coerce").notna().sum()
    ) if "Balance" in out else 0
    matches = int(best[1])
    score = best[0]

    if summary:
        closing = _num(summary.get("closing"))
        if pd.notna(closing) and known:
            last_bal = _num(out.iloc[-1].get("Balance"))
            if pd.notna(last_bal) and abs(last_bal - closing) <= 0.01:
                score += 80.0
                matches += 1
            else:
                score -= 60.0

    return out, {
        "matches": matches,
        "known": known,
        "side_swaps": int(best[2]),
        "score": score,
    }


def solve_ledger(df, opening_balance=np.nan, summary=None):
    """Choose order + side hypotheses using only source evidence.

    Returns the chosen dataframe and diagnostics. No monetary value is created
    or altered; only a Debit/Credit label may be swapped when the printed
    running balance proves the alternative direction is more consistent.
    """
    if df is None or df.empty:
        return df, {
            "order": "empty",
            "matches": 0,
            "known": 0,
            "side_swaps": 0,
            "score": -1e9,
        }

    best = None
    for name, frame in _order_variants(df):
        candidate, stats = _score_order(
            frame, opening_balance=opening_balance, summary=summary
        )
        stats = dict(stats, order=name)
        if best is None or stats["score"] > best[1]["score"]:
            best = (candidate, stats)

    return best
