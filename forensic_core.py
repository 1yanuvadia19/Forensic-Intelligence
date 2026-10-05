import io,re
from pathlib import Path
import pandas as pd
import numpy as np

CANON=["Date","Value_Date","Narration","Reference","Debit","Credit","Balance"]

def money(x):
    if pd.isna(x): return np.nan
    if isinstance(x,(int,float,np.integer,np.floating)): return float(x)
    s=str(x).replace(",","").replace("₹","").replace("INR","").strip()
    if s in ("","-","—","nan","None"): return np.nan
    m=re.search(r"-?\d+(?:\.\d+)?",s)
    return float(m.group()) if m else np.nan

def norm(s):
    return re.sub(r"[^a-z0-9]+"," ",str(s).lower()).strip()

def detect_header(raw):
    best=(-1,None)
    targets=["post date","date","value date","description","narration","particular",
             "remarks","cheque no","reference","debit","withdrawal","credit","deposit","balance"]
    for i in range(min(len(raw),40)):
        vals=[norm(v) for v in raw.iloc[i].tolist()]
        score=sum(any(t in v for t in targets) for v in vals)
        if score>best[0]: best=(score,i)
    if best[0]<3: raise ValueError("Could not confidently locate the transaction header row.")
    return best[1],best[0]

def pick(cols, patterns):
    best=(0,None)
    for c in cols:
        n=norm(c)
        score=max([len(p) for p in patterns if p in n] or [0])
        if score>best[0]: best=(score,c)
    return best[1]

def standardize(raw):
    raw=raw.dropna(how="all").dropna(axis=1,how="all").copy()
    if raw.empty: raise ValueError("Empty worksheet.")
    hi,score=detect_header(raw)
    df=raw.iloc[hi+1:].copy()
    df.columns=[str(x).strip() for x in raw.iloc[hi].tolist()]
    df=df.dropna(how="all")
    cols=df.columns
    date=pick(cols,["post date","transaction date","txn date","date"])
    value=pick(cols,["value date"])
    narr=pick(cols,["description","narration","particular","remarks","details"])
    ref=pick(cols,["cheque no","reference","ref no","utr","transaction id","txn id"])
    debit=pick(cols,["debit rs","debit","withdrawal","withdraw"])
    credit=pick(cols,["credit rs","credit","deposit"])
    bal=pick(cols,["balance rs","balance","closing balance"])
    if not narr: raise ValueError("Narration/Description column not found.")
    out=pd.DataFrame()
    out["Date"]=pd.to_datetime(df[date],errors="coerce",dayfirst=True) if date else pd.NaT
    out["Value_Date"]=pd.to_datetime(df[value],errors="coerce",dayfirst=True) if value else pd.NaT
    out["Narration"]=df[narr].fillna("").astype(str).str.strip()
    out["Reference"]=df[ref].fillna("").astype(str).str.strip() if ref else ""
    out["Debit"]=df[debit].map(money).fillna(0).abs() if debit else 0.0
    out["Credit"]=df[credit].map(money).fillna(0).abs() if credit else 0.0
    out["Balance"]=df[bal].map(money) if bal else np.nan
    keep=(out["Debit"]+out["Credit"]>0)|out["Balance"].notna()|out["Date"].notna()
    out=out[keep].reset_index(drop=True)
    if len(out)==0: raise ValueError("Header found, but no transaction rows with dates/amounts were detected.")
    return out[CANON],hi,score

def rail(t):
    u=str(t).upper()
    rules=[
      ("UPI",r"\bUPI\b|UPI/|BHIM|PHONEPE|GPAY|GOOGLEPAY|PAYTM|VPA"),
      ("IMPS",r"\bIMPS\b"),
      ("NEFT",r"\bNEFT\b"),
      ("RTGS",r"\bRTGS\b"),
      ("NACH/ECS/ACH",r"\bNACH\b|\bECS\b|\bACH\b|ACHDR|NACHDR"),
      ("ATM/CASH",r"\bATM\b|CASH WITHDRAWAL|CASH DEPOSIT"),
      ("CHEQUE",r"\bCHQ\b|\bCHEQUE\b|\bCTS\b|CAS PRES"),
      ("CARD",r"CREDIT CARD|DEBIT CARD|CARD PAYMENT"),
      ("BROKING/INVESTMENT",r"BROKING|BROKER|ZERODHA|ANGEL ONE|ANGELONE|SECURITIES"),
      ("INSURANCE/PREMIUM",r"PREMIUM|INSURANCE|LIC"),
      ("LOAN/FINANCE",r"LOAN|HOUSINGFINA|PIRAMAL|EMI"),
      ("BANK CHARGES",r"BANK CHARGES|CHARGES|GST ON"),
      ("INTEREST",r"INTEREST"),
      ("RECHARGE/UTILITY",r"RECHARGE|JIO|VODAFONE|ELECTRIC"),
      ("TRANSFER",r"\bTFR\b|TRANSFER|TRF"),
    ]
    for n,p in rules:
        if re.search(p,u): return n
    return "OTHER / UNIDENTIFIED"

def category(t,d,c):
    u=str(t).upper()
    if "CASH DEPOSIT" in u:return "Cash Deposit"
    if "CASH WITHDRAWAL" in u:return "Cash Withdrawal"
    if "BANK CHARGES" in u:return "Bank Charges"
    if "INTEREST" in u:return "Interest"
    if "PREMIUM" in u or "LIC" in u:return "Insurance / Premium"
    if "LOAN" in u or "HOUSINGFINA" in u or "PIRAMAL" in u or "EMI" in u:return "Loan / Finance"
    if "BROKING" in u or "ANGEL ONE" in u or "ZERODHA" in u or "SECURITIES" in u:return "Investment / Broking"
    if "CREDIT CARD" in u:return "Credit Card"
    if "RECHARGE" in u or "JIO" in u or "VODAFONE" in u:return "Recharge / Utility"
    if "ONLINE" in u or "AMAZON" in u or "FLIPKART" in u or "SHOPSY" in u:return "Online / Merchant"
    if "SELF" in u:return "Self / Internal"
    return "Other Credit / Receipt" if c>0 and d==0 else ("Other Debit / Expense" if d>0 and c==0 else "Other")

def counterparty(t):
    t=str(t).strip()
    if not t or t.upper() in {"CASH DEPOSIT","CASH WITHDRAWAL","SELF","BANK CHARGES","INTEREST CREDIT"}: return ""
    # Preserve full named narration for SBI-style statement entries; later entity extraction can refine it.
    return t[:160]

def enrich(df):
    x=df.copy()
    x["Counterparty"]=x["Narration"].map(counterparty)
    x["Payment_Rail"]=[rail(v) for v in x["Narration"]]
    x["Category"]=[category(t,d,c) for t,d,c in zip(x.Narration,x.Debit,x.Credit)]
    x["Abs_Amount"]=x.Debit+x.Credit
    q95=x.Abs_Amount[x.Abs_Amount>0].quantile(.95)
    q99=x.Abs_Amount[x.Abs_Amount>0].quantile(.99)
    reasons=[]; pri=[]
    for _,r in x.iterrows():
        rs=[]
        if r.Abs_Amount>=q99 and r.Abs_Amount>0: rs.append("Top 1% by transaction amount")
        elif r.Abs_Amount>=q95 and r.Abs_Amount>0: rs.append("Top 5% by transaction amount")
        if r.Payment_Rail=="OTHER / UNIDENTIFIED": rs.append("Payment rail not identified from narration")
        p="NORMAL"
        if rs:p="REVIEW"
        if r.Abs_Amount>=max(q99,1000000):p="CRITICAL"
        reasons.append("; ".join(rs));pri.append(p)
    x["Flag_Reason"]=reasons;x["Priority"]=pri;x["Rapid_Movement"]=False
    # Date-level rapid onward movement.
    cr=x[x.Credit>0]
    dr=x[x.Debit>0]
    for ci,c in cr.iterrows():
        if c.Credit<100000:continue
        cand=dr[(dr.Date>=c.Date)&(dr.Date<=c.Date+pd.Timedelta(days=1))] if pd.notna(c.Date) else dr.iloc[0:0]
        hits=cand[cand.Debit>=c.Credit*.8]
        if len(hits):
            x.loc[ci,"Rapid_Movement"]=True
            x.loc[hits.index,"Rapid_Movement"]=True
    x.loc[x.Rapid_Movement,"Flag_Reason"]=x.loc[x.Rapid_Movement,"Flag_Reason"].astype(str).str.strip("; ")+"; Same/next-day large onward movement"
    x.loc[x.Rapid_Movement,"Priority"]="CRITICAL"
    return x.drop(columns=["Abs_Amount"])

def analyze_excel(data,name):
    book=pd.ExcelFile(io.BytesIO(data))
    frames=[]; details=[]
    for sheet in book.sheet_names:
        raw=pd.read_excel(io.BytesIO(data),sheet_name=sheet,header=None)
        try:
            d,hi,score=standardize(raw)
            d["Source_Sheet"]=sheet
            d["Source_Row"]=range(hi+2,hi+2+len(d))
            frames.append(d);details.append((sheet,hi+1,score,len(d)))
        except Exception:
            continue
    if not frames: raise ValueError("No sheet contained a confident transaction table.")
    df=pd.concat(frames,ignore_index=True)
    df=enrich(df)
    flags=df[df.Priority.isin(["REVIEW","CRITICAL"])].copy()
    meta={"source_type":"Excel/CSV","location":f"{len(book.sheet_names)} sheet(s); {', '.join(s for s,_,_,_ in details)}",
          "layout_confidence":f"{max(x[2] for x in details)} matched header fields","warnings":[]}
    return {"transactions":df,"flags":flags,"meta":meta}

def analyze_pdf(data,name):
    import fitz
    doc=fitz.open(stream=data,filetype="pdf")
    pages=len(doc)
    texts=[p.get_text("text") for p in doc]
    nonempty=sum(bool(t.strip()) for t in texts)
    if nonempty<max(1,int(pages*.5)):
        raise ValueError(f"{pages}-page PDF appears scanned/image-based ({nonempty} pages contain extractable text). OCR + table reconstruction is required; the app will not invent rows from images.")
    rows=[]
    for pn,text in enumerate(texts,1):
        for ln in text.splitlines():
            ln=re.sub(r"\s+"," ",ln).strip()
            m=re.search(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",ln)
            if m and re.search(r"\d",ln):
                rows.append([m.group(),m.group(),ln,"",0,0,np.nan,pn])
    if not rows: raise ValueError("Digital PDF text exists, but a reliable transaction table could not be reconstructed.")
    raw=pd.DataFrame(rows,columns=CANON+["Source_Page"])
    df=enrich(raw)
    flags=df[df.Priority.isin(["REVIEW","CRITICAL"])].copy()
    meta={"source_type":f"PDF digital text","location":f"{pages} pages","layout_confidence":"Conservative text extraction",
          "warnings":["PDF extraction is conservative; scanned/image PDFs require OCR/table reconstruction."]}
    return {"transactions":df,"flags":flags,"meta":meta}

def analyze_upload(data,name):
    ext=Path(name).suffix.lower()
    if ext==".pdf": return analyze_pdf(data,name)
    if ext==".csv": return analyze_excel(data,name)
    return analyze_excel(data,name)

def build_workbook(df,flags,meta):
    out=io.BytesIO()
    summary=pd.DataFrame({"Metric":["Source type","Location","Transactions","Credits","Debits","Net Flow","Review","Critical","Balance mismatches"],
                          "Value":[meta["source_type"],meta["location"],len(df),df.Credit.sum(),df.Debit.sum(),
                                   df.Credit.sum()-df.Debit.sum(),(df.Priority=="REVIEW").sum(),
                                   (df.Priority=="CRITICAL").sum(),balance_mismatches(df)]})
    rails=df.groupby("Payment_Rail").agg(Transactions=("Narration","size"),Credits=("Credit","sum"),Debits=("Debit","sum")).reset_index()
    cats=df.groupby("Category").agg(Transactions=("Narration","size"),Credits=("Credit","sum"),Debits=("Debit","sum")).reset_index()
    with pd.ExcelWriter(out,engine="openpyxl") as w:
        summary.to_excel(w,index=False,sheet_name="01_Summary")
        df.to_excel(w,index=False,sheet_name="02_Normalised_Transactions")
        flags.to_excel(w,index=False,sheet_name="03_Review_Queue")
        rails.to_excel(w,index=False,sheet_name="04_Payment_Rails")
        cats.to_excel(w,index=False,sheet_name="05_Categories")
        if "Counterparty" in df:
            cp=df[df.Counterparty!=""].groupby("Counterparty").agg(Transactions=("Narration","size"),Credits=("Credit","sum"),Debits=("Debit","sum")).sort_values("Credits",ascending=False).reset_index()
            cp.to_excel(w,index=False,sheet_name="06_Counterparties")
        df.groupby(df.Date.dt.to_period("M").astype(str)).agg(Transactions=("Narration","size"),Credits=("Credit","sum"),Debits=("Debit","sum")).reset_index().to_excel(w,index=False,sheet_name="07_Monthly_Flow")
        balance_check(df).to_excel(w,index=False,sheet_name="08_Balance_Check")
        pd.DataFrame({"Note":["Generated from source evidence; verify flagged items against the original statement before forensic use."]}).to_excel(w,index=False,sheet_name="09_Notes")
        for ws in w.book.worksheets:
            ws.freeze_panes="A2"; ws.auto_filter.ref=ws.dimensions
            for c in ws[1]: c.font=__import__("openpyxl").styles.Font(bold=True)
            for col in ws.columns:
                letter=__import__("openpyxl").utils.get_column_letter(col[0].column)
                ws.column_dimensions[letter].width=min(max(len(str(col[0].value or ""))+2,12),42)
    out.seek(0);return out.getvalue()

def balance_check(df):
    rows=[]
    for i in range(1,len(df)):
        p=df.iloc[i-1];c=df.iloc[i]
        if pd.notna(p.Balance) and pd.notna(c.Balance):
            expected=p.Balance+c.Credit-c.Debit
            if abs(expected-c.Balance)>0.01:
                rows.append({"Source_Row":c.get("Source_Row",""),"Date":c.Date,"Narration":c.Narration,"Expected":expected,"Reported":c.Balance,"Difference":c.Balance-expected})
    return pd.DataFrame(rows)

def balance_mismatches(df): return len(balance_check(df))
