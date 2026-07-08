# app.py — Modern Branch Reconciliation App
import io, re, os, tempfile, requests
import pandas as pd
import numpy as np
from datetime import datetime
from difflib import SequenceMatcher
import streamlit as st
import plotly.express as px

# ================= CONFIG =================
ROUND_DP = 2
AMOUNT_TOLERANCE_DEFAULT = 5.0
NAME_SIM_THRESHOLD_DEFAULT = 0.80

st.set_page_config(
    page_title="Branch Reconciliation",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= MODERN CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 45%, #ecfeff 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #1e293b 100%);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.main-header {
    padding: 28px;
    border-radius: 24px;
    background: linear-gradient(135deg, #1e3a8a, #2563eb, #06b6d4);
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 20px 45px rgba(37, 99, 235, 0.25);
}

.main-header h1 {
    margin: 0;
    font-size: 34px;
    font-weight: 800;
}

.main-header p {
    margin-top: 8px;
    font-size: 16px;
    opacity: 0.95;
}

.badge-row {
    margin-top: 18px;
}

.badge {
    display: inline-block;
    padding: 8px 14px;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.28);
    border-radius: 999px;
    font-size: 13px;
    margin-right: 8px;
}

.kpi-card {
    padding: 20px;
    border-radius: 22px;
    background: rgba(255,255,255,0.78);
    border: 1px solid rgba(226,232,240,0.9);
    box-shadow: 0 14px 32px rgba(15,23,42,0.08);
    backdrop-filter: blur(12px);
    min-height: 132px;
}

.kpi-title {
    color: #64748b;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.kpi-value {
    color: #0f172a;
    font-size: 32px;
    font-weight: 800;
    margin-top: 8px;
}

.kpi-note {
    color: #64748b;
    font-size: 13px;
    margin-top: 6px;
}

.section-card {
    padding: 22px;
    border-radius: 24px;
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(226,232,240,0.95);
    box-shadow: 0 14px 34px rgba(15,23,42,0.07);
    margin-bottom: 18px;
}

.status-success {
    border-left: 7px solid #10b981;
}

.status-warning {
    border-left: 7px solid #f59e0b;
}

.status-danger {
    border-left: 7px solid #ef4444;
}

.status-info {
    border-left: 7px solid #3b82f6;
}

div.stButton > button:first-child {
    border-radius: 16px;
    border: none;
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    color: white;
    font-weight: 800;
    padding: 0.75rem 1.2rem;
    box-shadow: 0 10px 24px rgba(37,99,235,0.25);
}

div.stDownloadButton > button:first-child {
    border-radius: 16px;
    border: none;
    background: linear-gradient(135deg, #16a34a, #10b981);
    color: white;
    font-weight: 800;
    padding: 0.75rem 1.2rem;
}

.footer {
    margin-top: 36px;
    padding: 18px;
    border-radius: 18px;
    background: #0f172a;
    color: white;
    text-align: center;
    font-size: 13px;
}

.small-muted {
    color: #64748b;
    font-size: 13px;
}

hr {
    border: none;
    border-top: 1px solid #e2e8f0;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="main-header">
    <h1>🔗 Branch Reconciliation System</h1>
    <p>Issam Kabbani & Partners Unitech — Treasury Reconciliation Dashboard</p>
    <div class="badge-row">
        <span class="badge">✅ Exact Matching</span>
        <span class="badge">🟡 Partial Matching</span>
        <span class="badge">📊 KPI Summary</span>
        <span class="badge">⬇ Excel Export</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= DATE PARSING =================
_MONTHS = {
    'jan':1,'january':1,'feb':2,'february':2,'mar':3,'march':3,'apr':4,'april':4,
    'may':5,'jun':6,'june':6,'jul':7,'july':7,'aug':8,'august':8,
    'sep':9,'sept':9,'september':9,'oct':10,'october':10,
    'nov':11,'november':11,'dec':12,'december':12
}

date_patterns = [
    r'(?P<d>\\d{1,2})[^\\w\\s]?(?P<m>\\d{1,2})[^\\w\\s]?(?P<y>\\d{2,4})',
    r'(?P<y>\\d{4})[^\\w\\s]?(?P<m>\\d{1,2})[^\\w\\s]?(?P<d>\\d{1,2})',
    r'(?P<d>\\d{1,2})\\s+(?P<mon>[A-Za-z]{3,9})\\.?,?\\s+(?P<y>\\d{2,4})',
]

def parse_any_date(val):
    if pd.isna(val):
        return pd.NaT
    s = str(val).strip()
    try:
        return pd.to_datetime(s, errors="raise", dayfirst=True)
    except Exception:
        low = s.lower()
        for pat in date_patterns:
            m = re.search(pat, low)
            if not m:
                continue
            gd = m.groupdict()
            try:
                if 'mon' in gd and gd['mon']:
                    mon = _MONTHS.get(gd['mon'][:3].lower())
                    d, y = int(gd['d']), int(gd['y'])
                    y = y + 2000 if y < 100 else y
                    return pd.Timestamp(datetime(y, mon, d))
                else:
                    d, mth, y = int(gd['d']), int(gd['m']), int(gd['y'])
                    y = y + 2000 if y < 100 else y
                    return pd.Timestamp(datetime(y, mth, d))
            except Exception:
                continue
    return pd.NaT

# ================= UTILITIES =================
STOP_TOKENS_LATIN = {
    "DATE","DOC","BANK","DEBIT","CREDIT","REF","DEP","DEPOSIT","TRANSFER","TFR","PYT",
    "RIB","HSBC","SNB","SABB","BAB","JED","JEDDAH","RIYADH","COMPANY","CO","LTD","ACTUAL",
    "TOSL","PAYMENT","EXIT","ENTRY","RENEWAL","FEE","CHARGE","BONUS","REWARD","RWD",
    "DTD","BREF","B.REF","INV","FT","TT","CHQ","CHEQUE","SADAD","EXP","ERE","PAY"
}

STOP_TOKENS_AR = {
    "بنك","تحويل","حوالة","ايداع","سداد","رسوم","جدة","الرياض",
    "مبلغ","شركة","رقم","دفع","مدفوع"
}

def to_numeric(x):
    return pd.to_numeric(
        pd.Series(x).astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

def normalize_spaces(s: str) -> str:
    return re.sub(r"\\s+", " ", s.strip())

def extract_refs_all(voucher: str, description: str):
    text = f"{voucher or ''} {description or ''}"

    nums = re.findall(r"\\d{6,}", text)
    alnums = re.findall(r"[A-Za-z0-9][A-Za-z0-9_\\-/:]{3,}", text)

    for tok in alnums:
        nums += re.findall(r"\\d{5,}", tok)

    latin_names = re.findall(r"[A-Za-z]{3,}", text)
    arabic_names = re.findall(r"[\\u0600-\\u06FF]{2,}", text)

    num_set = set(nums)
    alnum_set = {t.upper() for t in alnums if t.upper() not in STOP_TOKENS_LATIN}
    name_set = {t.upper() for t in latin_names if t.upper() not in STOP_TOKENS_LATIN}
    name_set |= {t for t in arabic_names if t not in STOP_TOKENS_AR}

    return num_set, alnum_set, name_set

def name_similarity(a_names: set, b_names: set) -> float:
    if not a_names and not b_names:
        return 0.0

    jacc = (
        len(a_names & b_names) / len(a_names | b_names)
        if (a_names or b_names)
        else 0.0
    )

    a_str = normalize_spaces(" ".join(sorted(a_names)))
    b_str = normalize_spaces(" ".join(sorted(b_names)))

    seq = SequenceMatcher(None, a_str, b_str).ratio() if a_str and b_str else 0.0

    return max(jacc, seq)

def prep_sheet(df):
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*names, required=False):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        if required:
            raise ValueError(
                f"Missing required column among {names}. Found: {list(df.columns)}"
            )
        return None

    cDate = pick("date")
    cVno = pick("voucher no.", "voucher no", "voucher", "voucher_no")
    cDesc = pick("description", required=True)
    cDr = pick("debit", "dr")
    cCr = pick("credit", "cr")

    out = pd.DataFrame({
        "Date": df[cDate].apply(parse_any_date) if cDate else pd.NaT,
        "Voucher": df[cVno].astype(str).str.strip() if cVno else "",
        "Description": df[cDesc].astype(str).fillna(""),
        "Debit": to_numeric(df[cDr]) if cDr else 0.0,
        "Credit": to_numeric(df[cCr]) if cCr else 0.0,
    })

    out["Amt"] = (out["Debit"].fillna(0) + out["Credit"].fillna(0)).round(ROUND_DP)

    refs = out.apply(lambda r: extract_refs_all(r["Voucher"], r["Description"]), axis=1)

    out["NumRefs"] = refs.map(lambda t: t[0])
    out["AlnumRefs"] = refs.map(lambda t: t[1])
    out["NameRefs"] = refs.map(lambda t: t[2])
    out["AllRefs"] = out.apply(
        lambda r: set().union(r["NumRefs"], r["AlnumRefs"], r["NameRefs"]),
        axis=1
    )

    return out

def split_sides(df, tag):
    dr = df[df["Debit"] > 0].copy()
    dr["Side"] = f"{tag}-DR"
    dr["Amt"] = dr["Debit"].round(ROUND_DP)

    cr = df[df["Credit"] > 0].copy()
    cr["Side"] = f"{tag}-CR"
    cr["Amt"] = cr["Credit"].round(ROUND_DP)

    for d in (dr, cr):
        d.reset_index(drop=True, inplace=True)
        d["RowID"] = np.arange(len(d))

    return dr, cr

def strip_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()

    for c in out.columns:
        if "date" in c.lower():
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.date

    return out

# ================= SHEET ALIAS =================
def _casefold(s: str) -> str:
    return s.strip().lower()

def _eq(a: str, b: str) -> bool:
    return _casefold(a) == _casefold(b)

OUR_ALIASES = {"our book", "tosl", "ourbook", "our_book"}
BR_ALIASES = {"branch book", "branch", "br", "branchbook", "branch_book"}

def _find_by_alias(target, names):
    if not target:
        return None

    t = _casefold(target)

    for n in names:
        if _eq(n, target):
            return n

    if t in OUR_ALIASES:
        for n in names:
            if _casefold(n) in OUR_ALIASES:
                return n

    if t in BR_ALIASES:
        for n in names:
            if _casefold(n) in BR_ALIASES:
                return n

    return None

def _auto_pick_other(our_name, names):
    for n in names:
        if not _eq(n, our_name) and _casefold(n) in BR_ALIASES:
            return n

    for n in names:
        if not _eq(n, our_name):
            return n

    return None

# ================= MATCHING ENGINE =================
def pair_exact_best(left, right, labelL, labelR, tol, name_thresh=NAME_SIM_THRESHOLD_DEFAULT):
    ref_index = {}

    for j, r in right.iterrows():
        toks = r["AllRefs"]
        if not isinstance(toks, set):
            toks = set() if pd.isna(toks) else set(toks)

        for tok in toks:
            ref_index.setdefault(tok, set()).add(j)

    def candidates(lrow):
        toks = lrow["AllRefs"]
        if not isinstance(toks, set):
            toks = set() if pd.isna(toks) else set(toks)

        cand = set()
        for tok in toks:
            cand |= ref_index.get(tok, set())

        return list(cand) if cand else list(range(len(right)))

    usedL, usedR, matched = set(), set(), []

    for i, l in left.iterrows():
        cands = candidates(l)

        if not cands:
            continue

        def score(j):
            r = right.loc[j]
            num_ov = len(l["NumRefs"] & r["NumRefs"])
            aln_ov = len(l["AlnumRefs"] & r["AlnumRefs"])
            nm_ov = len(l["NameRefs"] & r["NameRefs"])
            nm_sim = name_similarity(l["NameRefs"], r["NameRefs"])
            amt_d = abs(float(l["Amt"]) - float(r["Amt"]))
            d1, d2 = l["Date"], r["Date"]
            d_d = abs((d1 - d2).days) if (pd.notna(d1) and pd.notna(d2)) else 10**9

            return (-num_ov, -(aln_ov + nm_ov * 0.5), -int(nm_sim * 1000), amt_d, d_d)

        cands = sorted(cands, key=score)

        for j in cands:
            if i in usedL or j in usedR:
                continue

            r = right.loc[j]

            num_ov = len(l["NumRefs"] & r["NumRefs"])
            tok_ov = len(l["AlnumRefs"] & r["AlnumRefs"])
            nm_sim = name_similarity(l["NameRefs"], r["NameRefs"])

            if not (num_ov >= 1 or tok_ov >= 1 or nm_sim >= name_thresh):
                continue

            amt_diff = abs(float(l["Amt"]) - float(r["Amt"]))

            if amt_diff <= tol:
                matched.append((l, r, round(amt_diff, ROUND_DP)))
                usedL.add(i)
                usedR.add(j)
                break

    match_df = pd.DataFrame([{
        f"{labelL} Date": a["Date"],
        f"{labelL} Voucher": a["Voucher"],
        f"{labelL} Description": a["Description"],
        f"{labelL} Amount": a["Amt"],
        f"{labelR} Date": b["Date"],
        f"{labelR} Voucher": b["Voucher"],
        f"{labelR} Description": b["Description"],
        f"{labelR} Amount": b["Amt"],
        "Amount_Diff": diff
    } for (a, b, diff) in matched])

    return match_df, usedL, usedR

def pair_exact_best_fast_same(left, right, labelL, labelR, tol, name_thresh=NAME_SIM_THRESHOLD_DEFAULT):
    right_amt = right["Amt"].astype(float).values

    def candidates_amount(l_amt):
        diff = np.abs(right_amt - float(l_amt))
        return list(np.nonzero(diff <= tol)[0])

    usedL, usedR, matched = set(), set(), []

    ref_index = {}

    for j, r in right.iterrows():
        toks = r["AllRefs"]
        if not isinstance(toks, set):
            toks = set() if pd.isna(toks) else set(toks)

        for tok in toks:
            ref_index.setdefault(tok, set()).add(j)

    for i, l in left.iterrows():
        cands = set(candidates_amount(l["Amt"]))

        toks = l["AllRefs"] if isinstance(l["AllRefs"], set) else set()

        if toks:
            ref_cands = set()

            for tok in toks:
                ref_cands |= ref_index.get(tok, set())

            if ref_cands:
                cands = cands & ref_cands if cands else ref_cands

        if not cands:
            continue

        cands = list(cands)

        def score(j):
            r = right.loc[j]
            num_ov = len(l["NumRefs"] & r["NumRefs"])
            aln_ov = len(l["AlnumRefs"] & r["AlnumRefs"])
            nm_ov = len(l["NameRefs"] & r["NameRefs"])
            nm_sim = name_similarity(l["NameRefs"], r["NameRefs"])
            amt_d = abs(float(l["Amt"]) - float(r["Amt"]))
            d1, d2 = l["Date"], r["Date"]
            d_d = abs((d1 - d2).days) if (pd.notna(d1) and pd.notna(d2)) else 10**9

            return (-num_ov, -(aln_ov + nm_ov * 0.5), -int(nm_sim * 1000), amt_d, d_d)

        cands = sorted(cands, key=score)

        for j in cands:
            if i in usedL or j in usedR:
                continue

            r = right.loc[j]

            num_ov = len(l["NumRefs"] & r["NumRefs"])
            tok_ov = len(l["AlnumRefs"] & r["AlnumRefs"])
            nm_sim = name_similarity(l["NameRefs"], r["NameRefs"])

            if not (num_ov >= 1 or tok_ov >= 1 or nm_sim >= name_thresh):
                continue

            amt_diff = abs(float(l["Amt"]) - float(r["Amt"]))

            if amt_diff <= tol:
                matched.append((l, r, round(amt_diff, ROUND_DP)))
                usedL.add(i)
                usedR.add(j)
                break

    match_df = pd.DataFrame([{
        f"{labelL} Date": a["Date"],
        f"{labelL} Voucher": a["Voucher"],
        f"{labelL} Description": a["Description"],
        f"{labelL} Amount": a["Amt"],
        f"{labelR} Date": b["Date"],
        f"{labelR} Voucher": b["Voucher"],
        f"{labelR} Description": b["Description"],
        f"{labelR} Amount": b["Amt"],
        "Amount_Diff": diff
    } for (a, b, diff) in matched])

    return match_df, usedL, usedR

def _build_ref_index(df):
    idx = {}

    for j, r in df.iterrows():
        toks = r["AllRefs"]

        if not isinstance(toks, set):
            toks = set() if pd.isna(toks) else set(toks)

        for tok in toks:
            idx.setdefault(tok, set()).add(j)

    return idx

def find_partial_by_refs(
    left,
    right,
    labelL,
    labelR,
    tol,
    name_thresh,
    exclude_left: set = None,
    exclude_right: set = None
):
    exclude_left = exclude_left or set()
    exclude_right = exclude_right or set()

    ref_index = _build_ref_index(right)

    partial = []
    left_used = set()
    right_used = set()

    for i, l in left.iterrows():
        if i in exclude_left:
            continue

        toks = l["AllRefs"] if isinstance(l["AllRefs"], set) else set()

        if not toks:
            continue

        cand = set()

        for tok in toks:
            cand |= ref_index.get(tok, set())

        cand = [
            j for j in cand
            if j not in exclude_right and j not in right_used
        ]

        if not cand:
            continue

        def score(j):
            r = right.loc[j]
            num_ov = len(l["NumRefs"] & r["NumRefs"])
            aln_ov = len(l["AlnumRefs"] & r["AlnumRefs"])
            nm_ov = len(l["NameRefs"] & r["NameRefs"])
            nm_sim = name_similarity(l["NameRefs"], r["NameRefs"])
            amt_d = abs(float(l["Amt"]) - float(r["Amt"]))
            d1, d2 = l["Date"], r["Date"]
            d_d = abs((d1 - d2).days) if (pd.notna(d1) and pd.notna(d2)) else 10**9

            return (-num_ov, -(aln_ov + nm_ov * 0.5), -int(nm_sim * 1000), amt_d, d_d)

        cand = sorted(cand, key=score)

        for j in cand:
            r = right.loc[j]

            num_ov = len(l["NumRefs"] & r["NumRefs"])
            tok_ov = len(l["AlnumRefs"] & r["AlnumRefs"])
            nm_sim = name_similarity(l["NameRefs"], r["NameRefs"])

            if not (num_ov >= 1 or tok_ov >= 1 or nm_sim >= name_thresh):
                continue

            amt_diff = abs(float(l["Amt"]) - float(r["Amt"]))

            if amt_diff > tol:
                partial.append((i, j, l, r, round(amt_diff, ROUND_DP)))
                left_used.add(i)
                right_used.add(j)
                break

    partial_df = pd.DataFrame([{
        f"{labelL} Date": a["Date"],
        f"{labelL} Voucher": a["Voucher"],
        f"{labelL} Description": a["Description"],
        f"{labelL} Amount": a["Amt"],
        f"{labelR} Date": b["Date"],
        f"{labelR} Voucher": b["Voucher"],
        f"{labelR} Description": b["Description"],
        f"{labelR} Amount": b["Amt"],
        "Amount_Diff": diff
    } for (_i, _j, a, b, diff) in partial])

    return partial_df, left_used, right_used

# ================= CORE RECON =================
def run_recon_core(
    xls_bytes,
    our_sheet_name,
    branch_sheet_name,
    amount_tol,
    name_sim_thresh,
    use_fast=False
):
    xls = pd.ExcelFile(xls_bytes)
    names = xls.sheet_names

    our_name = (
        _find_by_alias(our_sheet_name, names)
        or next((s for s in names if _eq(s, our_sheet_name)), None)
    )

    if not our_name:
        raise ValueError(f"Our book sheet '{our_sheet_name}' not found. Found: {names}")

    if branch_sheet_name:
        branch_name = (
            _find_by_alias(branch_sheet_name, names)
            or next((s for s in names if _eq(s, branch_sheet_name)), None)
        )

        if not branch_name:
            raise ValueError(
                f"Branch book sheet '{branch_sheet_name}' not found. Found: {names}"
            )
    else:
        branch_name = _auto_pick_other(our_name, names)

        if not branch_name:
            raise ValueError("Could not determine Branch sheet. Please type it explicitly.")

    OUR = prep_sheet(pd.read_excel(xls_bytes, sheet_name=our_name))
    BR = prep_sheet(pd.read_excel(xls_bytes, sheet_name=branch_name))

    OUR_DR, OUR_CR = split_sides(OUR, "OUR")
    BR_DR, BR_CR = split_sides(BR, "BR")

    label_our_dr = "Our book DR"
    label_our_cr = "Our book CR"
    label_br_dr = "Branch book DR"
    label_br_cr = "Branch book CR"

    matcher = pair_exact_best_fast_same if use_fast else pair_exact_best

    m1, usedL1, usedR1 = matcher(
        OUR_DR,
        BR_CR,
        label_our_dr,
        label_br_cr,
        amount_tol,
        name_sim_thresh
    )

    m2, usedL2, usedR2 = matcher(
        OUR_CR,
        BR_DR,
        label_our_cr,
        label_br_dr,
        amount_tol,
        name_sim_thresh
    )

    matching_df = (
        pd.concat([m1, m2], ignore_index=True)
        if (not m1.empty or not m2.empty)
        else pd.DataFrame()
    )

    p1, pL1, pR1 = find_partial_by_refs(
        OUR_DR,
        BR_CR,
        label_our_dr,
        label_br_cr,
        amount_tol,
        name_sim_thresh,
        exclude_left=usedL1,
        exclude_right=usedR1
    )

    p2, pL2, pR2 = find_partial_by_refs(
        OUR_CR,
        BR_DR,
        label_our_cr,
        label_br_dr,
        amount_tol,
        name_sim_thresh,
        exclude_left=usedL2,
        exclude_right=usedR2
    )

    partial_df = (
        pd.concat([p1, p2], ignore_index=True)
        if (not p1.empty or not p2.empty)
        else pd.DataFrame()
    )

    usedL_all = usedL1 | usedL2 | pL1 | pL2
    usedR_all = usedR1 | usedR2 | pR1 | pR2

    un_our = pd.concat([
        OUR_DR.loc[[i for i in OUR_DR.index if i not in usedL_all]],
        OUR_CR.loc[[i for i in OUR_CR.index if i not in usedL_all]],
    ], ignore_index=True)

    un_br = pd.concat([
        BR_DR.loc[[i for i in BR_DR.index if i not in usedR_all]],
        BR_CR.loc[[i for i in BR_CR.index if i not in usedR_all]],
    ], ignore_index=True)

    un_our["Which"] = "Our book"
    un_br["Which"] = "Branch book"

    unmatching_df = pd.concat([
        un_our[["Which", "Date", "Voucher", "Description", "Debit", "Credit", "Amt"]],
        un_br[["Which", "Date", "Voucher", "Description", "Debit", "Credit", "Amt"]],
    ], ignore_index=True)

    matching_df = strip_time_cols(matching_df)
    unmatching_df = strip_time_cols(unmatching_df)
    partial_df = strip_time_cols(partial_df)

    out = io.BytesIO()

    with pd.ExcelWriter(out, engine="xlsxwriter") as w:
        workbook = w.book

        header_format = workbook.add_format({
            "bold": True,
            "bg_color": "#1E3A8A",
            "font_color": "white",
            "border": 1
        })

        money_format = workbook.add_format({
            "num_format": "#,##0.00"
        })

        date_format = workbook.add_format({
            "num_format": "dd-mm-yyyy"
        })

        sheets = {
            "Matching": matching_df if not matching_df.empty else pd.DataFrame({"Info": ["No matches found"]}),
            "Unmatching": unmatching_df if not unmatching_df.empty else pd.DataFrame({"Info": ["No unmatching transactions"]}),
            "PartialMatching": partial_df if not partial_df.empty else pd.DataFrame({"Info": ["No partial ref mismatches"]})
        }

        for sheet_name, data in sheets.items():
            data.to_excel(w, sheet_name=sheet_name, index=False)
            ws = w.sheets[sheet_name]

            for col_num, value in enumerate(data.columns.values):
                ws.write(0, col_num, value, header_format)
                ws.set_column(col_num, col_num, 18)

            for idx, col in enumerate(data.columns):
                if "amount" in col.lower() or "debit" in col.lower() or "credit" in col.lower() or "diff" in col.lower() or col == "Amt":
                    ws.set_column(idx, idx, 16, money_format)
                elif "date" in col.lower():
                    ws.set_column(idx, idx, 14, date_format)
                elif "description" in col.lower():
                    ws.set_column(idx, idx, 45)

    out.seek(0)

    return matching_df, unmatching_df, partial_df, out

@st.cache_data(show_spinner=False)
def run_recon_cached_v2(
    file_bytes: bytes,
    our_sheet_name,
    branch_sheet_name,
    amount_tol,
    name_sim_thresh,
    use_fast,
    _version: str = "modern-v3"
):
    buf = io.BytesIO(file_bytes)
    return run_recon_core(
        buf,
        our_sheet_name,
        branch_sheet_name,
        amount_tol,
        name_sim_thresh,
        use_fast
    )

# ================= URL HELPERS =================
def drive_id_from_url(url: str):
    m = re.search(r"/(?:file|spreadsheets)/d/([A-Za-z0-9_-]+)", url)

    if m:
        return m.group(1)

    m = re.search(r"[?&]id=([A-Za-z0-9_-]+)", url)

    return m.group(1) if m else None

def onedrive_direct(url: str):
    if "1drv.ms" in url or "sharepoint.com" in url:
        if "download=1" not in url:
            url += ("&" if "?" in url else "?") + "download=1"

    return url

def download_to_temp(url: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    tmp.close()

    path = tmp.name

    fid = drive_id_from_url(url)

    if fid:
        if "docs.google.com/spreadsheets" in url:
            export_url = f"https://docs.google.com/spreadsheets/d/{fid}/export?format=xlsx"
        else:
            export_url = f"https://drive.google.com/uc?export=download&id={fid}"

        with requests.get(export_url, stream=True, timeout=180) as r:
            r.raise_for_status()

            with open(path, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)

        return path

    url2 = onedrive_direct(url)

    with requests.get(url2, stream=True, timeout=180) as r:
        r.raise_for_status()

        with open(path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

    return path

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("Branch Reconciliation Control Panel")

    st.divider()

    mode = st.radio(
        "Match Mode",
        ["Exact Original", "Fast Same Result"],
        index=0
    )

    use_fast = mode.startswith("Fast")

    st.markdown("### 📄 Sheet Names")

    our_sheet = st.text_input(
        "Our Book Sheet",
        value="Our book"
    )

    branch_sheet = st.text_input(
        "Branch Book Sheet",
        value="Branch book"
    )

    st.markdown("### 🎯 Matching Rules")

    st.info("Amount tolerance fixed at ± 5 SAR")

    name_thresh = st.slider(
        "Name Similarity Threshold",
        0.0,
        1.0,
        float(NAME_SIM_THRESHOLD_DEFAULT),
        0.05
    )

    st.divider()

    if st.button("♻️ Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared successfully.")

# ================= INPUT AREA =================
st.markdown("""
<div class="section-card status-info">
    <h3 style="margin-top:0;">📂 Upload or Connect Source File</h3>
    <p class="small-muted">
        Upload Excel file or paste Drive / Google Sheets / OneDrive / SharePoint link.
    </p>
</div>
""", unsafe_allow_html=True)

source = st.radio(
    "Choose input method",
    ["Upload File", "From URL"],
    horizontal=True
)

uploaded = None
file_url = None

if source == "Upload File":
    uploaded = st.file_uploader(
        "Upload Excel file",
        type=["xlsx", "xls"]
    )
else:
    file_url = st.text_input(
        "Paste share link here"
    )

run_btn = st.button(
    "🚀 Run Reconciliation",
    type="primary",
    disabled=(uploaded is None and not file_url)
)

# ================= KPI DISPLAY FUNCTION =================
def kpi_card(title, value, note, status_class="status-info"):
    st.markdown(f"""
    <div class="kpi-card {status_class}">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)

def safe_len(df):
    return 0 if df is None or df.empty else len(df)

# ================= RUN APP =================
if run_btn:
    try:
        with st.spinner("Processing reconciliation..."):
            if file_url:
                local_path = download_to_temp(file_url)

                with open(local_path, "rb") as f:
                    file_bytes = f.read()

                try:
                    os.unlink(local_path)
                except Exception:
                    pass
            else:
                file_bytes = uploaded.read()

            matching_df, unmatching_df, partial_df, out_xlsx = run_recon_cached_v2(
                file_bytes,
                our_sheet,
                branch_sheet,
                AMOUNT_TOLERANCE_DEFAULT,
                name_thresh,
                use_fast,
                _version="modern-v3"
            )

        st.success("Reconciliation completed successfully.")

        matched_count = safe_len(matching_df)
        unmatched_count = safe_len(unmatching_df)
        partial_count = safe_len(partial_df)
        total_count = matched_count + unmatched_count + partial_count

        match_rate = (
            round((matched_count / total_count) * 100, 2)
            if total_count > 0
            else 0
        )

        st.markdown("## 📊 Reconciliation Summary")

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            kpi_card("Total Items", f"{total_count:,}", "Total processed", "status-info")

        with c2:
            kpi_card("Matched", f"{matched_count:,}", "Exact matches", "status-success")

        with c3:
            kpi_card("Partial", f"{partial_count:,}", "Same reference, amount difference", "status-warning")

        with c4:
            kpi_card("Unmatched", f"{unmatched_count:,}", "Needs review", "status-danger")

        with c5:
            kpi_card("Match Rate", f"{match_rate}%", "Overall success", "status-success")

        st.markdown("---")

        tab_summary, tab_match, tab_partial, tab_unmatch, tab_export = st.tabs([
            "📊 Summary",
            "✅ Matching",
            "🟡 Partial",
            "❌ Unmatching",
            "⬇ Export"
        ])

        with tab_summary:
            st.markdown("### 📈 Status Overview")

            chart_df = pd.DataFrame({
                "Status": ["Matched", "Partial", "Unmatched"],
                "Count": [matched_count, partial_count, unmatched_count]
            })

            if chart_df["Count"].sum() > 0:
                fig = px.pie(
                    chart_df,
                    names="Status",
                    values="Count",
                    hole=0.45,
                    title="Reconciliation Breakdown"
                )
                fig.update_layout(
                    height=420,
                    showlegend=True,
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

                bar = px.bar(
                    chart_df,
                    x="Status",
                    y="Count",
                    text="Count",
                    title="Transaction Count by Status"
                )
                bar.update_layout(
                    height=380,
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                st.plotly_chart(bar, use_container_width=True)
            else:
                st.info("No data available for chart.")

        with tab_match:
            st.markdown("### ✅ Exact Matching Transactions")

            st.dataframe(
                matching_df if not matching_df.empty else pd.DataFrame({"Info": ["No matches found"]}),
                use_container_width=True,
                height=520
            )

        with tab_partial:
            st.markdown("### 🟡 Partial Reference Matches")

            st.warning("These items have same or similar references, but amount difference is more than ±5 SAR.")

            st.dataframe(
                partial_df if not partial_df.empty else pd.DataFrame({"Info": ["No partial ref mismatches"]}),
                use_container_width=True,
                height=520
            )

        with tab_unmatch:
            st.markdown("### ❌ Unmatched Transactions")

            st.dataframe(
                unmatching_df if not unmatching_df.empty else pd.DataFrame({"Info": ["No unmatching transactions"]}),
                use_container_width=True,
                height=520
            )

        with tab_export:
            st.markdown("""
            <div class="section-card status-success">
                <h3 style="margin-top:0;">⬇ Export Reconciliation Report</h3>
                <p class="small-muted">
                    Download complete Excel output with Matching, Unmatching and Partial Matching sheets.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.download_button(
                "⬇ Download Complete Excel Report",
                data=out_xlsx.getvalue(),
                file_name=f"Branch_Recon_Output_{'FAST' if use_fast else 'ORIG'}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(str(e))

# ================= FOOTER =================
st.markdown("""
<div class="footer">
    <strong>Branch Reconciliation v3.0</strong><br>
    Created by: Jaseer Pykarathodi — Treasury Officer<br>
    Issam Kabbani & Partners Unitech
</div>
""", unsafe_allow_html=True)
