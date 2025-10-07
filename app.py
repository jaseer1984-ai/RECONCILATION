# app.py — Branch Recon (Our book wording + URL or file upload) — Tolerance fixed at ±5 SAR
import io, re, os, tempfile, requests
import pandas as pd, numpy as np
from datetime import datetime
from difflib import SequenceMatcher
import streamlit as st

# ===== CONFIG =====
ROUND_DP = 2
AMOUNT_TOLERANCE_DEFAULT = 5.0  # fixed ±5 SAR
NAME_SIM_THRESHOLD_DEFAULT = 0.80

# -------------------- PAGE / BRANDING --------------------
st.set_page_config(page_title="Branch Reconciliation", layout="wide")
st.markdown("""
<div style="padding:16px 0;">
  <h1 style="margin:0;">Issam Kabbani and Partners Unitech</h1>
  <hr/>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("### 🏢 Issam Kabbani and Partners Unitech")
# st.sidebar.markdown("**Created by:** Jaseer Pykarathodi  \n**Dept:** Treasury Officer")

def _footer():
    st.markdown("""
    ---
    **Created by:** Jaseer Pykarathodi — Treasury Officer  |  **Company:** Issam Kabbani and Partners Unitech
    """)

# ---------- Robust date parsing ----------
_MONTHS = {'jan':1,'january':1,'feb':2,'february':2,'mar':3,'march':3,'apr':4,'april':4,'may':5,
           'jun':6,'june':6,'jul':7,'july':7,'aug':8,'august':8,'sep':9,'sept':9,'september':9,
           'oct':10,'october':10,'nov':11,'november':11,'dec':12,'december':12}
date_patterns = [
    r'(?P<d>\d{1,2})[^\w\s]?(?P<m>\d{1,2})[^\w\s]?(?P<y>\d{2,4})',
    r'(?P<y>\d{4})[^\w\s]?(?P<m>\d{1,2})[^\w\s]?(?P<d>\d{1,2})',
    r'(?P<d>\d{1,2})\s+(?P<mon>[A-Za-z]{3,9})\.?,?\s+(?P<y>\d{2,4})',
]
def parse_any_date(val):
    if pd.isna(val): return pd.NaT
    s = str(val).strip()
    try:
        return pd.to_datetime(s, errors="raise", dayfirst=True)
    except Exception:
        low = s.lower()
        for pat in date_patterns:
            m = re.search(pat, low)
            if not m: continue
            gd = m.groupdict()
            try:
                if 'mon' in gd and gd['mon']:
                    mon = _MONTHS.get(gd['mon'][:3].lower()); d, y = int(gd['d']), int(gd['y']); y = y+2000 if y<100 else y
                    return pd.Timestamp(datetime(y, mon, d))
                else:
                    d, mth, y = int(gd['d']), int(gd['m']), int(gd['y']); y = y+2000 if y<100 else y
                    return pd.Timestamp(datetime(y, mth, d))
            except Exception:
                continue
    return pd.NaT

# ---------- Utilities ----------
STOP_TOKENS_LATIN = {
    "DATE","DOC","BANK","DEBIT","CREDIT","REF","DEP","DEPOSIT","TRANSFER","TFR","PYT",
    "RIB","HSBC","SNB","SABB","BAB","JED","JEDDAH","RIYADH","COMPANY","CO","LTD","ACTUAL",
    "TOSL","PAYMENT","EXIT","ENTRY","RENEWAL","FEE","CHARGE","BONUS","REWARD","RWD",
    "DTD","BREF","B.REF","INV","FT","TT","CHQ","CHEQUE","SADAD","EXP","ERE","PAY"
}
STOP_TOKENS_AR = {"بنك","تحويل","حوالة","ايداع","سداد","رسوم","جدة","الرياض","مبلغ","شركة","رقم","دفع","مدفوع"}

def to_numeric(x):
    return pd.to_numeric(pd.Series(x).astype(str).str.replace(",", "", regex=False), errors="coerce")

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def extract_refs_all(voucher: str, description: str):
    text = f"{voucher or ''} {description or ''}"
    nums = re.findall(r"\d{6,}", text)
    alnums = re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-/:]{3,}", text)
    for tok in alnums:
        nums += re.findall(r"\d{5,}", tok)
    latin_names  = re.findall(r"[A-Za-z]{3,}", text)
    arabic_names = re.findall(r"[\u0600-\u06FF]{2,}", text)
    num_set   = set(nums)
    alnum_set = {t.upper() for t in alnums if t.upper() not in STOP_TOKENS_LATIN}
    name_set  = {t.upper() for t in latin_names if t.upper() not in STOP_TOKENS_LATIN}
    name_set |= {t for t in arabic_names if t not in STOP_TOKENS_AR}
    return num_set, alnum_set, name_set

def name_similarity(a_names: set, b_names: set) -> float:
    if not a_names and not b_names: return 0.0
    jacc = (len(a_names & b_names) / len(a_names | b_names)) if (a_names or b_names) else 0.0
    a_str = normalize_spaces(" ".join(sorted(a_names)))
    b_str = normalize_spaces(" ".join(sorted(b_names)))
    seq  = SequenceMatcher(None, a_str, b_str).ratio() if a_str and b_str else 0.0
    return max(jacc, seq)

def prep_sheet(df):
    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*names, required=False):
        for n in names:
            if n.lower() in cols: return cols[n.lower()]
        if required: raise ValueError(f"Missing required column among {names}. Found: {list(df.columns)}")
        return None
    cDate = pick("date")
    cVno  = pick("voucher no.", "voucher no", "voucher", "voucher_no")
    cDesc = pick("description", required=True)
    cDr   = pick("debit","dr")
    cCr   = pick("credit","cr")

    out = pd.DataFrame({
        "Date": df[cDate].apply(parse_any_date) if cDate else pd.NaT,
        "Voucher": df[cVno].astype(str).str.strip() if cVno else "",
        "Description": df[cDesc].astype(str).fillna(""),
        "Debit":  to_numeric(df[cDr]) if cDr else 0.0,
        "Credit": to_numeric(df[cCr]) if cCr else 0.0,
    })
    out["Amt"] = (out["Debit"].fillna(0) + out["Credit"].fillna(0)).round(ROUND_DP)

    refs = out.apply(lambda r: extract_refs_all(r["Voucher"], r["Description"]), axis=1)
    out["NumRefs"]   = refs.map(lambda t: t[0])
    out["AlnumRefs"] = refs.map(lambda t: t[1])
    out["NameRefs"]  = refs.map(lambda t: t[2])
    out["AllRefs"]   = out.apply(lambda r: set().union(r["NumRefs"], r["AlnumRefs"], r["NameRefs"]), axis=1)
    return out

def split_sides(df, tag):
    dr = df[df["Debit"]  > 0].copy(); dr["Side"]=f"{tag}-DR"; dr["Amt"]=dr["Debit"].round(ROUND_DP)
    cr = df[df["Credit"] > 0].copy(); cr["Side"]=f"{tag}-CR"; cr["Amt"]=cr["Credit"].round(ROUND_DP)
    for d in (dr, cr):
        d.reset_index(drop=True, inplace=True)
        d["RowID"] = np.arange(len(d))
    return dr, cr

def strip_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    out = df.copy()
    for c in out.columns:
        if "date" in c.lower():
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.date
    return out

# -------- Matching / Engine --------
def _casefold(s: str) -> str: return s.strip().lower()
def _eq(a: str, b: str) -> bool: return _casefold(a) == _casefold(b)

OUR_ALIASES     = {"our book", "tosl", "ourbook", "our_book"}
BR_ALIASES      = {"branch book", "branch", "br", "branchbook", "branch_book"}

def _find_by_alias(target, names):
    if not target: return None
    t = _casefold(target)
    for n in names:
        if _eq(n, target): return n
    if t in OUR_ALIASES:
        for n in names:
            if _casefold(n) in OUR_ALIASES: return n
    if t in BR_ALIASES:
        for n in names:
            if _casefold(n) in BR_ALIASES: return n
    return None

def _auto_pick_other(our_name, names):
    for n in names:
        if not _eq(n, our_name) and _casefold(n) in BR_ALIASES:
            return n
    for n in names:
        if not _eq(n, our_name):
            return n
    return None

# ---------- ORIGINAL matcher ----------
def pair_exact_best(left, right, labelL, labelR, tol, name_thresh=NAME_SIM_THRESHOLD_DEFAULT):
    # inverted index on right
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
            nm_ov  = len(l["NameRefs"] & r["NameRefs"])
            nm_sim = name_similarity(l["NameRefs"], r["NameRefs"])
            amt_d  = abs(float(l["Amt"]) - float(r["Amt"]))
            d1, d2 = l["Date"], r["Date"]
            d_d    = abs((d1 - d2).days) if (pd.notna(d1) and pd.notna(d2)) else 10**9
            return (-num_ov, -(aln_ov + nm_ov*0.5), -int(nm_sim*1000), amt_d, d_d)

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
                usedL.add(i); usedR.add(j)
                break

    match_df = pd.DataFrame([{
        f"{labelL} Date":  a["Date"],
        f"{labelL} Voucher": a["Voucher"],
        f"{labelL} Description": a["Description"],
        f"{labelL} Amount": a["Amt"],
        f"{labelR} Date":  b["Date"],
        f"{labelR} Voucher": b["Voucher"],
        f"{labelR} Description": b["Description"],
        f"{labelR} Amount": b["Amt"],
        "Amount_Diff": diff
    } for (a,b,diff) in matched])

    return match_df, usedL, usedR

# ---------- FAST (same-result) wrapper ----------
def pair_exact_best_fast_same(left, right, labelL, labelR, tol, name_thresh=NAME_SIM_THRESHOLD_DEFAULT):
    right_amt = right["Amt"].astype(float).values

    def candidates_amount(l_amt):
        diff = np.abs(right_amt - float(l_amt))
        return list(np.nonzero(diff <= tol)[0])

    usedL, usedR, matched = set(), set(), []

    # also build ref index like original (for prioritizing candidates)
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
            nm_ov  = len(l["NameRefs"] & r["NameRefs"])
            nm_sim = name_similarity(l["NameRefs"], r["NameRefs"])
            amt_d  = abs(float(l["Amt"]) - float(r["Amt"]))
            d1, d2 = l["Date"], r["Date"]
            d_d    = abs((d1 - d2).days) if (pd.notna(d1) and pd.notna(d2)) else 10**9
            return (-num_ov, -(aln_ov + nm_ov*0.5), -int(nm_sim*1000), amt_d, d_d)

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
                usedL.add(i); usedR.add(j)
                break

    match_df = pd.DataFrame([{
        f"{labelL} Date":  a["Date"],
        f"{labelL} Voucher": a["Voucher"],
        f"{labelL} Description": a["Description"],
        f"{labelL} Amount": a["Amt"],
        f"{labelR} Date":  b["Date"],
        f"{labelR} Voucher": b["Voucher"],
        f"{labelR} Description": b["Description"],
        f"{labelR} Amount": b["Amt"],
        "Amount_Diff": diff
    } for (a,b,diff) in matched])

    return match_df, usedL, usedR

# ---------- NEW: Partial matches by refs, amount > tol ----------
def _build_ref_index(df):
    idx = {}
    for j, r in df.iterrows():
        toks = r["AllRefs"]
        if not isinstance(toks, set):
            toks = set() if pd.isna(toks) else set(toks)
        for tok in toks:
            idx.setdefault(tok, set()).add(j)
    return idx

def find_partial_by_refs(left, right, labelL, labelR, tol, name_thresh,
                         exclude_left: set = None, exclude_right: set = None):
    """
    Find best candidate by reference overlap/name similarity where amount diff > tol.
    Excludes indices already used in exact matches.
    """
    exclude_left = exclude_left or set()
    exclude_right = exclude_right or set()

    ref_index = _build_ref_index(right)
    partial = []
    for i, l in left.iterrows():
        if i in exclude_left:
            continue
        toks = l["AllRefs"] if isinstance(l["AllRefs"], set) else set()
        if not toks:
            continue
        cand = set()
        for tok in toks:
            cand |= ref_index.get(tok, set())
        cand = [j for j in cand if j not in exclude_right]
        if not cand:
            continue

        def score(j):
            r = right.loc[j]
            num_ov = len(l["NumRefs"] & r["NumRefs"])
            aln_ov = len(l["AlnumRefs"] & r["AlnumRefs"])
            nm_ov  = len(l["NameRefs"] & r["NameRefs"])
            nm_sim = name_similarity(l["NameRefs"], r["NameRefs"])
            amt_d  = abs(float(l["Amt"]) - float(r["Amt"]))
            d1, d2 = l["Date"], r["Date"]
            d_d    = abs((d1 - d2).days) if (pd.notna(d1) and pd.notna(d2)) else 10**9
            # Prefer stronger ref overlaps, then closer amounts/dates (for readability)
            return (-num_ov, -(aln_ov + nm_ov*0.5), -int(nm_sim*1000), amt_d, d_d)

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
                partial.append((l, r, round(amt_diff, ROUND_DP)))
                break  # only best one per left

    partial_df = pd.DataFrame([{
        f"{labelL} Date":  a["Date"],
        f"{labelL} Voucher": a["Voucher"],
        f"{labelL} Description": a["Description"],
        f"{labelL} Amount": a["Amt"],
        f"{labelR} Date":  b["Date"],
        f"{labelR} Voucher": b["Voucher"],
        f"{labelR} Description": b["Description"],
        f"{labelR} Amount": b["Amt"],
        "Amount_Diff": diff
    } for (a,b,diff) in partial])

    return partial_df

# ---------------------- CORE RUN ----------------------
def run_recon_core(xls_bytes, our_sheet_name, branch_sheet_name, amount_tol, name_sim_thresh, use_fast=False):
    xls = pd.ExcelFile(xls_bytes)
    names = xls.sheet_names

    our_name = _find_by_alias(our_sheet_name, names) \
               or next((s for s in names if _eq(s, our_sheet_name)), None)
    if not our_name:
        raise ValueError(f"Our book sheet '{our_sheet_name}' not found. Found: {names}")

    if branch_sheet_name:
        branch_name = _find_by_alias(branch_sheet_name, names) \
                      or next((s for s in names if _eq(s, branch_sheet_name)), None)
        if not branch_name:
            raise ValueError(f"Branch book sheet '{branch_sheet_name}' not found. Found: {names}")
    else:
        branch_name = _auto_pick_other(our_name, names)
        if not branch_name:
            raise ValueError("Could not determine Branch sheet; please type it explicitly.")

    OUR   = prep_sheet(pd.read_excel(xls_bytes, sheet_name=our_name))
    BR    = prep_sheet(pd.read_excel(xls_bytes, sheet_name=branch_name))

    OUR_DR, OUR_CR = split_sides(OUR, "OUR")
    BR_DR,  BR_CR  = split_sides(BR,  "BR")

    # labels for output columns
    label_our_dr = "Our book DR"
    label_our_cr = "Our book CR"
    label_br_dr  = "Branch book DR"
    label_br_cr  = "Branch book CR"

    matcher = pair_exact_best_fast_same if use_fast else pair_exact_best

    # Exact matches within ± tol
    m1, usedL1, usedR1 = matcher(OUR_DR, BR_CR, label_our_dr, label_br_cr, amount_tol, name_sim_thresh)
    m2, usedL2, usedR2 = matcher(OUR_CR, BR_DR, label_our_cr, label_br_dr, amount_tol, name_sim_thresh)
    matching_df = pd.concat([m1, m2], ignore_index=True) if (not m1.empty or not m2.empty) else pd.DataFrame()

    # Partial matches (same refs / name similarity) but amount difference > tol
    p1 = find_partial_by_refs(
        OUR_DR, BR_CR, label_our_dr, label_br_cr, amount_tol, name_sim_thresh,
        exclude_left=usedL1, exclude_right=usedR1
    )
    p2 = find_partial_by_refs(
        OUR_CR, BR_DR, label_our_cr, label_br_dr, amount_tol, name_sim_thresh,
        exclude_left=usedL2, exclude_right=usedR2
    )
    partial_df = pd.concat([p1, p2], ignore_index=True) if (not p1.empty or not p2.empty) else pd.DataFrame()

    # Unmatching (anything not used in exact matches; partials are informative only)
    un_our = pd.concat([
        OUR_DR.loc[[i for i in OUR_DR.index if i not in usedL1]],
        OUR_CR.loc[[i for i in OUR_CR.index if i not in usedL2]],
    ], ignore_index=True)
    un_br = pd.concat([
        BR_DR.loc[[i for i in BR_DR.index if i not in usedR2]],
        BR_CR.loc[[i for i in BR_CR.index if i not in usedR1]],
    ], ignore_index=True)

    un_our["Which"] = "Our book"; un_br["Which"] = "Branch book"
    unmatching_df = pd.concat([
        un_our[["Which","Date","Voucher","Description","Debit","Credit","Amt"]],
        un_br[["Which","Date","Voucher","Description","Debit","Credit","Amt"]],
    ], ignore_index=True)

    # Clean date types for display/export
    matching_df   = strip_time_cols(matching_df)
    unmatching_df = strip_time_cols(unmatching_df)
    partial_df    = strip_time_cols(partial_df)

    # Build Excel output with three sheets
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as w:
        (matching_df if not matching_df.empty else pd.DataFrame({"Info":["No matches found"]})) \
            .to_excel(w, sheet_name="Matching", index=False)
        (unmatching_df if not unmatching_df.empty else pd.DataFrame({"Info":["No unmatching transactions"]})) \
            .to_excel(w, sheet_name="Unmatching", index=False)
        (partial_df if not partial_df.empty else pd.DataFrame({"Info":["No partial (ref) mismatches"]})) \
            .to_excel(w, sheet_name="PartialMatching", index=False)
    out.seek(0)
    return matching_df, unmatching_df, partial_df, out

# --------- CACHED wrapper (returns 4 values) ---------
@st.cache_data(show_spinner=False)
def run_recon_cached_v2(file_bytes: bytes, our_sheet_name, branch_sheet_name, amount_tol, name_sim_thresh, use_fast, _version: str = "v2"):
    buf = io.BytesIO(file_bytes)
    # returns: matching_df, unmatching_df, partial_df, out
    return run_recon_core(buf, our_sheet_name, branch_sheet_name, amount_tol, name_sim_thresh, use_fast)

# ---------- URL download helpers ----------
def drive_id_from_url(url: str):
    m = re.search(r"/(?:file|spreadsheets)/d/([A-Za-z0-9_-]+)", url)
    if m: return m.group(1)
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
                    if chunk: f.write(chunk)
        return path

    url2 = onedrive_direct(url)
    with requests.get(url2, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk: f.write(chunk)
    return path

# ---------------------- UI ----------------------
st.title("🔗 BRANCH RECON — Best-overlap Matcher")

with st.sidebar:
    # Mode selector (Fast = same result, just faster)
    mode = st.radio("Match mode", ["Exact (original)", "Fast (same result)"], index=0, horizontal=False)
    use_fast = mode.startswith("Fast")

    our_sheet     = st.text_input("Our book sheet name", value="Our book")
    branch_sheet  = st.text_input("Branch book sheet name (blank = auto)", value="Branch book")
    st.info("Amount tolerance fixed at ± 5 SAR")  # fixed; no number_input for tolerance
    name_thresh   = st.slider("Name-only similarity threshold", 0.0, 1.0, float(NAME_SIM_THRESHOLD_DEFAULT), 0.05)

    st.divider()
    if st.button("♻️ Clear cache"):
        st.cache_data.clear()
        st.success("Cache cleared.")

source = st.radio("Choose input method", ["Upload file", "From URL (Drive/Sheets/OneDrive/SharePoint)"], horizontal=True)

uploaded = None
file_url = None
if source == "Upload file":
    uploaded = st.file_uploader("Upload Excel (must contain the two sheets above)", type=["xlsx","xls"])
else:
    file_url = st.text_input("Paste share link here")

run_btn = st.button("Run Reconciliation", type="primary", disabled=(uploaded is None and not file_url))

if run_btn:
    try:
        with st.spinner("Processing…"):
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

            # Use v2 cache wrapper; backward-compat fallback if old cache returns 3 values
            res = run_recon_cached_v2(
                file_bytes, our_sheet, branch_sheet, AMOUNT_TOLERANCE_DEFAULT, name_thresh, use_fast, _version="v2"
            )
            if isinstance(res, tuple) and len(res) == 4:
                matching_df, unmatching_df, partial_df, out_xlsx = res
            else:
                # legacy (3-value) cache fallback
                matching_df, unmatching_df, out_xlsx = res
                partial_df = pd.DataFrame({"Info": ["No partial (ref) mismatches"]})

        st.success("Done!")
        st.subheader("✅ Matching")
        st.dataframe(matching_df if not matching_df.empty else pd.DataFrame({"Info":["No matches found"]}), use_container_width=True)

        st.subheader("❗ Unmatching")
        st.dataframe(unmatching_df if not unmatching_df.empty else pd.DataFrame({"Info":["No unmatching transactions"]}), use_container_width=True)

        st.subheader("🟡 Partial (same refs, amount > 5 SAR)")
        with st.expander("Show partial reference matches", expanded=False):
            st.dataframe(partial_df if not partial_df.empty else pd.DataFrame({"Info":["No partial (ref) mismatches"]}), use_container_width=True)

        st.download_button(
            "⬇️ Download Excel result",
            data=out_xlsx.getvalue(),
            file_name=f"Branch_Recon_Output_{'FAST' if use_fast else 'ORIG'}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(str(e))

_footer()
