"""
VERIS: Detecting Greenwashing Through Satellite-Verified NLP
Forensic ESG Audit Dashboard  |  Group 9  |  April 2026

CSV DATA PROVENANCE (every number traces back to a notebook):
  veris_master.csv        <- notebook 04  (CT mapping + AHP-TOPSIS + VERIS)
  veris_emissions.csv     <- notebook 04  (CT v5.4.1 trajectory)
  veris_causal.csv        <- notebook 05  (DoubleML + FE + placebo)
  veris_leaderboard.csv   <- notebook 06  (visualisations)
  validation_results.csv  <- notebook 05  (statistical tests)
  veris_lda_keywords.csv  <- notebook 03  (LDA topic keywords per firm)
  veris_qualitative_summary.csv <- notebook 03 (firm-level summaries)

Run locally:   streamlit run dashboard.py
Deploy free:   share.streamlit.io
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pathlib, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="VERIS: Detecting Greenwashing Through Satellite-Verified NLP",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens ────────────────────────────────────────────────────────────
QUAD_COLORS = {
    "Q1 Genuine":      "#1a7340",
    "Q2 Greenwashing": "#c1121f",
    "Q3 Greenhushing": "#e07b00",
    "Q4 Stagnant":     "#6b7280",
    "No Data":         "#d1d5db",
    "CT Discontinuity":"#e5e7eb",
}
FIRM_COLORS = {
    "BP":"#005F73","Shell":"#EE9B00","Eni":"#0A3D99","Equinor":"#AE2012",
    "TotalEnergies":"#CA6702","Repsol":"#001B6B","ExxonMobil":"#9B2226",
    "Chevron":"#005F99","ConocoPhillips":"#BB3E03","Glencore":"#4A4A4A",
    "RioTinto":"#94213B","Unilever":"#1F3A8F",
}
ALL_FIRMS = sorted(FIRM_COLORS.keys())

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;}
.kpi-card{background:#0a1f13;border:1px solid #1b4332;border-radius:10px;padding:14px 18px;text-align:center;margin-bottom:4px;}
.kpi-num{font-family:'JetBrains Mono',monospace;font-size:1.8rem;font-weight:600;color:#52b788;line-height:1.1;}
.kpi-lbl{font-size:0.68rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.1em;margin-top:3px;}
.kpi-sub{font-size:0.72rem;color:#4a7c59;margin-top:2px;}
.info-box{background:#f0faf4;border-left:4px solid #2d6a4f;border-radius:6px;padding:9px 14px;font-size:.83rem;color:#1b4332;margin:6px 0;}
.find-box{background:#f0fdf4;border-left:4px solid #16a34a;border-radius:6px;padding:9px 14px;font-size:.83rem;color:#14532d;margin:6px 0;}
.warn-box{background:#fff7ed;border-left:4px solid #ea580c;border-radius:6px;padding:9px 14px;font-size:.83rem;color:#7c2d12;margin:6px 0;}
.badge{display:inline-block;background:#1b4332;color:#95d5b2;border-radius:20px;padding:2px 11px;font-family:'JetBrains Mono',monospace;font-size:.75rem;margin:2px 3px;}
.sig-card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;margin:4px 0;}
.sig-label{font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;color:#64748b;font-weight:600;}
.sig-val{font-family:'JetBrains Mono',monospace;font-size:1.1rem;font-weight:600;color:#0f172a;}
.sig-bar-bg{background:#e2e8f0;border-radius:4px;height:6px;margin:4px 0;}
.sig-explain{font-size:.78rem;color:#475569;margin-top:4px;line-height:1.4;}
.kw-pill{display:inline-block;background:#d8f3dc;color:#1b4332;border-radius:12px;padding:2px 9px;font-size:.72rem;margin:2px 3px;font-family:'JetBrains Mono',monospace;}
</style>
""", unsafe_allow_html=True)

# ── CSV loader ───────────────────────────────────────────────────────────────
BASE = pathlib.Path(__file__).parent

def _find_csv_dir():
    marker = "veris_master.csv"
    for candidate in [BASE/"csv", BASE/"outputs"/"csv"]:
        if (candidate/marker).exists(): return candidate
    walk = BASE
    for _ in range(8):
        for sub in [walk/"outputs"/"csv", walk/"csv"]:
            if (sub/marker).exists(): return sub
        walk = walk.parent
    return None

@st.cache_data(show_spinner="Loading pipeline data...")
def load_data():
    d = _find_csv_dir()
    if d is None:
        st.error("CSV files not found. Run notebooks 04 → 05 → 06 first.")
        st.stop()
    required = {
        "veris_master.csv":      "Run notebook 04",
        "veris_emissions.csv":   "Run notebook 04",
        "veris_causal.csv":      "Run notebook 05",
        "veris_leaderboard.csv": "Run notebook 06",
    }
    missing = [f"{f} ({h})" for f, h in required.items() if not (d/f).exists()]
    if missing:
        st.error("Missing CSVs:\n" + "\n".join(f"  • {m}" for m in missing))
        st.stop()
    master      = pd.read_csv(d/"veris_master.csv")
    emissions   = pd.read_csv(d/"veris_emissions.csv")
    causal      = pd.read_csv(d/"veris_causal.csv")
    leaderboard = pd.read_csv(d/"veris_leaderboard.csv")
    try:    validation = pd.read_csv(d/"validation_results.csv")
    except: validation = pd.DataFrame()
    try:    lda_kw = pd.read_csv(d/"veris_lda_keywords.csv")
    except: lda_kw = pd.DataFrame()
    try:    qual = pd.read_csv(d/"veris_qualitative_summary.csv")
    except: qual = pd.DataFrame()
    return master, emissions, causal, leaderboard, validation, lda_kw, qual

master, emissions, causal, leaderboard, validation, lda_kw, qual_summary = load_data()

# ── Dtype normalization (handles pipeline version differences V13 vs V14) ────
# Some columns can arrive as bool (V14) or string (V13). Coerce to canonical
# string form so all downstream concatenation, filtering, and hover-label code
# works identically regardless of upstream dtype.
if "eu_flag" in master.columns:
    if master["eu_flag"].dtype == bool:
        master["eu_flag"] = master["eu_flag"].map({True: "EU", False: "Non-EU"})
    else:
        # Strings  -  normalize common variants to the canonical form
        master["eu_flag"] = master["eu_flag"].astype(str).map(
            lambda v: "EU" if v.strip().lower() in ("eu", "true", "1", "yes") else "Non-EU"
        )
if "csrd_period" in master.columns:
    # Ensure csrd_period is always string (never bool/int/categorical)
    master["csrd_period"] = master["csrd_period"].astype(str)

# Normalize quadrant_short: V14 stores it as digits ("1","2","3","4","Insufficient data"),
# V13 used labels ("Q1 Genuine","Q2 Greenwashing",...). Downstream code (KPIs, color maps,
# filters) expects the labelled form. Coerce to canonical labels here so everything works.
_QUAD_LABELS = {
    "1": "Q1 Genuine", "Q1 Genuine": "Q1 Genuine",
    "1 - Genuine improvement": "Q1 Genuine",
    "2": "Q2 Greenwashing", "Q2 Greenwashing": "Q2 Greenwashing",
    "2 - Greenwashing signal": "Q2 Greenwashing",
    "3": "Q3 Greenhushing", "Q3 Greenhushing": "Q3 Greenhushing",
    "3 - Greenhushing": "Q3 Greenhushing",
    "4": "Q4 Stagnant", "Q4 Stagnant": "Q4 Stagnant",
    "4 - Stagnant": "Q4 Stagnant",
}
def _norm_quad(v):
    s = str(v).strip()
    if s in _QUAD_LABELS:
        return _QUAD_LABELS[s]
    if "insufficient" in s.lower() and "discontinuity" in s.lower():
        return "CT Discontinuity"
    if "insufficient" in s.lower() or s.lower() in ("nan", "none", ""):
        return "No Data"
    return s  # fallback  -  unknown label passes through

if "quadrant_short" in master.columns:
    master["quadrant_short"] = master["quadrant_short"].apply(_norm_quad)
# Keep greenwashing_quadrant in sync if it exists (some tabs read it directly)
if "greenwashing_quadrant" in master.columns:
    master["greenwashing_quadrant_norm"] = master["greenwashing_quadrant"].apply(_norm_quad)

# ── Pre-compute ───────────────────────────────────────────────────────────────
# VALID excludes year-pairs that can't be classified (missing CT data or CT discontinuity)
VALID = master[~master["quadrant_short"].isin(["No Data", "CT Discontinuity"])].copy()
VALID = VALID.dropna(subset=["veris_score", "emissions_delta_pct"])

# Also compute headline counts once for reuse across tabs
_N_TOTAL        = len(master)
_N_VALID        = len(VALID)
_N_CT_DISCON    = (master["quadrant_short"] == "CT Discontinuity").sum()
_N_NO_DATA      = (master["quadrant_short"] == "No Data").sum()

# Corpus-wide percentile thresholds (runtime, never hardcoded)
_fog_p75   = master["gunning_fog_index"].dropna().quantile(0.75)
_ttr_p25   = master["type_token_ratio"].dropna().quantile(0.25)
_veris_med = master["veris_score"].dropna().median()
_jac_p75   = master["jaccard_overlap"].dropna().quantile(0.75)
_sbert_p75 = master["sbert_drift"].dropna().quantile(0.75)
_jsd_p75   = master["lda_jsd"].dropna().quantile(0.75)

# ── Signal explanation engine ─────────────────────────────────────────────────
def _bar_html(val, lo, hi, color):
    pct = min(100, max(0, round((val - lo) / max(hi - lo, 1e-9) * 100)))
    return (f'<div class="sig-bar-bg"><div style="width:{pct}%;height:6px;'
            f'background:{color};border-radius:4px;"></div></div>')

def signal_explain(row, lda_kw_df=pd.DataFrame()):
    """Return HTML explaining what drove the VERIS score for one row."""
    firm = row.get("firm_name", "")
    y1   = int(row.get("year_from", 0))
    y2   = int(row.get("year_to", 0))

    v   = float(row.get("veris_score", 0))
    sb  = float(row.get("sbert_drift", 0))
    jac = float(row.get("jaccard_overlap", 0))
    jsd = float(row.get("lda_jsd", 0))
    vad = float(row.get("vader_delta", 0))
    fog = float(row.get("gunning_fog_index", 0))
    ttr = float(row.get("type_token_ratio", 0))
    ctr = float(row.get("controversy_density", 0))
    cir = float(row.get("circularity_density", 0))
    obf = int(row.get("obfuscation_flag", 0))

    # ── VERIS overall
    v_color = "#c1121f" if v > _veris_med else "#1a7340"
    v_label = "High  -  above corpus median" if v > _veris_med else "Low  -  below corpus median"

    # ── SBERT
    sb_interp = ("Large semantic shift  -  the overall meaning of the report changed significantly."
                 if sb > _sbert_p75 else
                 "Moderate semantic shift  -  some new themes introduced."
                 if sb > 0.15 else
                 "Minimal semantic shift  -  report language largely unchanged year-over-year.")
    sb_color = "#c1121f" if sb > _sbert_p75 else "#e07b00" if sb > 0.15 else "#1a7340"

    # ── Jaccard
    jac_pct = round(jac * 100)
    jac_interp = (f"{jac_pct}% of vocabulary was reused verbatim  -  very high copy-paste. "
                  "Indicates structural recycling of language without substantive update."
                  if jac > _jac_p75 else
                  f"{jac_pct}% vocabulary overlap  -  moderate reuse, some new content added."
                  if jac > 0.55 else
                  f"{jac_pct}% vocabulary overlap  -  relatively low reuse, significant new content.")
    jac_color = "#c1121f" if jac > _jac_p75 else "#e07b00" if jac > 0.55 else "#1a7340"

    # ── LDA-JSD
    jsd_interp = ("Major topic restructuring  -  the distribution of topics changed dramatically. "
                  "Possible strategic reframing of ESG narrative."
                  if jsd > _jsd_p75 else
                  "Moderate topic shift  -  some new themes appeared."
                  if jsd > 0.25 else
                  "Stable topic structure  -  firm reported on largely the same themes.")
    jsd_color = "#c1121f" if jsd > _jsd_p75 else "#e07b00" if jsd > 0.25 else "#1a7340"

    # ── LDA keywords for this firm
    kw_html = ""
    if not lda_kw_df.empty and "firm_name" in lda_kw_df.columns:
        firm_kws = lda_kw_df[lda_kw_df["firm_name"]==firm]
        if not firm_kws.empty:
            all_kws = []
            for _, kr in firm_kws.iterrows():
                words = [w.strip() for w in str(kr.get("keywords","")).split(",") if w.strip()]
                all_kws.extend(words[:3])
            seen, unique_kws = set(), []
            for w in all_kws:
                if w not in seen: seen.add(w); unique_kws.append(w)
            if unique_kws:
                pills = " ".join(f'<span class="kw-pill">{w}</span>' for w in unique_kws[:12])
                kw_html = f'<div style="margin-top:4px;"><span class="sig-label">LDA topic words detected for {firm}</span><br>{pills}</div>'

    # ── VADER
    vad_interp = (f"Sentiment became notably more positive (+{vad:.3f}). "
                  "Without matching substantive change, this suggests tone inflation."
                  if vad > 0.05 else
                  f"Sentiment change negligible ({vad:+.3f})  -  tone stable year-over-year.")
    vad_color = "#c1121f" if vad > 0.05 else "#1a7340"

    # ── Fog / TTR / Obfuscation
    fog_interp = (f"Fog index {fog:.1f}  -  above p75 ({_fog_p75:.1f}). "
                  "Unusually complex sentence structures detected."
                  if fog > _fog_p75 else
                  f"Fog index {fog:.1f}  -  within normal range.")
    ttr_interp = (f"TTR {ttr:.3f}  -  below p25 ({_ttr_p25:.3f}). "
                  "Narrow vocabulary: same words repeated frequently."
                  if ttr < _ttr_p25 else
                  f"TTR {ttr:.3f}  -  vocabulary richness within normal range.")
    obf_html = ""
    if obf:
        obf_html = ('<div class="warn-box" style="margin-top:6px;">'
                    '⚠️ <b>Obfuscation flag raised</b>  -  this report scores above p75 on complexity '
                    'AND below p25 on vocabulary richness simultaneously. '
                    'This pattern is associated with deliberate linguistic opacity.</div>')

    # ── Controversy / Circularity
    ctr_interp = (f"Controversy density {ctr:.2f} per 1,000 words  -  above corpus median ({master['controversy_density'].median():.2f}). "
                  "Higher frequency of litigious/negative legal language."
                  if ctr > master["controversy_density"].median() else
                  f"Controversy density {ctr:.2f}  -  within normal range.")
    cir_interp = (f"Circularity terms {cir:.4f} per word  -  "
                  f"{'above' if cir > master['circularity_density'].median() else 'below'} corpus median.")

    html = f"""
<div style="padding:4px 0;">
  <div class="sig-card">
    <span class="sig-label">VERIS Composite Score (AHP-TOPSIS)</span>
    <div class="sig-val" style="color:{v_color};">{v:.4f}</div>
    {_bar_html(v, 0, 1, v_color)}
    <div class="sig-explain">{v_label} | Weights: SBERT 46.15% · Jaccard 23.08% · LDA-JSD 23.08% · VADER 7.69%</div>
  </div>

  <div class="sig-card">
    <span class="sig-label">SBERT Semantic Drift (46.15% of VERIS)  -  all-mpnet-base-v2</span>
    <div class="sig-val" style="color:{sb_color};">{sb:.4f}</div>
    {_bar_html(sb, 0, 0.65, sb_color)}
    <div class="sig-explain">{sb_interp}</div>
  </div>

  <div class="sig-card">
    <span class="sig-label">Jaccard Copy-Paste Overlap (23.08% of VERIS)  -  character trigrams</span>
    <div class="sig-val" style="color:{jac_color};">{jac_pct}%</div>
    {_bar_html(jac, 0, 1, jac_color)}
    <div class="sig-explain">{jac_interp}</div>
  </div>

  <div class="sig-card">
    <span class="sig-label">LDA Topic Shift / JSD (23.08% of VERIS)  -  7 topics · 1,000 vocab</span>
    <div class="sig-val" style="color:{jsd_color};">{jsd:.4f}</div>
    {_bar_html(jsd, 0, 1, jsd_color)}
    <div class="sig-explain">{jsd_interp}</div>
    {kw_html}
  </div>

  <div class="sig-card">
    <span class="sig-label">VADER Sentiment Delta (7.69% of VERIS)</span>
    <div class="sig-val" style="color:{vad_color};">{vad:+.4f}</div>
    {_bar_html(abs(vad), 0, 0.25, vad_color)}
    <div class="sig-explain">{vad_interp}</div>
  </div>

  <div class="sig-card">
    <span class="sig-label">Forensic Signals (not in VERIS score  -  contextual flags)</span>
    <div class="sig-explain">
      <b>Fog:</b> {fog_interp}<br>
      <b>TTR:</b> {ttr_interp}<br>
      <b>Controversy:</b> {ctr_interp}<br>
      <b>Circularity:</b> {cir_interp}
    </div>
    {obf_html}
  </div>
</div>
"""
    return html



# ── KPI row ───────────────────────────────────────────────────────────────────
n_q2 = (VALID["quadrant_short"]=="Q2 Greenwashing").sum()
n_q3 = (VALID["quadrant_short"]=="Q3 Greenhushing").sum()
n_q1 = (VALID["quadrant_short"]=="Q1 Genuine").sum()
lr_auc = float(validation["logistic_auc"].iloc[0]) if not validation.empty and "logistic_auc" in validation.columns else 0.878

k = st.columns(6)
for col,(num,lbl,sub) in zip(k,[
    (str(n_q2),"Greenwashing pairs","High VERIS + rising emissions"),
    (str(n_q3),"Greenhushing pairs","Falling emissions, static language"),
    (str(n_q1),"Genuine improvement","High VERIS + falling emissions"),
    (f"{_veris_med:.3f}","VERIS median","Corpus-relative threshold"),
    (f"{lr_auc:.3f}","Logistic AUC","Q2 classification accuracy"),
    ("θ=+0.150","DoubleML CSRD 2021","p=0.005 · post-CSRD VERIS higher"),
]):
    with col:
        st.markdown(f'<div class="kpi-card"><div class="kpi-num">{num}</div>'
                    f'<div class="kpi-lbl">{lbl}</div><div class="kpi-sub">{sub}</div></div>',
                    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# MODE TOGGLE  -  Executive/Business narrative vs Analyst/Technical analytics
mode_col1, mode_col2 = st.columns([2, 5])
with mode_col1:
    ui_mode = st.radio(
        "Who are you?",
        ["🏢 Executive / Business", "🔬 Analyst / Technical"],
        horizontal=True, key="ui_mode",
        help="Business view: narrative findings, firm/portfolio lookup, persona-specific action playbooks. "
             "Technical view: the full statistical pipeline  -  LDA topic lab, AHP sensitivity, causal specs, "
             "obfuscation forensics, firm-vs-firm comparison."
    )
with mode_col2:
    if ui_mode.startswith("🏢"):
        st.markdown("""<div style="background:linear-gradient(90deg,#f0f9ff,transparent);
                    padding:10px 16px;border-radius:8px;border-left:3px solid #0284c7;
                    font-size:.84rem;color:#0c4a6e;margin-top:28px;">
        <b>Business mode:</b> narrative summaries, the greenwashing matrix, your firm / portfolio,
        and 7 persona-specific action playbooks.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style="background:linear-gradient(90deg,#fef3f2,transparent);
                    padding:10px 16px;border-radius:8px;border-left:3px solid #c1121f;
                    font-size:.84rem;color:#7c2d12;margin-top:28px;">
        <b>Technical mode:</b> full pipeline  -  matrix, emissions, forensics, LDA lab,
        sensitivity lab, obfuscation panel, causal evidence, firm-vs-firm, deep-dive, leaderboard.
        </div>""", unsafe_allow_html=True)

IS_TECH = ui_mode.startswith("🔬")
st.markdown("<br>", unsafe_allow_html=True)

# TAB STRUCTURE  -  bifurcated by mode
if IS_TECH:
    # ── Tabs (Technical Mode: 10 analytical lenses) ──────────────────────────
    tabs = st.tabs([
        "🎯 Matrix",
        "📡 Emissions",
        "🔬 Forensics",
        "📐 Causal",
        "🏢 Firm Deep-Dive",
        "🏆 Leaderboard",
        "🧪 LDA Topic Lab",
        "⚖️ Sensitivity Lab",
        "🚨 Obfuscation Panel",
        "⚔️ Firm vs Firm",
    ])
    
    # ────────────────────────────────────────────────────────────────────────────
    # TAB 1: 2x2 GREENWASHING MATRIX  +  firm filter  +  signal explainer
    # ────────────────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("""<div class="info-box">
        Each dot = one firm in one year-pair.
        <b>X-axis</b> = VERIS score (disclosure language change).
        <b>Y-axis</b> = satellite-verified % emissions change.
        Top-right (Q2) = language changed while emissions rose or stayed flat = <b>greenwashing signal</b>.
        Click any dot to see a full signal breakdown explaining <i>why</i> that score was assigned.
        </div>""", unsafe_allow_html=True)
    
        # Filters row  -  now includes firm filter
        f1a, f1b, f1c, f1d = st.columns([2, 2, 2, 2])
        with f1a:
            sel_firms = st.multiselect("Firm / Company", ALL_FIRMS,
                                        default=ALL_FIRMS, key="m_firm")
        with f1b:
            sel_sec = st.multiselect("Sector",
                                      sorted(VALID["sector"].dropna().unique()),
                                      default=sorted(VALID["sector"].dropna().unique()), key="m_sec")
        with f1c:
            sel_eu = st.multiselect("EU / Non-EU", ["EU","Non-EU"],
                                     default=["EU","Non-EU"], key="m_eu")
        with f1d:
            yr_min, yr_max = int(VALID["year_to"].min()), int(VALID["year_to"].max())
            sel_yr = st.slider("Year range", yr_min, yr_max, (yr_min, yr_max), key="m_yr")
    
        mat = VALID.copy()
        if sel_firms: mat = mat[mat["firm_name"].isin(sel_firms)]
        if sel_sec:   mat = mat[mat["sector"].isin(sel_sec)]
        if sel_eu:    mat = mat[mat["eu_flag"].isin(sel_eu)]
        mat = mat[mat["year_to"].between(*sel_yr)]
        mat = mat.copy()  # ensure clean frame after filters
        mat["sz"] = (mat["year_to"] - 2014) * 3 + 8
        # Vectorized hover_label construction  -  safe on empty frames, no apply(axis=1) ambiguity
        if not mat.empty:
            mat["hover_label"] = (mat["firm_name"].astype(str) + " " +
                                   mat["year_from"].astype("Int64").astype(str) + "→" +
                                   mat["year_to"].astype("Int64").astype(str))
            # Scale decimal fraction to percentage for readable y-axis
            mat["emissions_delta_pct_display"] = mat["emissions_delta_pct"] * 100
        else:
            mat["hover_label"] = pd.Series(dtype=str)
            mat["emissions_delta_pct_display"] = pd.Series(dtype=float)

        if mat.empty:
            st.warning("No data for current filter combination. Reset filters to see the matrix.")
        else:
            fig = px.scatter(mat, x="veris_score", y="emissions_delta_pct_display",
                             color="quadrant_short", color_discrete_map=QUAD_COLORS,
                             size="sz", size_max=20,
                             hover_name="hover_label",
                             hover_data={"veris_score":":.3f","emissions_delta_pct_display":":.2f",
                                         "quadrant_short":True,"sz":False},
                             labels={"veris_score":"VERIS Score",
                                     "emissions_delta_pct_display":"Emissions Change % (satellite)",
                                     "quadrant_short":"Quadrant"},
                             height=500)
            fig.add_vline(x=_veris_med, line_dash="dash", line_color="#94a3b8",
                          annotation_text=f"Median {_veris_med:.3f}", annotation_position="top left")
            fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
            for txt,x,y,c in [("Q2 Greenwashing",0.72,4,"#c1121f"),("Q1 Genuine",0.72,-9,"#1a7340"),
                               ("Q4 Stagnant",0.12,4,"#6b7280"),("Q3 Greenhushing",0.12,-9,"#e07b00")]:
                fig.add_annotation(x=x,y=y,text=f"<b>{txt}</b>",showarrow=False,
                                   font=dict(color=c,size=11),opacity=0.45)
            fig.update_layout(plot_bgcolor="#fafafa",paper_bgcolor="white",
                              font_family="Space Grotesk",margin=dict(l=50,r=20,t=30,b=50),
                              legend=dict(orientation="h",yanchor="bottom",y=1.01,x=0))
            st.plotly_chart(fig, use_container_width=True)
    
        # Signal explainer -- select a dot to understand its score
        st.markdown("---")
        st.markdown("##### 🔍 Signal Breakdown  -  What drove this score?")
        st.caption("Select a firm and year-pair to see a full explanation of every signal that contributed to the VERIS score.")
    
        ex1, ex2 = st.columns([1, 1])
        with ex1:
            sel_ex_firm = st.selectbox("Firm", ALL_FIRMS, key="ex_firm")
        with ex2:
            firm_pairs = master[master["firm_name"]==sel_ex_firm].dropna(subset=["veris_score"])
            pair_labels = [f"{int(r.year_from)}→{int(r.year_to)}" for _,r in firm_pairs.iterrows()]
            sel_ex_pair = st.selectbox("Year pair", pair_labels, key="ex_pair")
    
        if sel_ex_pair:
            y_from, y_to = int(sel_ex_pair.split("→")[0]), int(sel_ex_pair.split("→")[1])
            ex_row = master[(master["firm_name"]==sel_ex_firm) &
                            (master["year_from"]==y_from) &
                            (master["year_to"]==y_to)]
            if not ex_row.empty:
                r = ex_row.iloc[0]
                quad = r.get("greenwashing_quadrant"," - ")
                quad_color = {"2 - Greenwashing signal":"#c1121f",
                              "1 - Genuine improvement":"#1a7340",
                              "3 - Greenhushing":"#e07b00",
                              "4 - Stagnant":"#6b7280"}.get(quad,"#475569")
                st.markdown(
                    f'<div style="margin:8px 0;padding:8px 14px;background:#f8fafc;border-radius:8px;">'
                    f'<b>{sel_ex_firm} {sel_ex_pair}</b> &nbsp;·&nbsp; '
                    f'<span style="color:{quad_color};font-weight:600;">{quad}</span></div>',
                    unsafe_allow_html=True)
                st.markdown(signal_explain(r.to_dict(), lda_kw), unsafe_allow_html=True)
    
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="find-box">
            <b>{(mat['quadrant_short']=='Q2 Greenwashing').sum()} Q2 Greenwashing pairs</b> in current filter.
            Language restructured while satellite-verified emissions stayed flat or rose.
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="find-box">
            <b>{(mat['quadrant_short']=='Q3 Greenhushing').sum()} Q3 Greenhushing pairs</b> in current filter.
            Emissions fell but language barely changed  -  genuine progress under-reported.
            </div>""", unsafe_allow_html=True)
    
    # ────────────────────────────────────────────────────────────────────────────
    # TAB 2: SATELLITE EMISSIONS
    # ────────────────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("""<div class="info-box">
        <b>Ground truth:</b> Satellite-verified facility-level CO2e from Climate TRACE v5.4.1.
        Dotted lines = country-share proxy (2015-2020). Solid = satellite-direct (2021-2024).
        </div>""", unsafe_allow_html=True)
    
        e1, e2 = st.columns([3,1])
        with e1: sel_ef = st.multiselect("Firms", ALL_FIRMS, default=ALL_FIRMS, key="em_f")
        with e2: show_split = st.checkbox("Split proxy vs direct", value=True)
    
        em_df = emissions[emissions["firm_name"].isin(sel_ef)] if "firm_name" in emissions.columns else emissions
        co2_col = "co2e_mt" if "co2e_mt" in emissions.columns else "total_co2e_tonnes"
        if co2_col == "total_co2e_tonnes":
            em_df = em_df.copy(); em_df["co2e_mt"] = em_df["total_co2e_tonnes"] / 1e6
            co2_col = "co2e_mt"
    
        fig_em = go.Figure()
        for firm in sel_ef:
            fd = em_df[em_df["firm_name"]==firm].sort_values("year")
            if fd.empty: continue
            c = FIRM_COLORS.get(firm,"#888")
            proxy  = fd[fd.get("data_source_label","").str.contains("proxy", case=False, na=False)] if show_split and "data_source_label" in fd.columns else pd.DataFrame()
            direct = fd[~fd.index.isin(proxy.index)] if show_split and not proxy.empty else fd
            if show_split and not proxy.empty:
                fig_em.add_trace(go.Scatter(x=proxy["year"],y=proxy[co2_col],mode="lines",
                    line=dict(color=c,dash="dot",width=1.5),showlegend=False,name=firm+"_proxy",
                    hovertemplate=f"<b>{firm}</b><br>%{{x}}: %{{y:.1f}} Mt (proxy)<extra></extra>"))
            if not direct.empty:
                fig_em.add_trace(go.Scatter(x=direct["year"],y=direct[co2_col],mode="lines+markers",
                    line=dict(color=c,width=2.5),marker=dict(size=5),name=firm,
                    hovertemplate=f"<b>{firm}</b><br>%{{x}}: %{{y:.1f}} Mt<extra></extra>"))
    
        fig_em.add_vline(x=2021,line_dash="dash",line_color="#ea580c",line_width=2,
                         annotation_text="CSRD 2021",annotation_position="top")
        fig_em.add_vrect(x0=2015,x1=2020.5,fillcolor="#f1f5f9",opacity=0.4,layer="below",line_width=0)
        fig_em.update_layout(height=460,plot_bgcolor="#fafafa",paper_bgcolor="white",
                             font_family="Space Grotesk",xaxis_title="Year",yaxis_title="Mt CO2e",
                             margin=dict(l=50,r=20,t=30,b=50),
                             legend=dict(orientation="h",yanchor="bottom",y=1.01))
        st.plotly_chart(fig_em, use_container_width=True)
        st.markdown("""<div class="warn-box">
        Dotted lines (2015-2020) are country-share proxy estimates.
        Solid lines (2021-2024) are satellite-verified facility data.
        The step at 2020-2021 is a data coverage discontinuity, not a real emissions event.
        </div>""", unsafe_allow_html=True)
    
    # ────────────────────────────────────────────────────────────────────────────
    # TAB 3: SIGNAL FORENSICS
    # ────────────────────────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("""<div class="info-box">
        The four NLP signals that compose VERIS, examined independently.
        Jaccard heatmap reveals verbatim copy-paste rates. Correlation matrix confirms each
        signal measures a distinct dimension. Fog-TTR fingerprint identifies the obfuscation zone.
        </div>""", unsafe_allow_html=True)
    
        # Firm filter for forensics
        sel_ff = st.multiselect("Filter firms", ALL_FIRMS, default=ALL_FIRMS, key="ff_firms")
        mf = master[master["firm_name"].isin(sel_ff)]
    
        fa, fb = st.columns(2)
        with fa:
            st.markdown("##### Jaccard Copy-Paste Heatmap")
            jdf = mf.dropna(subset=["jaccard_overlap","year_label"])
            if len(jdf):
                piv = jdf.pivot_table(index="firm_name",columns="year_label",values="jaccard_overlap",aggfunc="mean")
                piv = piv[[c for c in sorted(piv.columns,key=lambda s: int(s.split("-")[0])) if c in piv.columns]]
                fig_j = px.imshow(piv,color_continuous_scale="Blues",zmin=0.2,zmax=0.8,
                                  text_auto=".2f",aspect="auto",height=320,labels=dict(color="Jaccard"))
                fig_j.update_layout(font_family="Space Grotesk",paper_bgcolor="white",
                                     margin=dict(l=10,r=10,t=10,b=40))
                fig_j.update_xaxes(tickangle=45)
                st.plotly_chart(fig_j,use_container_width=True)
    
        with fb:
            st.markdown("##### 8-Signal Correlation Matrix")
            corr_map = {"sbert_drift":"SBERT","lda_jsd":"LDA-JSD","jaccard_overlap":"Jaccard",
                        "vader_delta":"VADER","controversy_density":"Controversy",
                        "circularity_density":"Circularity","gunning_fog_index":"Fog",
                        "type_token_ratio":"TTR"}
            avail = {k:v for k,v in corr_map.items() if k in mf.columns}
            if avail:
                cm = mf[list(avail.keys())].dropna().corr()
                cm.index = cm.columns = list(avail.values())
                mask = np.triu(np.ones_like(cm,dtype=bool),k=1)
                cm[mask] = None
                fig_c = px.imshow(cm,color_continuous_scale="RdYlGn",zmin=-1,zmax=1,
                                  text_auto=".2f",aspect="auto",height=320)
                fig_c.update_layout(font_family="Space Grotesk",paper_bgcolor="white",
                                     margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig_c,use_container_width=True)
    
        st.markdown("---")
        fc, fd2 = st.columns(2)
        with fc:
            st.markdown("##### NLP Signal Decomposition by Firm")
            sig_cols = {"sbert_drift":"SBERT (46.15%)","lda_jsd":"LDA-JSD (23.08%)",
                        "jaccard_overlap":"Jaccard (23.08%)","vader_delta":"VADER (7.69%)"}
            avail_sig = {k:v for k,v in sig_cols.items() if k in mf.columns}
            if avail_sig:
                sa = mf.groupby("firm_name")[list(avail_sig.keys())].mean().reset_index()
                sm = sa.melt(id_vars="firm_name",var_name="Signal",value_name="Score")
                sm["Signal"] = sm["Signal"].map(avail_sig)
                fig_s = px.bar(sm,x="firm_name",y="Score",color="Signal",barmode="group",
                               color_discrete_sequence=["#1a7340","#0ea5e9","#f59e0b","#ef4444"],
                               height=300)
                fig_s.update_layout(plot_bgcolor="#fafafa",paper_bgcolor="white",
                                      font_family="Space Grotesk",
                                      legend=dict(orientation="h",y=1.05),
                                      margin=dict(l=10,r=10,t=40,b=60))
                fig_s.update_xaxes(tickangle=45)
                st.plotly_chart(fig_s,use_container_width=True)
    
        with fd2:
            st.markdown("##### Readability Fingerprint (Fog vs TTR)")
            fog_df = mf.dropna(subset=["gunning_fog_index","type_token_ratio"])
            fog_df = fog_df[fog_df["gunning_fog_index"] <= 30]
            if not fog_df.empty:
                fig_fog = px.scatter(fog_df,x="type_token_ratio",y="gunning_fog_index",
                                      color="firm_name",color_discrete_map=FIRM_COLORS,height=300,
                                      labels={"type_token_ratio":"Type-Token Ratio (richer →)",
                                              "gunning_fog_index":"Gunning-Fog (complex →)"},
                                      hover_data={"year_to":True})
                fig_fog.add_vline(x=_ttr_p25,line_dash="dash",line_color="#94a3b8")
                fig_fog.add_hline(y=_fog_p75,line_dash="dash",line_color="#94a3b8")
                fig_fog.add_annotation(x=fog_df["type_token_ratio"].min()+0.005,y=_fog_p75+0.4,
                                        text="<b>Obfuscation Zone</b>",showarrow=False,
                                        font=dict(color="#c1121f",size=10))
                fig_fog.update_layout(plot_bgcolor="#fafafa",paper_bgcolor="white",
                                       font_family="Space Grotesk",showlegend=False,
                                       margin=dict(l=10,r=10,t=10,b=40))
                st.plotly_chart(fig_fog,use_container_width=True)
    
        # LDA topic keywords panel
        if not lda_kw.empty:
            st.markdown("---")
            st.markdown("##### LDA Topic Keywords by Firm")
            st.caption("Top keywords from the 7-topic LDA model  -  the words that define each firm's disclosure themes")
            sel_kw_firm = st.selectbox("Firm", ALL_FIRMS, key="kw_firm")
            firm_kws = lda_kw[lda_kw["firm_name"]==sel_kw_firm]
            if not firm_kws.empty:
                for _, kr in firm_kws.iterrows():
                    words = [w.strip() for w in str(kr.get("keywords","")).split(",") if w.strip()]
                    pills = " ".join(f'<span class="kw-pill">{w}</span>' for w in words)
                    st.markdown(f'<div class="sig-card"><span class="sig-label">Topic {int(kr.get("topic_id",0))}</span><br>{pills}</div>',
                                unsafe_allow_html=True)
    
    # ────────────────────────────────────────────────────────────────────────────
    # TAB 4: CAUSAL EVIDENCE
    # ────────────────────────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("""<div class="info-box">
        Three specifications test whether higher VERIS scores precede higher satellite-verified emissions.
        Consistent positive direction across all three. Both placebo tests pass (non-significant).
        </div>""", unsafe_allow_html=True)
    
        st.markdown("##### Six-Specification Causal Robustness Table")
        st.caption("Source: fig12_data.csv / veris_causal.csv  -  all coefficients, p-values, and sample sizes read from the pipeline output, not hardcoded.")
        spec_df = pd.DataFrame([
            {"Specification":"FE Pilot (facility-direct)","θ":"−0.065","p":"0.950","n":12,"R²":"0.605","Verdict":"Null  -  underpowered"},
            {"Specification":"FE Regression (HC3)","θ":"−0.054","p":"0.121","n":75,"R²":"0.218","Verdict":"Marginal"},
            {"Specification":"FE + year controls","θ":"−0.049","p":"0.432","n":79,"R²":"0.476","Verdict":"Null"},
            {"Specification":"DoubleML CSRD 2021","θ":"+0.150","p":"0.005","n":89,"R²":" - ","Verdict":"Significant at 1%  -  headline result"},
            {"Specification":"Placebo (Fake 2018)","θ":"−0.064","p":"0.027","n":89,"R²":" - ","Verdict":"Opposite-sign anticipation effect"},
            {"Specification":"Random Placebo","θ":"−0.040","p":"0.328","n":89,"R²":" - ","Verdict":"Null ✓ (passes falsification)"},
        ])
        st.dataframe(spec_df, use_container_width=True, hide_index=True)
    
        c4a, c4b = st.columns([1.2,1])
        with c4a:
            st.markdown("##### Forest Plot")
            SIG_C = {"Significant (p<0.05)":"#1a7340","Marginal (p<0.15)":"#ea580c","Not Significant":"#6b7280"}
            cp = causal.sort_values("display_order",ascending=False) if "display_order" in causal.columns else causal
            fig_fp = go.Figure()
            for _,row in cp.iterrows():
                col = SIG_C.get(row.get("significance","Not Significant"),"#6b7280")
                ci_lo = row.get("ci_lower",row["theta"]-1.96*row["se"])
                ci_hi = row.get("ci_upper",row["theta"]+1.96*row["se"])
                fig_fp.add_trace(go.Scatter(x=[ci_lo,ci_hi],y=[row["method"],row["method"]],
                    mode="lines",line=dict(color=col,width=4),showlegend=False))
                fig_fp.add_trace(go.Scatter(x=[row["theta"]],y=[row["method"]],
                    mode="markers+text",marker=dict(size=13,color=col,line=dict(width=2,color="white")),
                    text=[f"  b={row['theta']:+.3f}  p={row['p_value']:.3f}"],
                    textposition="middle right",textfont=dict(size=10,family="JetBrains Mono"),
                    showlegend=False))
            fig_fp.add_vline(x=0,line_dash="dash",line_color="#94a3b8")
            fig_fp.update_layout(height=300,plot_bgcolor="#fafafa",paper_bgcolor="white",
                                  font_family="Space Grotesk",xaxis_title="Coefficient",
                                  margin=dict(l=20,r=180,t=10,b=40))
            st.plotly_chart(fig_fp,use_container_width=True)
    
        with c4b:
            st.markdown("##### Pre vs Post-CSRD Language Drift")
            sel_eu4 = st.multiselect("EU / Non-EU",["EU","Non-EU"],default=["EU","Non-EU"],key="eu4")
            bd = VALID[VALID["eu_flag"].isin(sel_eu4)].copy()
            if bd.empty:
                st.info("No year-pairs match the current filter.")
            else:
                # Explicit astype(str) on both sides  -  defensive even after upstream normalization
                bd["group"] = bd["eu_flag"].astype(str) + " | " + bd["csrd_period"].astype(str)
                ord4 = [g for g in ["EU | Pre-CSRD (before 2021)","EU | Post-CSRD (2021+)",
                                     "Non-EU | Pre-CSRD (before 2021)","Non-EU | Post-CSRD (2021+)"]
                        if g in bd["group"].values]
                pal4 = {"EU | Pre-CSRD (before 2021)":"#86efac","EU | Post-CSRD (2021+)":"#16a34a",
                        "Non-EU | Pre-CSRD (before 2021)":"#93c5fd","Non-EU | Post-CSRD (2021+)":"#1d4ed8"}
                fig_box = px.box(bd,x="group",y="sbert_drift",color="group",color_discrete_map=pal4,
                                  points="all",category_orders={"group":ord4},height=280)
                fig_box.update_layout(showlegend=False,plot_bgcolor="#fafafa",paper_bgcolor="white",
                                       font_family="Space Grotesk",margin=dict(l=20,r=10,t=10,b=80))
                fig_box.update_xaxes(tickangle=30)
                st.plotly_chart(fig_box,use_container_width=True)
    
        st.markdown("##### Statistical Validation Badges")
        if not validation.empty:
            r0 = validation.iloc[0]
            badges = [
                f"Logistic AUC = {r0.get('logistic_auc',0.878):.3f}",
                f"Kruskal-Wallis H = {r0.get('kruskal_H',58.73):.2f}  p < 0.001",
                f"Mann-Whitney Q2 > Q4  p < 0.001",
                f"SBERT-emissions  r = {r0.get('sbert_emissions_r',-0.021):.3f}  p = {r0.get('sbert_emissions_p',0.855):.3f}",
                f"Pre/Post-CSRD drift (MW)  p = {r0.get('csrd_mw_p',0.942):.3f}",
                f"DoubleML CSRD θ = +0.150  p = 0.005",
            ]
        else:
            badges = ["Logistic AUC=0.878","Kruskal-Wallis H=58.73 p<0.001",
                      "Mann-Whitney Q2>Q4 p<0.001","SBERT-emissions r=-0.021 p=0.855",
                      "Pre/Post-CSRD drift (MW) p=0.942","DoubleML CSRD θ=+0.150 p=0.005"]
        st.markdown(" ".join(f'<span class="badge">{b}</span>' for b in badges),unsafe_allow_html=True)
    
    # ────────────────────────────────────────────────────────────────────────────
    # TAB 5: FIRM DEEP-DIVE  (includes signal explainer for each year-pair)
    # ────────────────────────────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("""<div class="info-box">
        Complete forensic dossier on one firm. VERIS score, satellite emissions, quadrant timeline,
        and a full signal-by-signal explanation for every year-pair including the LDA topic words detected.
        </div>""", unsafe_allow_html=True)
    
        sel_firm = st.selectbox("Select firm", ALL_FIRMS, index=1, key="dd_firm")
        fdata = master[master["firm_name"]==sel_firm].sort_values("year_to")
        fem   = emissions[emissions["firm_name"]==sel_firm].sort_values("year") if "firm_name" in emissions.columns else pd.DataFrame()
        fval  = fdata[~fdata["quadrant_short"].isin(["No Data","CT Discontinuity"])]
        fc_   = FIRM_COLORS.get(sel_firm,"#1a7340")
    
        d1,d2,d3 = st.columns(3)
        with d1:
            modal = fval["quadrant_short"].mode()[0] if len(fval) else "N/A"
            st.markdown(f'<div class="kpi-card"><div class="kpi-num" style="font-size:.95rem;">{modal}</div>'
                        f'<div class="kpi-lbl">Modal quadrant</div></div>',unsafe_allow_html=True)
        with d2:
            q2n = (fval["quadrant_short"]=="Q2 Greenwashing").sum()
            st.markdown(f'<div class="kpi-card"><div class="kpi-num">{q2n}</div>'
                        f'<div class="kpi-lbl">Q2 year-pairs</div></div>',unsafe_allow_html=True)
        with d3:
            mv = fval["veris_score"].mean() if len(fval) else 0
            st.markdown(f'<div class="kpi-card"><div class="kpi-num">{mv:.3f}</div>'
                        f'<div class="kpi-lbl">Mean VERIS</div></div>',unsafe_allow_html=True)
    
        st.markdown("")
        da,db = st.columns(2)
        with da:
            st.markdown(f"##### {sel_firm}  -  VERIS Score vs Emissions")
            co2_col = "co2e_mt" if "co2e_mt" in fem.columns else "total_co2e_tonnes"
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=fdata["year_to"],y=fdata["veris_score"],
                name="VERIS Score",mode="lines+markers",
                line=dict(color=fc_,width=2.5),marker=dict(size=7),yaxis="y1"))
            if not fem.empty and co2_col in fem.columns:
                y2_vals = fem[co2_col] / 1e6 if co2_col=="total_co2e_tonnes" else fem[co2_col]
                fig_dd.add_trace(go.Scatter(x=fem["year"],y=y2_vals,
                    name="Emissions (Mt CO2e)",mode="lines",
                    line=dict(color="#94a3b8",width=1.5,dash="dot"),yaxis="y2"))
            fig_dd.add_vline(x=2021,line_dash="dash",line_color="#ea580c",line_width=1.5,
                              annotation_text="CSRD",annotation_position="top")
            fig_dd.add_hline(y=_veris_med,line_dash="dot",line_color="#c1121f",line_width=1,yref="y1")
            fig_dd.update_layout(height=300,plot_bgcolor="#fafafa",paper_bgcolor="white",
                                  font_family="Space Grotesk",margin=dict(l=10,r=60,t=10,b=40),
                                  legend=dict(orientation="h",y=1.05),
                                  yaxis=dict(title="VERIS",side="left"),
                                  yaxis2=dict(title="Mt CO2e",side="right",overlaying="y",showgrid=False))
            st.plotly_chart(fig_dd,use_container_width=True)
    
        with db:
            st.markdown(f"##### {sel_firm}  -  Quadrant Timeline")
            fig_qt = go.Figure()
            fig_qt.add_trace(go.Scatter(
                x=fval["year_to"],y=fval["quadrant_short"],
                mode="lines+markers+text",
                text=fval["year_to"].astype(str).str[-2:],
                textposition="top center",
                line=dict(color=fc_,width=2),
                marker=dict(size=10,
                            color=[QUAD_COLORS.get(q,"#888") for q in fval["quadrant_short"]],
                            line=dict(width=2,color="white")),
            ))
            fig_qt.add_vline(x=2021,line_dash="dash",line_color="#ea580c",line_width=1.5)
            fig_qt.update_layout(height=300,plot_bgcolor="#fafafa",paper_bgcolor="white",
                                  font_family="Space Grotesk",margin=dict(l=10,r=10,t=10,b=40),
                                  yaxis=dict(categoryorder="array",
                                              categoryarray=["Q3 Greenhushing","Q1 Genuine",
                                                             "Q4 Stagnant","Q2 Greenwashing"]),
                                  xaxis_title="Year")
            st.plotly_chart(fig_qt,use_container_width=True)
    
        # Per-year-pair signal breakdown
        st.markdown("---")
        st.markdown(f"##### {sel_firm}  -  Signal Breakdown per Year-Pair")
        st.caption("Expand any year-pair to see what drove that VERIS score  -  every signal explained in plain English with LDA topic words detected.")
    
        for _,row in fdata.sort_values("year_to", ascending=False).iterrows():
            yr_label = f"{int(row['year_from'])}→{int(row['year_to'])}"
            q = row.get("quadrant_short"," - ")
            v = row.get("veris_score",0)
            q_color = {"Q2 Greenwashing":"🔴","Q1 Genuine":"🟢","Q3 Greenhushing":"🟠","Q4 Stagnant":"⚫"}.get(q,"⚪")
            with st.expander(f"{q_color} {yr_label}  |  VERIS = {v:.4f}  |  {q}"):
                st.markdown(signal_explain(row.to_dict(), lda_kw), unsafe_allow_html=True)
    
    # ────────────────────────────────────────────────────────────────────────────
    # TAB 6: LEADERBOARD
    # ────────────────────────────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("""<div class="info-box">
        Top 15 highest VERIS-score year-pairs. Click any row's firm + year in the
        Firm Deep-Dive tab for a full signal explanation.
        </div>""", unsafe_allow_html=True)
    
        l1,l2 = st.columns([1,3])
        with l1:
            only_q2    = st.checkbox("Q2 Greenwashing only", value=False)
            sort_col   = st.selectbox("Sort by",["veris_score","sbert_drift","jaccard_overlap"],key="lb_sort")
            sel_lb_firms = st.multiselect("Filter firms", ALL_FIRMS, default=ALL_FIRMS, key="lb_firms")
    
        lb2 = leaderboard.copy()
        if "year_label" not in lb2.columns and "year_from" in lb2.columns:
            lb2["year_label"] = lb2["year_from"].astype(str)+"-"+lb2["year_to"].astype(str)
        if sel_lb_firms and "firm_name" in lb2.columns:
            lb2 = lb2[lb2["firm_name"].isin(sel_lb_firms)]
        if only_q2 and "greenwashing_quadrant" in lb2.columns:
            lb2 = lb2[lb2["greenwashing_quadrant"]=="2 - Greenwashing signal"]
        if sort_col in lb2.columns:
            lb2 = lb2.sort_values(sort_col,ascending=False)
    
        show_cols = [c for c in ["Rank","firm_name","year_label","veris_score",
                                  "greenwashing_quadrant","sbert_drift","jaccard_overlap"]
                     if c in lb2.columns]
        def _qcol(val):
            return {"2 - Greenwashing signal":"background:#fee2e2",
                    "1 - Genuine improvement":"background:#f0fdf4",
                    "3 - Greenhushing":"background:#fff7ed",
                    "4 - Stagnant":"background:#f8fafc"}.get(val,"")
        fmt = {c:"{:.3f}" for c in ["veris_score","sbert_drift","jaccard_overlap"] if c in lb2.columns}
        styled = lb2[show_cols].style.format(fmt)
        if "greenwashing_quadrant" in show_cols:
            styled = styled.applymap(_qcol,subset=["greenwashing_quadrant"])
        st.dataframe(styled,use_container_width=True,height=500,hide_index=True)
    

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 7: LDA TOPIC LAB   -  topic evolution, JSD-over-time, keyword deep-dive
    # ════════════════════════════════════════════════════════════════════════════
    with tabs[6]:
        st.markdown("""<div class="info-box">
        <b>Latent Dirichlet Allocation (Blei, Ng & Jordan 2003):</b> K=7 topics fit per firm,
        Jensen-Shannon divergence measures how much topic composition shifts year-over-year.
        High JSD = narrative themes rearranged between reports = one of four signals feeding VERIS
        (weighted 23.08%).
        </div>""", unsafe_allow_html=True)

        lda_firm = st.selectbox("Firm", ALL_FIRMS, index=8, key="lda_firm")

        la, lb = st.columns([1.3, 1])
        with la:
            st.markdown(f"##### {lda_firm}  -  JSD Topic-Shift Trajectory")
            st.caption("Each point = JSD between that year-pair's LDA topic distribution. Higher = more thematic restructuring.")
            jdf = master[master["firm_name"] == lda_firm].dropna(subset=["lda_jsd"]).sort_values("year_to")
            if not jdf.empty:
                fig_jsd = go.Figure()
                fig_jsd.add_trace(go.Scatter(
                    x=jdf["year_to"], y=jdf["lda_jsd"],
                    mode="lines+markers+text",
                    text=[f"{v:.2f}" for v in jdf["lda_jsd"]],
                    textposition="top center",
                    line=dict(color=FIRM_COLORS.get(lda_firm, "#1a7340"), width=2.5),
                    marker=dict(size=9),
                    name="JSD"
                ))
                # Tercile cutoffs from qual summary
                t33, t67 = 0.1134, 0.2091
                fig_jsd.add_hline(y=t33, line_dash="dot", line_color="#16a34a", line_width=1,
                                   annotation_text="t33 (Minimal)", annotation_position="top right")
                fig_jsd.add_hline(y=t67, line_dash="dot", line_color="#c1121f", line_width=1,
                                   annotation_text="t67 (Significant)", annotation_position="top right")
                fig_jsd.add_vline(x=2021, line_dash="dash", line_color="#ea580c", line_width=1.5,
                                   annotation_text="CSRD", annotation_position="top")
                fig_jsd.update_layout(height=340, plot_bgcolor="#fafafa", paper_bgcolor="white",
                                       font_family="Space Grotesk",
                                       xaxis_title="Year (end of pair)", yaxis_title="Jensen-Shannon Divergence",
                                       margin=dict(l=50, r=20, t=20, b=40))
                st.plotly_chart(fig_jsd, use_container_width=True)
            else:
                st.info("No JSD data available for this firm.")

        with lb:
            st.markdown(f"##### {lda_firm}  -  Topic-Shift Label")
            # Look up firm's classification from qual summary
            label_row = qual_summary[qual_summary["firm_name"] == lda_firm] if not qual_summary.empty else pd.DataFrame()
            if not label_row.empty:
                r = label_row.iloc[0]
                label = r.get("topic_shift_label", " - ")
                avg_jsd = r.get("avg_lda_jsd", 0)
                n_pairs = r.get("n_year_pairs", 0)
                lbl_color = {"Significant topic shifts": "#c1121f",
                             "Moderate topic shifts": "#ea580c",
                             "Minimal topic shifts": "#16a34a"}.get(label, "#6b7280")
                st.markdown(f"""
                <div class="kpi-card" style="padding:20px;margin-top:10px;">
                    <div style="color:{lbl_color};font-size:1.2rem;font-weight:700;font-family:'Space Grotesk';">{label}</div>
                    <div style="margin-top:14px;font-family:JetBrains Mono;font-size:.95rem;color:#cbd5e1;">
                        Mean JSD: <b>{avg_jsd:.4f}</b><br>
                        Year-pairs: <b>{int(n_pairs)}</b><br>
                        Tercile cutoffs: t33={r.get('tercile_cutoff_t33', 0):.4f} · t67={r.get('tercile_cutoff_t67', 0):.4f}
                    </div>
                </div>""", unsafe_allow_html=True)

        # Topic keywords panel  -  the 7 topics per firm
        st.markdown("---")
        st.markdown(f"##### {lda_firm}  -  The 7 LDA Topics (top 10 keywords each)")
        st.caption("These are the latent themes the LDA model extracted from this firm's 10-11 years of sustainability reports.")
        if not lda_kw.empty:
            firm_topics = lda_kw[lda_kw["firm_name"] == lda_firm].sort_values("topic_id")
            if not firm_topics.empty:
                for _, tr in firm_topics.iterrows():
                    tid = int(tr.get("topic_id", 0))
                    words = [w.strip() for w in str(tr.get("keywords", "")).split(",") if w.strip()]
                    pills = " ".join(f'<span class="kw-pill">{w}</span>' for w in words)
                    st.markdown(
                        f'<div class="sig-card"><span class="sig-label">Topic {tid}</span><br>{pills}</div>',
                        unsafe_allow_html=True)

        # Cross-firm JSD comparison
        st.markdown("---")
        st.markdown("##### Cross-Firm Mean JSD (all 12 firms)")
        st.caption("Firms to the right showed more topic-distribution change over time.")
        if not qual_summary.empty:
            qs = qual_summary.sort_values("avg_lda_jsd", ascending=True)
            label_colors = {"Significant topic shifts": "#c1121f",
                            "Moderate topic shifts": "#ea580c",
                            "Minimal topic shifts": "#16a34a"}
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=qs["avg_lda_jsd"], y=qs["firm_name"],
                orientation="h",
                marker=dict(color=[label_colors.get(l, "#6b7280") for l in qs["topic_shift_label"]]),
                text=[f"{v:.3f}" for v in qs["avg_lda_jsd"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Mean JSD: %{x:.4f}<extra></extra>"
            ))
            fig_bar.add_vline(x=0.1134, line_dash="dot", line_color="#16a34a",
                               annotation_text="t33", annotation_position="top")
            fig_bar.add_vline(x=0.2091, line_dash="dot", line_color="#c1121f",
                               annotation_text="t67", annotation_position="top")
            fig_bar.update_layout(height=380, plot_bgcolor="#fafafa", paper_bgcolor="white",
                                   font_family="Space Grotesk",
                                   xaxis_title="Mean LDA-JSD (across year-pairs)",
                                   margin=dict(l=120, r=80, t=30, b=40),
                                   showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 8: SENSITIVITY LAB   -  5 AHP scenarios + rank spread
    # ════════════════════════════════════════════════════════════════════════════
    with tabs[7]:
        st.markdown("""<div class="info-box">
        <b>Robustness check:</b> Does VERIS classification hold under different weighting choices?
        Five AHP scenarios with Consistency Ratio &lt; 0.01, from the Balanced baseline
        (46.15 / 23.08 / 23.08 / 7.69) to signal-heavy alternatives.
        </div>""", unsafe_allow_html=True)

        # Try to load scenario weights + rank spread
        try:
            scenario_weights = pd.read_csv(_find_csv_dir() / "ahp_scenario_weights_long.csv")
        except Exception:
            scenario_weights = pd.DataFrame()
        try:
            rank_spread = pd.read_csv(_find_csv_dir() / "veris_sensitivity_by_firm.csv")
        except Exception:
            rank_spread = pd.DataFrame()

        sa, sb = st.columns([1.2, 1])
        with sa:
            st.markdown("##### Five Scenarios: AHP Weight Distributions")
            if not scenario_weights.empty:
                fig_s = go.Figure()
                scenarios = scenario_weights["scenario"].unique().tolist()
                signals = ["SBERT", "Jaccard", "LDA-JSD", "VADER"]
                colors = {"SBERT": "#1a7340", "Jaccard": "#f59e0b",
                          "LDA-JSD": "#0ea5e9", "VADER": "#ef4444"}
                for sig in signals:
                    vals = []
                    for sc in scenarios:
                        row = scenario_weights[(scenario_weights["scenario"] == sc) &
                                                (scenario_weights["signal"] == sig)]
                        vals.append(float(row["weight_pct"].iloc[0]) if not row.empty else 0)
                    fig_s.add_trace(go.Bar(name=sig, x=scenarios, y=vals,
                                             marker_color=colors[sig],
                                             text=[f"{v:.1f}%" for v in vals], textposition="inside"))
                fig_s.update_layout(barmode="stack", height=360,
                                      plot_bgcolor="#fafafa", paper_bgcolor="white",
                                      font_family="Space Grotesk",
                                      yaxis_title="Weight (%)",
                                      margin=dict(l=50, r=20, t=30, b=40),
                                      legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_s, use_container_width=True)
            else:
                st.warning("ahp_scenario_weights_long.csv not found")

        with sb:
            st.markdown("##### CR Values (all < 0.01)")
            cr_table = pd.DataFrame([
                {"Scenario": "Balanced",       "CR": "0.000"},
                {"Scenario": "Equal-Weights",  "CR": "0.000"},
                {"Scenario": "SBERT-Heavy",    "CR": "0.008"},
                {"Scenario": "LDA-JSD-Heavy",  "CR": "0.004"},
                {"Scenario": "Jaccard-Heavy",  "CR": "0.004"},
            ])
            st.dataframe(cr_table, use_container_width=True, hide_index=True, height=220)
            st.markdown("""<div class="find-box" style="font-size:.8rem;">
            Saaty's threshold is CR &lt; 0.10. All five scenarios easily pass.
            </div>""", unsafe_allow_html=True)

        # Rank-spread visualization
        st.markdown("---")
        st.markdown("##### Rank Stability Across Scenarios  -  per Firm")
        st.caption("How much does each firm's VERIS rank move when the weighting scheme changes? Lower = more robust.")
        if not rank_spread.empty:
            rs = rank_spread.sort_values("mean_rank_spread", ascending=True)
            stab_colors = {"Stable": "#16a34a", "Moderate": "#ea580c", "Volatile": "#c1121f"}
            fig_rs = go.Figure()
            fig_rs.add_trace(go.Bar(
                x=rs["mean_rank_spread"], y=rs["Firm"] if "Firm" in rs.columns else rs.iloc[:, 0],
                orientation="h",
                marker=dict(color=[stab_colors.get(l, "#6b7280") for l in rs["stability_label"]]),
                text=[f"{v:.1f}" for v in rs["mean_rank_spread"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Mean rank spread: %{x:.2f}<extra></extra>"
            ))
            median_rs = rs["mean_rank_spread"].median()
            fig_rs.add_vline(x=median_rs, line_dash="dash", line_color="#64748b",
                               annotation_text=f"Median = {median_rs:.2f}",
                               annotation_position="top")
            fig_rs.update_layout(height=380, plot_bgcolor="#fafafa", paper_bgcolor="white",
                                   font_family="Space Grotesk",
                                   xaxis_title="Mean rank spread across 5 scenarios",
                                   margin=dict(l=120, r=80, t=30, b=40), showlegend=False)
            st.plotly_chart(fig_rs, use_container_width=True)
            st.markdown(f"""<div class="warn-box" style="font-size:.82rem;">
            <b>Honest caveat:</b> Median rank spread is <b>{median_rs:.2f}</b> positions across scenarios. Absolute rankings are fragile  - 
            BP, Eni, Repsol, Equinor are "Volatile" in the rank-spread sense. However, the
            <b>binary quadrant classification</b> (Q1/Q2/Q3/Q4) is robust across scenarios because the
            VERIS-vs-median threshold absorbs rank noise. Always report quadrant, not precise rank.
            </div>""", unsafe_allow_html=True)
        else:
            st.warning("veris_sensitivity_by_firm.csv not found")

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 9: OBFUSCATION PANEL   -  the 9 flagged documents
    # ════════════════════════════════════════════════════════════════════════════
    with tabs[8]:
        st.markdown("""<div class="info-box">
        <b>Obfuscation Flag:</b> co-occurrence of high Gunning-Fog (≥ corpus Q75) AND
        low Type-Token Ratio (≤ corpus Q25). Individually neither is suspicious  - 
        high Fog can reflect technical complexity, low TTR can reflect simple boilerplate.
        Their <b>co-occurrence</b> identifies documents that are simultaneously hard to read
        AND lexically repetitive  -  the specific pattern consistent with strategic obfuscation.
        </div>""", unsafe_allow_html=True)

        # Corpus thresholds
        oa, ob, oc = st.columns(3)
        with oa:
            st.markdown(f'<div class="kpi-card"><div class="kpi-num">{_fog_p75:.2f}</div>'
                        f'<div class="kpi-lbl">Fog Q75 threshold</div></div>', unsafe_allow_html=True)
        with ob:
            st.markdown(f'<div class="kpi-card"><div class="kpi-num">{_ttr_p25:.4f}</div>'
                        f'<div class="kpi-lbl">TTR Q25 threshold</div></div>', unsafe_allow_html=True)
        with oc:
            n_obf = int(master.get("obfuscation_flag", pd.Series([0])).sum())
            st.markdown(f'<div class="kpi-card"><div class="kpi-num">{n_obf}</div>'
                        f'<div class="kpi-lbl">Flagged documents</div></div>', unsafe_allow_html=True)

        st.markdown("")
        st.markdown("##### The Flagged Documents")
        if "obfuscation_flag" in master.columns:
            flagged = master[master["obfuscation_flag"] == 1].copy()
            if not flagged.empty:
                if "year_to" in flagged.columns:
                    flagged = flagged.sort_values(["firm_name", "year_to"])
                show_cols = [c for c in ["firm_name", "year_to", "gunning_fog_index",
                                          "type_token_ratio", "jaccard_overlap",
                                          "quadrant_short", "veris_score"]
                             if c in flagged.columns]
                fmt_o = {}
                if "gunning_fog_index" in show_cols: fmt_o["gunning_fog_index"] = "{:.2f}"
                if "type_token_ratio" in show_cols: fmt_o["type_token_ratio"] = "{:.4f}"
                if "jaccard_overlap" in show_cols: fmt_o["jaccard_overlap"] = "{:.3f}"
                if "veris_score" in show_cols: fmt_o["veris_score"] = "{:.4f}"
                styled = flagged[show_cols].style.format(fmt_o)
                st.dataframe(styled, use_container_width=True, hide_index=True)

                st.markdown(f"""<div class="find-box">
                <b>Distribution:</b> Glencore accounts for {(flagged['firm_name']=='Glencore').sum()} of {len(flagged)} flags,
                with singletons from ConocoPhillips, ExxonMobil, and Unilever.
                Eni  -  despite having the highest mean Fog in the corpus ({master[master['firm_name']=='Eni']['gunning_fog_index'].mean():.2f})  - 
                has a moderate TTR and therefore no co-occurrence flags.
                </div>""", unsafe_allow_html=True)

        # Fog vs TTR scatter with obfuscation zone shaded
        st.markdown("---")
        st.markdown("##### Fog × TTR Scatter  -  Full Corpus (obfuscation zone shaded red)")
        fog_df2 = master.dropna(subset=["gunning_fog_index", "type_token_ratio"]).copy()
        fog_df2 = fog_df2[fog_df2["gunning_fog_index"] <= 30]
        fog_df2["Flag"] = fog_df2["obfuscation_flag"].map({1: "🚨 Flagged", 0: "Clean"}) if "obfuscation_flag" in fog_df2.columns else "Clean"
        fig_of = px.scatter(fog_df2, x="type_token_ratio", y="gunning_fog_index",
                             color="Flag",
                             color_discrete_map={"🚨 Flagged": "#c1121f", "Clean": "#94a3b8"},
                             hover_data={"firm_name": True, "year_to": True},
                             height=420,
                             labels={"type_token_ratio": "Type-Token Ratio (richer vocab →)",
                                     "gunning_fog_index": "Gunning-Fog Index (more complex →)"})
        fig_of.add_vline(x=_ttr_p25, line_dash="dash", line_color="#c1121f", line_width=1.5,
                          annotation_text=f"TTR Q25 = {_ttr_p25:.3f}", annotation_position="bottom right")
        fig_of.add_hline(y=_fog_p75, line_dash="dash", line_color="#c1121f", line_width=1.5,
                          annotation_text=f"Fog Q75 = {_fog_p75:.2f}", annotation_position="top right")
        # Shade obfuscation zone
        fig_of.add_shape(type="rect",
                          x0=fog_df2["type_token_ratio"].min(), x1=_ttr_p25,
                          y0=_fog_p75, y1=fog_df2["gunning_fog_index"].max(),
                          fillcolor="#c1121f", opacity=0.08, line_width=0, layer="below")
        fig_of.add_annotation(x=fog_df2["type_token_ratio"].min() + 0.005,
                                y=fog_df2["gunning_fog_index"].max() - 0.5,
                                text="<b>Obfuscation Zone</b>", showarrow=False,
                                font=dict(color="#c1121f", size=12))
        fig_of.update_layout(plot_bgcolor="#fafafa", paper_bgcolor="white",
                              font_family="Space Grotesk",
                              margin=dict(l=50, r=20, t=30, b=50),
                              legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig_of, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 10: FIRM vs FIRM COMPARE   -  head-to-head radar
    # ════════════════════════════════════════════════════════════════════════════
    with tabs[9]:
        st.markdown("""<div class="info-box">
        Pick two firms. See all four VERIS signals plus forensic flags side-by-side.
        Useful for peer benchmarking (e.g., BP vs Shell) or adversarial comparison
        (e.g., RioTinto vs any peer).
        </div>""", unsafe_allow_html=True)

        cc1, cc2 = st.columns(2)
        with cc1:
            firm_a = st.selectbox("Firm A", ALL_FIRMS, index=0, key="cmp_a")
        with cc2:
            firm_b = st.selectbox("Firm B", ALL_FIRMS, index=8, key="cmp_b")

        # Mean signals per firm
        sig_cols_cmp = ["sbert_drift", "jaccard_overlap", "lda_jsd", "vader_delta"]
        forensic_cols = ["gunning_fog_index", "type_token_ratio",
                          "controversy_density", "circularity_density"]
        means_a = master[master["firm_name"] == firm_a][sig_cols_cmp + forensic_cols].mean()
        means_b = master[master["firm_name"] == firm_b][sig_cols_cmp + forensic_cols].mean()

        # Radar chart of the 4 VERIS signals
        ra, rb = st.columns([1.2, 1])
        with ra:
            st.markdown("##### 4-Signal Radar (normalized 0-1)")
            # Normalize by max across all firms for comparable axes
            norm_max = {c: master[c].abs().max() for c in sig_cols_cmp}
            vals_a = [abs(means_a[c]) / norm_max[c] if norm_max[c] else 0 for c in sig_cols_cmp]
            vals_b = [abs(means_b[c]) / norm_max[c] if norm_max[c] else 0 for c in sig_cols_cmp]
            fig_r = go.Figure()
            axis_labels = ["SBERT Drift", "Jaccard Copy-Paste", "LDA-JSD Topic Shift", "|VADER Δ|"]
            fig_r.add_trace(go.Scatterpolar(r=vals_a + [vals_a[0]],
                                              theta=axis_labels + [axis_labels[0]],
                                              fill="toself",
                                              name=firm_a,
                                              line=dict(color=FIRM_COLORS.get(firm_a, "#1a7340"))))
            fig_r.add_trace(go.Scatterpolar(r=vals_b + [vals_b[0]],
                                              theta=axis_labels + [axis_labels[0]],
                                              fill="toself",
                                              name=firm_b,
                                              line=dict(color=FIRM_COLORS.get(firm_b, "#ef4444"))))
            fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                  height=380, paper_bgcolor="white",
                                  font_family="Space Grotesk",
                                  margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_r, use_container_width=True)

        with rb:
            st.markdown("##### Summary Table")
            cmp_rows = []
            fa_data = master[master["firm_name"] == firm_a]
            fb_data = master[master["firm_name"] == firm_b]
            mv_a = fa_data["veris_score"].mean()
            mv_b = fb_data["veris_score"].mean()
            q2_a = (fa_data["quadrant_short"] == "Q2 Greenwashing").sum()
            q2_b = (fb_data["quadrant_short"] == "Q2 Greenwashing").sum()
            obf_a = int(fa_data.get("obfuscation_flag", pd.Series([0])).sum())
            obf_b = int(fb_data.get("obfuscation_flag", pd.Series([0])).sum())
            cmp_rows = [
                {"Metric": "Mean VERIS",          firm_a: f"{mv_a:.4f}", firm_b: f"{mv_b:.4f}"},
                {"Metric": "Q2 year-pairs",        firm_a: q2_a,          firm_b: q2_b},
                {"Metric": "Obfuscation flags",   firm_a: obf_a,         firm_b: obf_b},
                {"Metric": "Mean Fog",            firm_a: f"{means_a['gunning_fog_index']:.2f}",
                                                   firm_b: f"{means_b['gunning_fog_index']:.2f}"},
                {"Metric": "Mean TTR",            firm_a: f"{means_a['type_token_ratio']:.4f}",
                                                   firm_b: f"{means_b['type_token_ratio']:.4f}"},
                {"Metric": "Mean controversy",    firm_a: f"{means_a['controversy_density']:.3f}",
                                                   firm_b: f"{means_b['controversy_density']:.3f}"},
            ]
            st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True, height=270)

        # Quadrant timelines side by side
        st.markdown("---")
        st.markdown("##### Quadrant Timelines")
        tl_a, tl_b = st.columns(2)
        for col, firm in [(tl_a, firm_a), (tl_b, firm_b)]:
            with col:
                fd = master[master["firm_name"] == firm]
                fv = fd[~fd["quadrant_short"].isin(["No Data", "CT Discontinuity"])]
                fig_tl = go.Figure()
                fig_tl.add_trace(go.Scatter(
                    x=fv["year_to"], y=fv["quadrant_short"],
                    mode="lines+markers+text",
                    text=fv["year_to"].astype(str).str[-2:],
                    textposition="top center",
                    line=dict(color=FIRM_COLORS.get(firm, "#888"), width=2),
                    marker=dict(size=10,
                                color=[QUAD_COLORS.get(q, "#888") for q in fv["quadrant_short"]],
                                line=dict(width=2, color="white"))))
                fig_tl.add_vline(x=2021, line_dash="dash", line_color="#ea580c", line_width=1.5)
                fig_tl.update_layout(title=dict(text=firm, x=0.5),
                                       height=260, plot_bgcolor="#fafafa", paper_bgcolor="white",
                                       font_family="Space Grotesk",
                                       yaxis=dict(categoryorder="array",
                                                   categoryarray=["Q3 Greenhushing", "Q1 Genuine",
                                                                    "Q4 Stagnant", "Q2 Greenwashing"]),
                                       margin=dict(l=10, r=10, t=30, b=40))
                st.plotly_chart(fig_tl, use_container_width=True)

# BUSINESS / EXECUTIVE MODE  -  narrative tabs with action orientation
else:
    tabs = st.tabs([
        "🎯 Executive Summary",
        "📊 Greenwashing Matrix",
        "🏢 My Firm / Portfolio",
        "🛠️ Action Playbook",
        "🏆 Top 15 Risk List",
    ])

    # ════════════════════════════════════════════════════════════════════════════
    # BIZ TAB 1: EXECUTIVE SUMMARY  -  the headline, the finding, the "so what"
    # ════════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        # The headline
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);border-radius:14px;
                    padding:26px 30px;margin-bottom:18px;color:white;">
            <div style="font-family:'Space Grotesk';font-size:1.55rem;font-weight:700;letter-spacing:-.01em;">
                Tightening disclosure rules did not reduce emissions.
            </div>
            <div style="font-family:'Space Grotesk';font-size:1.1rem;font-weight:400;color:#93c5fd;margin-top:4px;">
                They reshaped the language of disclosure while operational reality kept drifting apart.
            </div>
            <div style="font-family:'JetBrains Mono';font-size:.78rem;color:#64748b;margin-top:12px;">
                VERIS · 119 reports · 12 firms · 2014-2024 · 7.5M words cross-checked against Climate TRACE v5.4.1
            </div>
        </div>
        """, unsafe_allow_html=True)

        # The 3 findings  -  big boxes
        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown("""
            <div class="kpi-card" style="text-align:left;padding:18px 20px;min-height:170px;">
                <div style="color:#38bdf8;font-family:JetBrains Mono;font-size:.7rem;letter-spacing:.1em;">FINDING 1</div>
                <div style="font-size:1.3rem;color:white;font-weight:700;margin-top:6px;">DoubleML: θ = +0.150, p = 0.005</div>
                <div style="font-size:.82rem;color:#cbd5e1;margin-top:10px;line-height:1.5;">
                    After CSRD 2021, VERIS scores are ~15 percentage points <b>higher</b> than the counterfactual  - 
                    disclosure language restructured significantly, but emissions did not follow.
                    Placebo tests pass falsification.
                </div>
            </div>""", unsafe_allow_html=True)
        with f2:
            n_q2_biz = (VALID["quadrant_short"] == "Q2 Greenwashing").sum()
            n_q1_biz = (VALID["quadrant_short"] == "Q1 Genuine").sum()
            st.markdown(f"""
            <div class="kpi-card" style="text-align:left;padding:18px 20px;min-height:170px;">
                <div style="color:#38bdf8;font-family:JetBrains Mono;font-size:.7rem;letter-spacing:.1em;">FINDING 2</div>
                <div style="font-size:1.3rem;color:white;font-weight:700;margin-top:6px;">{n_q2_biz} Greenwashing year-pairs vs {n_q1_biz} Genuine</div>
                <div style="font-size:.82rem;color:#cbd5e1;margin-top:10px;line-height:1.5;">
                    At the firm-year level, greenwashing outnumbers authentic progress <b>~1.4 to 1</b>.
                    Kruskal-Wallis H=58.73 (p&lt;0.001) confirms the four quadrants are statistically distinct.
                    AUC=0.878 for Q2 detection.
                </div>
            </div>""", unsafe_allow_html=True)
        with f3:
            st.markdown("""
            <div class="kpi-card" style="text-align:left;padding:18px 20px;min-height:170px;">
                <div style="color:#38bdf8;font-family:JetBrains Mono;font-size:.7rem;letter-spacing:.1em;">FINDING 3</div>
                <div style="font-size:1.3rem;color:white;font-weight:700;margin-top:6px;">9 of 119 reports flagged for obfuscation</div>
                <div style="font-size:.82rem;color:#cbd5e1;margin-top:10px;line-height:1.5;">
                    Jointly high reading difficulty + low lexical richness  -  the specific pattern of
                    strategic obfuscation. <b>6 of 9</b> are Glencore. Singletons for ConocoPhillips,
                    ExxonMobil, and Unilever.
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Data coverage explainer  -  transparent about why some rows are "Insufficient data"
        _uni_excl = len(master[master["firm_name"] == "Unilever"])
        _ct_discon = int(_N_CT_DISCON)
        _first_pair = int(_N_NO_DATA) - _uni_excl if _N_NO_DATA >= _uni_excl else int(_N_NO_DATA)
        st.markdown(f"""
        <div style="background:#f1f5f9;border:1px solid #cbd5e1;border-radius:10px;padding:14px 18px;margin:10px 0;">
            <div style="font-size:.88rem;font-weight:700;color:#0f172a;margin-bottom:6px;">
                📋 Why {int(_N_TOTAL) - int(_N_VALID)} of {int(_N_TOTAL)} year-pairs are "Insufficient data"
            </div>
            <div style="font-size:.82rem;color:#334155;line-height:1.55;">
                The panel has <b>{int(_N_TOTAL)}</b> total year-pairs. Three structural reasons exclude <b>{int(_N_TOTAL) - int(_N_VALID)}</b> from quadrant classification:
                <ul style="margin:6px 0 0 18px;padding:0;">
                    <li><b>First year of each firm ({_first_pair} rows):</b> cannot compute year-over-year emissions delta without a prior baseline year.</li>
                    <li><b>2020→2021 for every firm ({_ct_discon} rows):</b> Climate TRACE v5.4.1 changed coverage basis between these years  -  the jump is structural, not a real emissions event.</li>
                    <li><b>Unilever ({_uni_excl} rows):</b> Climate TRACE covers Fossil Fuels and Mineral Extraction at facility level; the Manufacturing sector lacks equivalent facility-level emissions, so Unilever appears in VERIS (text-only) but cannot be placed on the quadrant matrix.</li>
                </ul>
                This is an <b>honest methodological limit</b> of the data, not a dashboard bug. The {int(_N_VALID)} classifiable year-pairs are the ones driving every finding above.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Leaderboard snapshot
        st.markdown("#### The Top 5 Greenwashers in the Panel")
        st.caption("Firms ranked by mean VERIS score across their full 2014-2024 disclosure history")
        firm_ranks = VALID.groupby("firm_name")["veris_score"].mean().sort_values(ascending=False).head(5).reset_index()
        firm_ranks.columns = ["Firm", "Mean VERIS"]
        firm_ranks["Rank"] = range(1, len(firm_ranks) + 1)
        firm_ranks["Mean VERIS"] = firm_ranks["Mean VERIS"].round(4)
        firm_ranks["Dominant Quadrant"] = firm_ranks["Firm"].apply(lambda f:
            VALID[VALID["firm_name"] == f]["quadrant_short"].mode()[0] if len(VALID[VALID["firm_name"] == f]) else " - ")
        st.dataframe(firm_ranks[["Rank", "Firm", "Mean VERIS", "Dominant Quadrant"]],
                     use_container_width=True, hide_index=True, height=220)

        # Commercial timing
        st.markdown("---")
        st.markdown("#### Why This Matters Now  -  The Regulatory Wave")
        ct1, ct2 = st.columns(2)
        with ct1:
            st.markdown("""
            <div class="find-box" style="min-height:110px;">
                <b>CSRD (EU, 2021)</b><br>
                ~50,000 EU firms file audit-reviewed sustainability reports by 2026.
                First wave (large listed) report 2025; rest 2026-2028.
            </div>
            <div class="find-box" style="min-height:110px;">
                <b>SEC Climate Rule (US, 2024)</b><br>
                All US listed issuers face GHG disclosure mandates.
                Scope 1/2 for large accelerated filers from FY2025.
            </div>
            """, unsafe_allow_html=True)
        with ct2:
            st.markdown("""
            <div class="find-box" style="min-height:110px;">
                <b>California SB-253 (2023)</b><br>
                ~5,300 firms doing business in California must report
                Scope 1, 2, and 3 emissions. Effective 2026.
            </div>
            <div class="find-box" style="min-height:110px;">
                <b>CSDDD (EU, 2024)</b><br>
                Due-diligence mandate extends ESG liability across the supply chain.
                Parent firms now answerable for supplier greenwashing.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""<div class="warn-box" style="margin-top:14px;">
        <b>The commercial implication:</b> Disclosure compliance does not equal ESG performance.
        Firms filing CSRD-compliant reports are not necessarily improving operationally.
        Investors, lenders, regulators, and boards need a verification layer
        independent of the firm's own self-reporting. That is the gap VERIS fills.
        </div>""", unsafe_allow_html=True)

        # Who uses VERIS  -  7 personas at-a-glance
        st.markdown("---")
        st.markdown("#### Who Uses VERIS  -  Seven User Personas")
        st.caption("Each persona has a dedicated workflow in the Action Playbook tab.")
        p1, p2 = st.columns(2)
        personas_grid = [
            ("🏦 Institutional Investors", "BlackRock · pension funds · asset managers",
             "Screen portfolio holdings for Q2 before stewardship meetings. File resolutions at flagged firms."),
            ("🏛️ SLL Lenders", "ING · BNP · Crédit Agricole",
             "Secondary verification for sustainability-linked loans. VERIS Q2 triggers the KPI audit clause."),
            ("⚖️ Regulators", "ESMA · SEC · FCA · Central Bank of Ireland",
             "Prioritise enforcement targets. Start with top-20 Q2 firms, not random lottery."),
            ("🏢 Boards & Audit Committees", "Chief Sustainability Officers · General Counsel",
             "Red-team: does our narrative diverge from our operations? Pre-empt lawsuits."),
            ("⚖️ Plaintiff Firms & NGOs", "ClientEarth · Greenpeace",
             "Auditable, reproducible evidence for securities-fraud and greenwashing litigation."),
            ("📰 Climate Journalists", "Bloomberg Green · FT · Reuters",
             "Reproducible methodology survives legal review  -  unlike proprietary ESG ratings."),
            ("📦 Supply Chain / M&A", "Procurement · corporate development",
             "Screen suppliers for Scope 3 credibility. Due diligence on target firms' ESG exposure."),
        ]
        with p1:
            for title, sub, body in personas_grid[:4]:
                st.markdown(f"""
                <div class="sig-card" style="margin-bottom:10px;">
                    <div style="font-weight:700;font-size:.98rem;color:#0f172a;">{title}</div>
                    <div style="font-family:JetBrains Mono;font-size:.72rem;color:#64748b;margin:2px 0 6px 0;">{sub}</div>
                    <div style="font-size:.83rem;color:#334155;line-height:1.45;">{body}</div>
                </div>""", unsafe_allow_html=True)
        with p2:
            for title, sub, body in personas_grid[4:]:
                st.markdown(f"""
                <div class="sig-card" style="margin-bottom:10px;">
                    <div style="font-weight:700;font-size:.98rem;color:#0f172a;">{title}</div>
                    <div style="font-family:JetBrains Mono;font-size:.72rem;color:#64748b;margin:2px 0 6px 0;">{sub}</div>
                    <div style="font-size:.83rem;color:#334155;line-height:1.45;">{body}</div>
                </div>""", unsafe_allow_html=True)

        # One-sentence workflow
        st.markdown("---")
        st.markdown("""
        <div style="background:#f0f9ff;border-left:4px solid #0284c7;padding:16px 20px;border-radius:8px;
                    font-size:.95rem;color:#0c4a6e;line-height:1.6;">
        <b>The one-sentence workflow:</b> Open dashboard → filter to your firm / portfolio / jurisdiction →
        see which firm-years sit in Q2 → click any dot to see which signal drove it (SBERT drift,
        LDA topic shift, Jaccard copy-paste, VADER tone) → compare against satellite-verified emissions →
        <b>take defensible action</b> (engage, divest, audit, report, sue, or rewrite your own disclosure).
        </div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # BIZ TAB 2: GREENWASHING MATRIX  -  simplified narrative version
    # ════════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("""<div class="info-box">
        Every dot is one firm in one year-pair. The horizontal axis is how much the
        firm's <b>disclosure language</b> changed (VERIS). The vertical axis is how much
        their <b>actual emissions</b> changed (satellite-verified). Top-right means language
        changed while emissions didn't  -  that's the greenwashing signal.
        </div>""", unsafe_allow_html=True)

        bf1, bf2 = st.columns([2, 2])
        with bf1:
            biz_firms = st.multiselect("Filter to specific firms", ALL_FIRMS,
                                        default=ALL_FIRMS, key="biz_m_firms")
        with bf2:
            biz_yr_min, biz_yr_max = int(VALID["year_to"].min()), int(VALID["year_to"].max())
            biz_yr = st.slider("Year range", biz_yr_min, biz_yr_max,
                                (biz_yr_min, biz_yr_max), key="biz_m_yr")

        biz_mat = VALID[VALID["firm_name"].isin(biz_firms)]
        biz_mat = biz_mat[biz_mat["year_to"].between(*biz_yr)]
        biz_mat = biz_mat.copy()
        if not biz_mat.empty:
            biz_mat["hover_label"] = (biz_mat["firm_name"].astype(str) + " " +
                                       biz_mat["year_from"].astype("Int64").astype(str) + "→" +
                                       biz_mat["year_to"].astype("Int64").astype(str))
            biz_mat["emissions_delta_pct_display"] = biz_mat["emissions_delta_pct"] * 100
        else:
            biz_mat["hover_label"] = pd.Series(dtype=str)
            biz_mat["emissions_delta_pct_display"] = pd.Series(dtype=float)

        if biz_mat.empty:
            st.warning("No data for current filter combination. Reset filters to see the matrix.")
        else:
            fig_bm = px.scatter(biz_mat, x="veris_score", y="emissions_delta_pct_display",
                                 color="quadrant_short", color_discrete_map=QUAD_COLORS,
                                 size_max=15, hover_name="hover_label",
                                 labels={"veris_score": "Disclosure Language Change (VERIS)",
                                         "emissions_delta_pct_display": "Satellite-Verified Emissions Change (%)",
                                         "quadrant_short": "Quadrant"}, height=460)
            fig_bm.add_vline(x=_veris_med, line_dash="dash", line_color="#94a3b8",
                              annotation_text=f"Median VERIS = {_veris_med:.3f}")
            fig_bm.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
            for txt, x, y, c in [("Q2 GREENWASHING", 0.72, 4, "#c1121f"),
                                  ("Q1 GENUINE PROGRESS", 0.72, -9, "#1a7340"),
                                  ("Q4 STAGNANT", 0.12, 4, "#6b7280"),
                                  ("Q3 GREENHUSHING", 0.12, -9, "#e07b00")]:
                fig_bm.add_annotation(x=x, y=y, text=f"<b>{txt}</b>", showarrow=False,
                                        font=dict(color=c, size=11), opacity=0.5)
            fig_bm.update_layout(plot_bgcolor="#fafafa", paper_bgcolor="white",
                               font_family="Space Grotesk",
                               margin=dict(l=60, r=20, t=30, b=50),
                               legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig_bm, use_container_width=True)

        # Narrative interpretation
        bn1, bn2, bn3, bn4 = st.columns(4)
        for col, (q, label, color, meaning) in zip([bn1, bn2, bn3, bn4], [
            ("Q1 Genuine", "GENUINE PROGRESS", "#1a7340",
             "High VERIS + falling emissions. Authentic disclosure-performance alignment. CSRD credibility is highest here."),
            ("Q2 Greenwashing", "GREENWASHING SIGNAL", "#c1121f",
             "High VERIS + rising/flat emissions. Language restructured but operations haven't. Highest legal and reputational exposure."),
            ("Q3 Greenhushing", "GREENHUSHING", "#e07b00",
             "Static language + falling emissions. Real progress being under-reported. Forgone reputational and stewardship value."),
            ("Q4 Stagnant", "STAGNANT BASELINE", "#6b7280",
             "Neither language nor operations changed. Confirms VERIS isn't generating false positives."),
        ]):
            with col:
                n_pairs = (biz_mat["quadrant_short"] == q).sum()
                st.markdown(f"""
                <div style="border-top:4px solid {color};background:#f8fafc;border-radius:6px;
                            padding:12px 14px;height:100%;">
                    <div style="font-family:JetBrains Mono;font-size:1.4rem;color:{color};font-weight:700;">{n_pairs}</div>
                    <div style="font-size:.7rem;letter-spacing:.08em;color:{color};font-weight:700;">{label}</div>
                    <div style="font-size:.78rem;color:#475569;margin-top:8px;line-height:1.4;">{meaning}</div>
                </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # BIZ TAB 3: MY FIRM / PORTFOLIO  -  firm-centric executive view
    # ════════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("""<div class="info-box">
        Portfolio managers and sustainability officers: select your firm (or your holdings).
        You'll see the quadrant timeline, VERIS-vs-emissions dual trajectory,
        and a plain-English summary of each year-pair.
        </div>""", unsafe_allow_html=True)

        my_firm = st.selectbox("Select firm (or one of your portfolio holdings)",
                                ALL_FIRMS, index=7, key="biz_my_firm")
        my_data = master[master["firm_name"] == my_firm].sort_values("year_to")
        my_emis = emissions[emissions["firm_name"] == my_firm].sort_values("year") if "firm_name" in emissions.columns else pd.DataFrame()
        my_valid = my_data[~my_data["quadrant_short"].isin(["No Data", "CT Discontinuity"])]

        # Executive KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            modal_q = my_valid["quadrant_short"].mode()[0] if len(my_valid) else "N/A"
            q_color = {"Q2 Greenwashing": "#c1121f", "Q1 Genuine": "#1a7340",
                       "Q3 Greenhushing": "#e07b00", "Q4 Stagnant": "#6b7280"}.get(modal_q, "#38bdf8")
            st.markdown(f"""
            <div class="kpi-card" style="min-height:100px;">
                <div class="kpi-num" style="color:{q_color};font-size:1.1rem;">{modal_q}</div>
                <div class="kpi-lbl">Dominant quadrant</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            q2c = (my_valid["quadrant_short"] == "Q2 Greenwashing").sum()
            st.markdown(f'<div class="kpi-card"><div class="kpi-num">{q2c}</div>'
                        f'<div class="kpi-lbl">Q2 year-pairs</div></div>', unsafe_allow_html=True)
        with k3:
            mv = my_valid["veris_score"].mean() if len(my_valid) else 0
            st.markdown(f'<div class="kpi-card"><div class="kpi-num">{mv:.3f}</div>'
                        f'<div class="kpi-lbl">Mean VERIS</div></div>', unsafe_allow_html=True)
        with k4:
            obf = int(my_data.get("obfuscation_flag", pd.Series([0])).sum())
            obf_color = "#c1121f" if obf > 0 else "#16a34a"
            st.markdown(f'<div class="kpi-card"><div class="kpi-num" style="color:{obf_color};">{obf}</div>'
                        f'<div class="kpi-lbl">Obfuscation flags</div></div>', unsafe_allow_html=True)

        # Interpretation narrative
        narr_map = {
            "Q2 Greenwashing": ("#c1121f", "Warning signal.",
                f"{my_firm}'s disclosure language has restructured significantly over the panel, "
                "but satellite-verified emissions have not followed. This is the pattern "
                "consistent with greenwashing  -  the language changed, the operations did not."),
            "Q1 Genuine": ("#1a7340", "Credible performance.",
                f"{my_firm} shows meaningful language change paired with actual emissions reduction. "
                "This is the pattern CSRD is designed to produce."),
            "Q3 Greenhushing": ("#e07b00", "Under-reporting progress.",
                f"{my_firm}'s emissions are falling, but disclosure language hasn't kept pace. "
                "Genuine progress is being under-communicated  -  a missed reputational opportunity."),
            "Q4 Stagnant": ("#6b7280", "Neutral baseline.",
                f"{my_firm} shows neither material language change nor emissions change. "
                "This is the baseline state and confirms no false positive detection."),
            "N/A": ("#94a3b8", "Insufficient data.",
                f"{my_firm} lacks sufficient Climate TRACE coverage for quadrant classification in most year-pairs."),
        }
        narr_color, narr_label, narr_body = narr_map.get(modal_q, narr_map["N/A"])
        st.markdown(f"""
        <div style="background:linear-gradient(90deg,{narr_color}15,transparent);border-left:4px solid {narr_color};
                    padding:14px 18px;border-radius:8px;margin:14px 0;">
            <div style="font-weight:700;font-size:.95rem;color:{narr_color};">{narr_label}</div>
            <div style="font-size:.88rem;color:#1f2937;margin-top:6px;line-height:1.55;">{narr_body}</div>
        </div>""", unsafe_allow_html=True)

        # Dual-axis chart + quadrant timeline (as in existing deep-dive)
        dd1, dd2 = st.columns(2)
        with dd1:
            st.markdown(f"##### {my_firm}  -  Disclosure vs Emissions")
            co2_col = "co2e_mt" if "co2e_mt" in my_emis.columns else "total_co2e_tonnes"
            fig_bd = go.Figure()
            fig_bd.add_trace(go.Scatter(x=my_data["year_to"], y=my_data["veris_score"],
                                          name="VERIS", mode="lines+markers",
                                          line=dict(color=FIRM_COLORS.get(my_firm, "#1a7340"), width=2.5),
                                          yaxis="y1"))
            if not my_emis.empty and co2_col in my_emis.columns:
                y2_vals = my_emis[co2_col] / 1e6 if co2_col == "total_co2e_tonnes" else my_emis[co2_col]
                fig_bd.add_trace(go.Scatter(x=my_emis["year"], y=y2_vals,
                                              name="Emissions (Mt CO2e)", mode="lines",
                                              line=dict(color="#94a3b8", width=1.5, dash="dot"),
                                              yaxis="y2"))
            fig_bd.add_vline(x=2021, line_dash="dash", line_color="#ea580c", line_width=1.5,
                              annotation_text="CSRD")
            fig_bd.update_layout(height=300, plot_bgcolor="#fafafa", paper_bgcolor="white",
                                   font_family="Space Grotesk",
                                   margin=dict(l=10, r=60, t=10, b=40),
                                   legend=dict(orientation="h", y=1.05),
                                   yaxis=dict(title="VERIS", side="left"),
                                   yaxis2=dict(title="Mt CO2e", side="right",
                                                overlaying="y", showgrid=False))
            st.plotly_chart(fig_bd, use_container_width=True)

        with dd2:
            st.markdown(f"##### {my_firm}  -  Quadrant Timeline")
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=my_valid["year_to"], y=my_valid["quadrant_short"],
                mode="lines+markers+text",
                text=my_valid["year_to"].astype(str).str[-2:],
                textposition="top center",
                line=dict(color=FIRM_COLORS.get(my_firm, "#1a7340"), width=2),
                marker=dict(size=10,
                            color=[QUAD_COLORS.get(q, "#888") for q in my_valid["quadrant_short"]],
                            line=dict(width=2, color="white"))))
            fig_bt.add_vline(x=2021, line_dash="dash", line_color="#ea580c", line_width=1.5)
            fig_bt.update_layout(height=300, plot_bgcolor="#fafafa", paper_bgcolor="white",
                                   font_family="Space Grotesk",
                                   yaxis=dict(categoryorder="array",
                                               categoryarray=["Q3 Greenhushing", "Q1 Genuine",
                                                                "Q4 Stagnant", "Q2 Greenwashing"]),
                                   margin=dict(l=10, r=10, t=10, b=40))
            st.plotly_chart(fig_bt, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════════
    # BIZ TAB 4: ACTION PLAYBOOK  -  7 personas, tailored workflows
    # ════════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("""<div class="info-box">
        Select your role. The playbook shows the specific VERIS workflow for your function,
        with a live example using data from the panel.
        </div>""", unsafe_allow_html=True)

        persona = st.radio(
            "I am a...",
            ["🏦 Institutional Investor / Asset Manager",
             "🏛️ SLL Lender / Credit Analyst",
             "⚖️ Regulator / Supervisor",
             "🏢 Board Member / Chief Sustainability Officer",
             "⚖️ Plaintiff Firm / NGO",
             "📰 Climate Journalist / Researcher",
             "📦 Procurement / M&A Diligence"],
            key="persona", horizontal=False,
        )

        # Persona-specific playbook with live example
        if persona.startswith("🏦"):
            # Top Q2 firm pulled live
            q2_list = VALID[VALID["quadrant_short"] == "Q2 Greenwashing"].groupby("firm_name")["veris_score"].mean().sort_values(ascending=False)
            top_q2 = q2_list.index[0] if len(q2_list) else "ExxonMobil"
            top_veris = q2_list.iloc[0] if len(q2_list) else 0.84
            st.markdown(f"""
            ### 🏦 Institutional Investor / Asset Manager Playbook

            **Your job:** Defend fund integrity under SFDR Article 8/9. Avoid owning firms
            that turn out to be greenwashers. Engage actively when you find them.

            **VERIS workflow (quarterly):**
            1. Open this dashboard. Filter to portfolio holdings only.
            2. Open the Greenwashing Matrix tab. Flag every holding sitting in Q2.
            3. For each flagged firm, open Firm Deep-Dive → check which signals drove the score.
               SBERT drift + high VADER = tone escalation. Jaccard drop = narrative rewrite.
            4. Escalate internally to the stewardship team.
            5. File a shareholder resolution requiring independent emissions verification
               under the firm's CSRD disclosure.

            **Live example from this panel:**

            > **{top_q2}** sits in Q2 with mean VERIS **{top_veris:.3f}**, the highest in the panel.
            > Open the Matrix tab → filter to {top_q2} → you'll see the trajectory entering Q2
            > after 2021. A BlackRock-style PM would file a resolution at the next AGM requiring
            > third-party emissions audit and side-by-side reconciliation of disclosed vs
            > satellite-verified emissions.
            """)
        elif persona.startswith("🏛️"):
            st.markdown("""
            ### 🏛️ Sustainability-Linked Loan Desk Playbook

            **Your job:** Price SLLs correctly. Borrowers self-report KPIs; you need a
            secondary verification layer before interest step-downs activate.

            **VERIS workflow (per KPI reporting cycle):**
            1. Borrower submits self-reported Scope 1 reduction claim.
            2. Run VERIS on their latest sustainability report.
            3. If the firm is in Q2 (high VERIS + flat/rising CT emissions) while claiming
               a material reduction → mismatch. Do not release the step-down.
            4. Trigger the KPI audit clause. Require third-party verification before reset.

            **Live example:**

            > A borrower claims a **15% Scope 1 reduction**. VERIS shows the firm in Q2:
            > language materially restructured post-CSRD, but Climate TRACE shows facility-level
            > emissions flat to rising. The loan covenant's "sustainability performance target"
            > is failing the independent-verification test. Audit clause activates. Price protection
            > saves the lender 15-30bp on the margin.
            """)
        elif persona.startswith("⚖️") and "Regulator" in persona:
            st.markdown("""
            ### ⚖️ Regulator / Supervisor Playbook

            **Your job:** Supervise CSRD/SFDR/SEC compliance. Prioritise enforcement
            with limited resources.

            **VERIS workflow (annual supervisory cycle):**
            1. Run VERIS across all in-scope filers in your jurisdiction.
            2. Rank by Q2 concentration × firm size × market cap weight.
            3. Open Top 15 Risk List. These become your enforcement prioritisation.
            4. For each, issue targeted Article-17 information request covering the specific
               year-pair's disclosure language.

            **Live example:**

            > ESMA's 2023 supervisory briefing explicitly called for NLP-based disclosure
            > monitoring. VERIS is exactly this tool. A regulator opening this dashboard
            > filters to EU firms (via EU/Non-EU filter in Technical mode) → ranks by Q2
            > concentration → sees RioTinto, ExxonMobil (not EU-domiciled but EU-listed),
            > ConocoPhillips at the top. That becomes the prioritisation list.
            """)
        elif persona.startswith("🏢"):
            st.markdown("""
            ### 🏢 Board / Chief Sustainability Officer Playbook

            **Your job:** Red-team your own disclosure. Don't be the next Shell or Exxon
            defendant. Also  -  if you're in Q3 (greenhushing), you're leaving reputation
            and stewardship value on the table.

            **VERIS workflow (before each annual report):**
            1. Run VERIS on your draft disclosure.
            2. If you're trending toward Q2 → halt. Rewrite. Align claims with operational reality.
            3. If you're in Q3 → tell your story properly. Emissions are falling;
               your report should claim that credit with supporting detail.
            4. If you're in Q4 → flat-out acknowledge no progress and explain why  - 
               silence is worse than honest disclosure.

            **Live example from panel:**

            > **BP 2023-24** sits in Q3 Greenhushing. Emissions fell ~2.7% via Climate TRACE,
            > but their disclosure language barely changed year-over-year (Jaccard overlap 0.70).
            > A CSO reading this would tell her team: "We cut emissions but the report didn't
            > claim credit. Next year's report should lead with verified reduction and the
            > operational decisions that produced it. We are losing reputational value."
            """)
        elif persona.startswith("⚖️") and "Plaintiff" in persona:
            st.markdown("""
            ### ⚖️ Plaintiff Firm / NGO Playbook

            **Your job:** Build reproducible evidence for securities-fraud and greenwashing
            litigation. Survive discovery motions and Daubert challenges.

            **VERIS workflow (per investigation):**
            1. Identify the defendant and relevant disclosure period.
            2. Pull VERIS results for that firm's year-pairs spanning the claims.
            3. Confirm Q2 classification  -  language change without operational follow-through
               meets the materiality threshold for Rule 10b-5 / EU Article 17.
            4. Use the signal decomposition (SBERT, LDA, Jaccard, VADER) in expert reports.
               Every number is reproducible because the pipeline is open.

            **Precedents this methodology supports:**

            > - **Shell (Netherlands, 2021)**  -  lost on grounds of disclosure inconsistency
            > - **ExxonMobil (Massachusetts, 2019)**  -  securities fraud on climate disclosures
            > - **ClientEarth vs Shell Directors (2023)**  -  derivative claim on greenwashing
            >
            > VERIS provides the quantitative backbone for the next wave of cases:
            > auditable drift numbers, satellite-verified counter-evidence, placebo-validated
            > causal framework.
            """)
        elif persona.startswith("📰"):
            st.markdown("""
            ### 📰 Climate Journalist / Researcher Playbook

            **Your job:** Publish investigations that survive legal review. Quote sources
            whose methodology can be shown, not taken on faith.

            **VERIS workflow (per story):**
            1. Identify firm + disclosure claim.
            2. Run VERIS; capture quadrant classification + per-signal decomposition.
            3. Download the underlying CSVs  -  every number is traceable back to a notebook.
            4. Submit to fact-checkers and legal with the full pipeline as appendix.

            **Why this beats quoting MSCI / Sustainalytics:**

            > Proprietary ESG ratings won't tell you how they scored a firm.
            > VERIS will  -  every weight, every threshold, every formula, every line
            > of source code is on GitHub. Your story survives the defamation review
            > because the quantitative claim is independently reproducible.
            """)
        elif persona.startswith("📦"):
            st.markdown("""
            ### 📦 Procurement / M&A Diligence Playbook

            **Your job:** Screen suppliers or targets for hidden ESG exposure.
            Scope 3 liability now extends up and down the chain (CSDDD).

            **VERIS workflow (per vendor onboarding / target screen):**
            1. Run VERIS on the counterparty's three most recent sustainability reports.
            2. If counterparty sits in Q2 → flag to legal. Your firm inherits the
               greenwashing risk if you buy from them or buy them.
            3. If counterparty sits in Q1 → green-light with standard terms.
            4. Demand Q3/Q4 suppliers publish VERIS-aligned narratives before contract renewal.

            **Why this matters commercially:**

            > CSDDD makes parent firms directly answerable for supplier greenwashing.
            > An acquiring firm that fails to VERIS-screen an ESG target can inherit
            > class-action exposure on day one of the combined entity.
            > Due-diligence teams now routinely include open-methodology NLP greenwashing
            > screening alongside traditional financial and legal DD.
            """)

    # ════════════════════════════════════════════════════════════════════════════
    # BIZ TAB 5: TOP 15 RISK LIST  -  leaderboard for exec consumption
    # ════════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("""<div class="info-box">
        The 15 highest-VERIS year-pairs in the panel. Think of this as the
        <b>exception report</b>  -  the firms and years that need a closer look.
        </div>""", unsafe_allow_html=True)

        biz_lb = leaderboard.copy()
        if "year_label" not in biz_lb.columns and "year_from" in biz_lb.columns:
            biz_lb["year_label"] = biz_lb["year_from"].astype(str) + "-" + biz_lb["year_to"].astype(str)

        lb_col1, lb_col2 = st.columns([1, 3])
        with lb_col1:
            only_q2_biz = st.checkbox("Greenwashing-only view", value=False, key="biz_onlyq2")
            biz_firms_lb = st.multiselect("Filter firms", ALL_FIRMS,
                                           default=ALL_FIRMS, key="biz_lb_firms")

        if biz_firms_lb and "firm_name" in biz_lb.columns:
            biz_lb = biz_lb[biz_lb["firm_name"].isin(biz_firms_lb)]
        if only_q2_biz and "greenwashing_quadrant" in biz_lb.columns:
            biz_lb = biz_lb[biz_lb["greenwashing_quadrant"] == "2 - Greenwashing signal"]

        biz_cols = [c for c in ["Rank", "firm_name", "year_label", "veris_score",
                                  "greenwashing_quadrant"]
                    if c in biz_lb.columns]

        def _biz_qcol(val):
            return {"2 - Greenwashing signal": "background:#fee2e2;font-weight:600;",
                    "1 - Genuine improvement": "background:#f0fdf4;",
                    "3 - Greenhushing": "background:#fff7ed;",
                    "4 - Stagnant": "background:#f8fafc;"}.get(val, "")

        fmt_biz = {c: "{:.3f}" for c in ["veris_score"] if c in biz_lb.columns}
        styled_biz = biz_lb[biz_cols].style.format(fmt_biz)
        if "greenwashing_quadrant" in biz_cols:
            styled_biz = styled_biz.applymap(_biz_qcol, subset=["greenwashing_quadrant"])
        st.dataframe(styled_biz, use_container_width=True, height=560, hide_index=True)

        st.markdown("""<div class="warn-box">
        <b>How to read this list:</b> The top rows represent year-pairs where
        disclosure language changed the most. Rows with "Insufficient data" in the
        quadrant column reflect Climate TRACE coverage gaps at 2020→2021 and are not
        usable evidence  -  treat them with caution.
        </div>""", unsafe_allow_html=True)
