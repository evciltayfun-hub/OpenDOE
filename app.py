"""FactorLab — Design of Experiments Platform."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from modules.design_wizard import render_design_wizard
from modules.analysis_wizard import render_analysis
from modules.optimization import render_optimization
from modules.visualization import render_visualization
from modules.ai_insights import render_ai_insights

APP_NAME = "FactorLab"
APP_TAGLINE = "Design · Analyze · Optimize"
APP_VERSION = "1.0"

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"{APP_NAME} — DOE Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS — Theme: Deep Navy (Option A) ────────────────────────────────
st.markdown("""
<style>
  /* ════════════════════════════════════════════════════
     DEEP NAVY — Professional Dark
     Background: #020817  |  Card: #0a1628
     Accent:     #3b82f6  |  Green: #10b981
     Text:       #f1f5f9  |  Muted: #64748b
  ════════════════════════════════════════════════════ */

  /* ── Hide Streamlit toolbar & footer ──────────────── */
  [data-testid="stToolbar"]        { display: none !important; }
  [data-testid="stDecoration"]     { display: none !important; }
  [data-testid="stStatusWidget"]   { display: none !important; }
  #MainMenu                        { display: none !important; }
  footer                           { display: none !important; }
  header[data-testid="stHeader"]   { display: none !important; }

  /* ── Keep sidebar always open ──────────────────────── */
  [data-testid="stSidebarCollapseButton"] { display: none !important; }
  [data-testid="collapsedControl"]        { display: none !important; }

  /* ── Page background ───────────────────────────────── */
  .stApp { background-color: #020817; color: #f1f5f9; }
  .block-container { background: transparent; }

  /* ── Sidebar ───────────────────────────────────────── */
  [data-testid="stSidebar"] {
    background: #060f1e;
    border-right: 1px solid #1e2d45;
  }
  [data-testid="stSidebar"] * { color: #94a3b8 !important; }
  [data-testid="stSidebar"] hr { border-color: #1e2d45 !important; }

  /* ── Nav items — hide Streamlit radio circles, show clean links ── */
  [data-testid="stSidebar"] .stRadio > label { display: none; }

  /* Hide the actual radio input and the circular indicator div */
  [data-testid="stSidebar"] .stRadio [role="radiogroup"] input[type="radio"] {
    position: absolute; opacity: 0; width: 0; height: 0; pointer-events: none;
  }
  [data-testid="stSidebar"] .stRadio [role="radiogroup"] label > div:first-child {
    display: none !important;
  }

  /* Style each nav option as a clean link row */
  [data-testid="stSidebar"] .stRadio [role="radiogroup"] {
    display: flex; flex-direction: column; gap: 1px;
  }
  [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    background: transparent;
    border-radius: 4px;
    padding: 8px 12px !important;
    margin: 0 !important;
    cursor: pointer;
    transition: background 0.12s;
    font-size: 0.85rem !important;
    border-left: 2px solid transparent !important;
    color: #4b5e78 !important;
    line-height: 1.3 !important;
  }
  [data-testid="stSidebar"] .stRadio [role="radiogroup"] label p {
    font-size: 0.85rem !important;
    color: inherit !important;
    margin: 0 !important;
    padding: 0 !important;
  }
  [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
    background: rgba(59,130,246,0.08) !important;
    color: #93c5fd !important;
    border-left-color: #1e3a6e !important;
  }
  [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked),
  [data-testid="stSidebar"] .stRadio [role="radiogroup"] label[data-checked="true"] {
    background: rgba(59,130,246,0.12) !important;
    border-left-color: #3b82f6 !important;
    color: #93c5fd !important;
  }

  /* ── Streamlit default text override ──────────────── */
  .stApp, .stApp p, .stApp span, .stApp div,
  .stApp label, .stApp li, .stApp td, .stApp th {
    color: #e2e8f0;
  }

  /* ── Metric cards ──────────────────────────────────── */
  [data-testid="metric-container"] {
    background: #0a1628;
    border: 1px solid #1e2d45;
    border-top: 2px solid #3b82f6;
    border-radius: 6px;
    padding: 10px 14px;
    box-shadow: none;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #60a5fa !important;
    font-weight: 700;
    font-size: 1.5rem !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  /* ── Section panels ────────────────────────────────── */
  .fl-panel {
    background: #0a1628;
    border: 1px solid #1e2d45;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 12px;
  }

  /* ── Step cards ────────────────────────────────────── */
  .fl-step-card {
    background: #0a1628;
    border: 1px solid #1e2d45;
    border-radius: 6px;
    padding: 16px 14px 14px 14px;
    text-align: center;
    height: 100%;
    transition: border-color 0.15s;
  }
  .fl-step-card:hover {
    border-color: #3b82f6;
  }
  .fl-step-icon { font-size: 1.6rem; margin-bottom: 6px; }
  .fl-step-num {
    display: inline-block;
    background: #1e3a6e;
    border: 1px solid #3b82f6;
    color: #60a5fa !important;
    border-radius: 3px;
    width: 22px; height: 22px; line-height: 22px;
    font-size: 0.72rem; font-weight: 700; margin-bottom: 8px;
  }
  .fl-step-title { font-weight: 600; font-size: 0.88rem; color: #e2e8f0; margin-bottom: 5px; }
  .fl-step-desc  { font-size: 0.78rem; color: #64748b; line-height: 1.45; }
  .fl-arrow {
    font-size: 1.2rem; color: #1e3a6e;
    display: flex; align-items: center; justify-content: center;
    height: 100%; padding-top: 36px;
  }

  /* ── Hero banner ───────────────────────────────────── */
  .fl-hero {
    background: linear-gradient(135deg, #0a1628 0%, #0f2040 60%, #0d1f3c 100%);
    border-radius: 8px;
    padding: 28px 36px;
    color: white;
    margin-bottom: 20px;
    border: 1px solid #1e3a6e;
    border-left: 3px solid #3b82f6;
  }
  .fl-hero h1 { color: #f1f5f9 !important; font-size: 1.9rem !important; margin: 0; font-weight: 700; }
  .fl-hero p  { color: #60a5fa; margin: 6px 0 0 0; font-size: 0.9rem; letter-spacing: 0.3px; }

  /* ── Feature badge ─────────────────────────────────── */
  .fl-badge {
    display: inline-block;
    background: #0f2040;
    border: 1px solid #1e3a6e;
    border-radius: 3px;
    padding: 2px 10px;
    font-size: 0.75rem;
    color: #60a5fa;
    font-weight: 500;
    margin: 2px 2px 2px 0;
  }

  /* ── Compare table ─────────────────────────────────── */
  .fl-compare td { padding: 6px 12px; font-size: 0.82rem; border-bottom: 1px solid #1e2d45; color: #cbd5e1; }
  .fl-compare th { background: #0f1e35; padding: 7px 12px; font-size: 0.78rem;
                   color: #60a5fa; font-weight: 600; border-bottom: 1px solid #1e3a6e; }

  /* ── Primary button ────────────────────────────────── */
  .stButton > button[kind="primary"] {
    background: #1d4ed8;
    border: 1px solid #3b82f6;
    border-radius: 5px;
    font-weight: 600;
    font-size: 0.85rem;
    color: #f1f5f9 !important;
    transition: all 0.15s;
  }
  .stButton > button[kind="primary"]:hover {
    background: #2563eb;
    border-color: #60a5fa;
  }
  /* Secondary / default button */
  .stButton > button:not([kind="primary"]) {
    border: 1px solid #1e2d45 !important;
    color: #94a3b8 !important;
    background: #0a1628 !important;
    border-radius: 5px;
    font-size: 0.85rem;
    font-weight: 500;
  }
  .stButton > button:not([kind="primary"]):hover {
    background: #0f1e35 !important;
    border-color: #3b82f6 !important;
    color: #60a5fa !important;
  }

  /* ── Tabs ──────────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: #060f1e; border-radius: 5px; padding: 3px;
    border: 1px solid #1e2d45;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 4px; padding: 6px 16px;
    color: #64748b !important; font-weight: 500; font-size: 0.85rem;
  }
  .stTabs [aria-selected="true"] {
    background: #0a1628;
    color: #60a5fa !important;
    font-weight: 600;
  }

  /* ── Expanders ─────────────────────────────────────── */
  [data-testid="stExpander"] {
    border: 1px solid #1e2d45 !important;
    border-radius: 5px !important;
    background: #0a1628;
  }
  [data-testid="stExpander"] summary {
    color: #94a3b8 !important;
    font-size: 0.87rem !important;
  }

  /* ── Data tables ───────────────────────────────────── */
  [data-testid="stDataFrame"] {
    border: 1px solid #1e2d45;
    border-radius: 5px;
  }

  /* ── Progress & sliders ────────────────────────────── */
  [data-testid="stProgress"] > div > div {
    background: #3b82f6 !important;
  }

  /* ── Alerts ────────────────────────────────────────── */
  .stAlert { border-radius: 5px; font-size: 0.85rem; }
  [data-testid="stNotification"][kind="success"],
  div[data-testid="stMarkdownContainer"] .stSuccess {
    background: #052e16 !important;
    border-left: 3px solid #10b981 !important;
    color: #6ee7b7 !important;
  }
  [data-testid="stNotification"][kind="info"],
  div[data-testid="stMarkdownContainer"] .stInfo {
    background: #0f172a !important;
    border-left: 3px solid #3b82f6 !important;
    color: #93c5fd !important;
  }

  /* ── Typography ─────────────────────────────────────── */
  h1 { color: #f1f5f9 !important; font-weight: 700; font-size: 1.6rem !important; }
  h2 { color: #e2e8f0 !important; font-weight: 600; font-size: 1.3rem !important; }
  h3 { color: #cbd5e1 !important; font-weight: 600; font-size: 1.1rem !important; }
  h4 { color: #94a3b8 !important; font-weight: 600; font-size: 0.95rem !important;
       text-transform: uppercase; letter-spacing: 0.5px; }
  hr { border-color: #1e2d45 !important; margin: 14px 0; }
  p  { font-size: 0.88rem; color: #cbd5e1; }

  /* ── Input fields ──────────────────────────────────── */
  [data-testid="stTextInput"] > div > div > input,
  [data-testid="stNumberInput"] > div > div > input {
    background: #060f1e !important;
    border-radius: 4px !important;
    border: 1px solid #1e2d45 !important;
    color: #e2e8f0 !important;
    font-size: 0.87rem !important;
  }
  [data-testid="stTextInput"] > div > div > input:focus,
  [data-testid="stNumberInput"] > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
  }
  [data-testid="stSelectbox"] > div > div {
    background: #060f1e !important;
    border-radius: 4px !important;
    border: 1px solid #1e2d45 !important;
    color: #e2e8f0 !important;
    font-size: 0.87rem !important;
  }
  [data-testid="stTextArea"] textarea {
    background: #060f1e !important;
    border: 1px solid #1e2d45 !important;
    color: #e2e8f0 !important;
    font-size: 0.87rem !important;
    border-radius: 4px !important;
  }
  /* Selectbox / dropdown popover */
  [data-baseweb="select"] * { color: #e2e8f0 !important; }
  [data-baseweb="popover"] { background: #0f1e35 !important; border: 1px solid #1e2d45 !important; }
  [data-baseweb="option"]:hover { background: #1e3a6e !important; }

  /* Labels */
  [data-testid="stWidgetLabel"] p,
  label[data-testid] p { font-size: 0.82rem !important; color: #64748b !important; font-weight: 500; }

  /* ── Divider ───────────────────────────────────────── */
  .fl-divider {
    height: 1px;
    background: linear-gradient(90deg, #3b82f6, transparent);
    margin: 16px 0;
  }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "experiment": {
            "name": "", "objective": "", "description": "",
            "factors": [], "responses": [],
            "design_type_name": None, "design_key": None,
            "design_matrix": None, "natural_matrix": None,
            "response_data": None,
        },
        "models": {},
        "optimization": {},
        "ai_history": [],
        "wizard_step": 1,
        "page": "🏠 Dashboard",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:16px 6px 10px 6px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="background:#0f2040; border:1px solid #1e3a6e;
                        border-radius:5px; width:36px; height:36px;
                        display:flex; align-items:center; justify-content:center;
                        font-size:1.2rem; flex-shrink:0;">🔬</div>
            <div>
                <div style="font-size:1.2rem; font-weight:700; letter-spacing:-0.3px; color:#f1f5f9;">
                    {APP_NAME}
                </div>
                <div style="font-size:0.62rem; color:#3b82f6; letter-spacing:1.5px;
                            text-transform:uppercase; font-weight:500; margin-top:1px;">
                    {APP_TAGLINE}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    pages = [
        "🏠 Dashboard",
        "🔬 Design Wizard",
        "📊 Analysis",
        "🎯 Optimization",
        "📈 Visualization",
        "🤖 AI Insights",
        "📤 Export",
    ]
    page = st.radio("Navigation", pages,
                    index=pages.index(st.session_state.get("page", "🏠 Dashboard")))
    st.session_state.page = page

    # Experiment status panel
    exp = st.session_state.experiment
    if exp.get("name"):
        st.divider()
        n_f = len(exp.get("factors", []))
        n_r = len(exp.get("responses", []))
        has_design = exp.get("design_matrix") is not None
        has_data   = exp.get("response_data") is not None
        has_models = bool(st.session_state.get("models"))
        has_opt    = bool(st.session_state.get("optimization"))

        st.markdown(f"""
        <div style="font-size:0.8rem; color:#94a3b8; line-height:1.5;">
          <b style="color:#e2e8f0; font-size:0.87rem;">{exp['name']}</b><br>
          <span style="color:#3b82f6; font-size:0.78rem;">{exp.get('design_type_name', '—')}</span><br>
          <span style="font-size:0.76rem;">{n_f} factors &nbsp;·&nbsp; {n_r} responses</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:8px;'>", unsafe_allow_html=True)
        steps = [("Design", has_design), ("Data", has_data),
                 ("Models", has_models), ("Optimized", has_opt)]
        for label, done in steps:
            icon  = "●" if done else "○"
            color = "#10b981" if done else "#334155"
            st.markdown(f"<span style='color:{color}; font-size:0.78rem;'>{icon} {label}</span>",
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"<span style='color:#334155; font-size:0.72rem;'>v{APP_VERSION} &nbsp;·&nbsp; Open Source &nbsp;·&nbsp; Free Forever</span>",
                unsafe_allow_html=True)
    if st.button("🗑️ New Experiment", use_container_width=True):
        for k in ["experiment", "models", "optimization", "ai_history", "wizard_step"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()


# ─── Demo Loader ──────────────────────────────────────────────────────────────
def _load_demo():
    from utils.doe_engine import generate_design, build_design_dataframe
    from utils.stats import fit_mlr

    factors = [
        {"name": "Temperature",   "low": 60.0,  "high": 90.0,  "unit": "°C"},
        {"name": "pH",            "low": 4.0,   "high": 8.0,   "unit": ""},
        {"name": "Stirring Speed","low": 200.0, "high": 600.0, "unit": "rpm"},
    ]
    responses = [
        {"name": "Yield",  "unit": "%", "importance": 3},
        {"name": "Purity", "unit": "%", "importance": 3},
    ]
    factor_names  = [f["name"] for f in factors]
    factor_ranges = [(f["low"], f["high"]) for f in factors]

    design = generate_design("ccd", 3, center_points=3, ccd_face="ccf", randomize=False)
    coded_df, natural_df = build_design_dataframe(design, factor_names, factor_ranges)

    # Synthetic response data (deterministic, no noise — clean demo curves)
    T, pH, S = design[:, 0], design[:, 1], design[:, 2]
    yield_vals = (78 + 7*T - 4*pH + 2*S
                  - 5*T**2 - 3*pH**2 - 1.5*S**2
                  + 2*T*pH - 1.5*T*S + pH*S)
    purity_vals = (85 - 3*T + 5*pH + 1.5*S
                   - 4*T**2 - 6*pH**2 - 2*S**2
                   - 2*T*pH + T*S - 2*pH*S)

    resp_df = pd.DataFrame({"Yield": yield_vals, "Purity": purity_vals})

    models = {}
    for rname, y_arr in [("Yield", yield_vals), ("Purity", purity_vals)]:
        models[rname] = fit_mlr(design, y_arr, degree="quadratic",
                                factor_names=factor_names)

    st.session_state.experiment = {
        "name": "Reaction Optimization (Demo)",
        "objective": "Maximize Yield and Purity",
        "description": "3-factor CCD demo — Temperature, pH, Stirring Speed",
        "factors": factors,
        "responses": responses,
        "design_type_name": "Central Composite Design (CCD)",
        "design_key": "ccd",
        "design_matrix": coded_df,
        "natural_matrix": natural_df,
        "response_data": resp_df,
    }
    st.session_state.models = models
    st.session_state.optimization = {}
    st.session_state.wizard_step = 1
    st.session_state.page = "📈 Visualization"


# ─── Dashboard ────────────────────────────────────────────────────────────────
def render_dashboard():
    exp = st.session_state.experiment

    if exp.get("name"):
        _render_experiment_dashboard(exp)
    else:
        _render_welcome()


def _render_welcome():
    # Hero
    st.markdown(f"""
    <div class="fl-hero">
        <h1>🔬 {APP_NAME}</h1>
        <p>{APP_TAGLINE} — Professional Design of Experiments, free and open source.</p>
    </div>
    """, unsafe_allow_html=True)

    # CTA buttons
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        if st.button("🚀 Start New Experiment", type="primary", use_container_width=True):
            st.session_state.page = "🔬 Design Wizard"
            st.rerun()
    with c2:
        if st.button("⚡ Load Demo Experiment", use_container_width=True):
            _load_demo()
            st.rerun()

    st.markdown("---")

    # How It Works — Step Cards
    st.markdown("### How It Works")
    st.markdown("Follow the 6-step workflow to go from raw factors to optimized process conditions.")

    steps = [
        ("🔬", "1", "Design Wizard",
         "Define factors & responses. Choose a design type (CCD, Factorial, BBD…). Download your experiment template."),
        ("🧪", "2", "Run Experiments",
         "Perform the experimental runs in the lab or plant. Fill in measured response values."),
        ("📊", "3", "Model Analysis",
         "Upload data. Fit MLR or PLS models. Review R², Q², ANOVA tables and coefficient plots."),
        ("🎯", "4", "Optimization",
         "Set goals (maximize / minimize / target) for each response. Run multi-objective optimizer."),
        ("📈", "5", "Visualization",
         "Explore 3D response surfaces, contour maps, sweet spot plots and interaction charts."),
        ("🤖", "6", "AI Insights",
         "Claude analyzes your results and delivers a plain-language expert interpretation."),
    ]

    # Row 1
    cols = st.columns([1, 0.18, 1, 0.18, 1])
    for i, col_idx in enumerate([0, 2, 4]):
        s = steps[i]
        with cols[col_idx]:
            st.markdown(f"""
            <div class="fl-step-card">
                <div class="fl-step-num">{s[1]}</div><br>
                <div class="fl-step-icon">{s[0]}</div>
                <div class="fl-step-title">{s[2]}</div>
                <div class="fl-step-desc">{s[3]}</div>
            </div>
            """, unsafe_allow_html=True)
        if col_idx in [0, 2]:
            with cols[col_idx + 1]:
                st.markdown('<div class="fl-arrow">→</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2
    cols2 = st.columns([1, 0.18, 1, 0.18, 1])
    for i, col_idx in enumerate([0, 2, 4]):
        s = steps[i + 3]
        with cols2[col_idx]:
            st.markdown(f"""
            <div class="fl-step-card">
                <div class="fl-step-num">{s[1]}</div><br>
                <div class="fl-step-icon">{s[0]}</div>
                <div class="fl-step-title">{s[2]}</div>
                <div class="fl-step-desc">{s[3]}</div>
            </div>
            """, unsafe_allow_html=True)
        if col_idx in [0, 2]:
            with cols2[col_idx + 1]:
                st.markdown('<div class="fl-arrow">→</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Feature / comparison section
    col_feat, col_cmp = st.columns([3, 2])

    with col_feat:
        st.markdown("### Design Types Supported")
        design_info = [
            ("🔲", "Full Factorial (2^k)", "2–6 factors", "All factor combinations. Complete information."),
            ("📐", "Fractional Factorial", "3–15 factors", "Efficient screening; some interactions aliased."),
            ("🎯", "Plackett-Burman", "2–23 factors", "Maximum screening power, minimum runs."),
            ("⭕", "Central Composite (CCD)", "2–8 factors", "Gold standard for response surface modeling."),
            ("🔷", "Box-Behnken (BBD)", "3–7 factors", "No corner points — ideal for constrained ranges."),
        ]
        for icon, name, factors, desc in design_info:
            st.markdown(f"""
            <div style="display:flex; align-items:flex-start; gap:10px; padding:8px 0;
                        border-bottom:1px solid #1e2d45;">
                <span style="font-size:1.1rem; margin-top:2px; opacity:0.8;">{icon}</span>
                <div>
                    <b style="font-size:0.85rem; color:#e2e8f0;">{name}</b>
                    <span style="color:#3b82f6; font-size:0.78rem; margin-left:8px;">{factors}</span><br>
                    <span style="color:#64748b; font-size:0.78rem;">{desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_cmp:
        st.markdown("### FactorLab vs MODDE")
        st.markdown("""
<table class="fl-compare" style="width:100%; border-collapse:collapse; border-radius:5px; overflow:hidden; border:1px solid #1e2d45; background:#0a1628;">
<tr><th>Feature</th><th>FactorLab</th><th>MODDE</th></tr>
<tr><td>Price</td><td style="color:#10b981;">✓ Free</td><td style="color:#64748b;">€3,000+/yr</td></tr>
<tr><td>Platform</td><td style="color:#10b981;">✓ Web</td><td style="color:#64748b;">Windows only</td></tr>
<tr><td>AI Interpretation</td><td style="color:#10b981;">✓ Claude AI</td><td style="color:#64748b;">—</td></tr>
<tr><td>Open Source</td><td style="color:#10b981;">✓ GitHub</td><td style="color:#64748b;">—</td></tr>
<tr><td>MLR / PLS</td><td style="color:#10b981;">✓</td><td style="color:#60a5fa;">✓</td></tr>
<tr><td>Response Surface</td><td style="color:#10b981;">✓ Interactive</td><td style="color:#60a5fa;">✓ Static</td></tr>
<tr><td>Sweet Spot Plot</td><td style="color:#10b981;">✓ + Zone Box</td><td style="color:#60a5fa;">✓</td></tr>
<tr><td>Collaboration</td><td style="color:#10b981;">✓ Multi-user</td><td style="color:#64748b;">—</td></tr>
</table>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Supported output formats:**")
        for badge in ["Excel (.xlsx)", "CSV", "Markdown (AI Report)", "Plotly Interactive"]:
            st.markdown(f'<span class="fl-badge">{badge}</span>', unsafe_allow_html=True)


def _render_experiment_dashboard(exp):
    has_design = exp.get("design_matrix") is not None
    has_models = bool(st.session_state.get("models"))
    has_opt    = bool(st.session_state.get("optimization"))

    st.markdown(f"""
    <div style="background:#0a1628; border:1px solid #1e2d45; border-left:3px solid #3b82f6;
                border-radius:6px; padding:14px 20px; margin-bottom:16px;">
        <div style="font-size:1.1rem; font-weight:700; color:#f1f5f9;">🔬 {exp['name']}</div>
        <div style="color:#60a5fa; font-size:0.8rem; margin-top:4px;">
            {exp.get('design_type_name','—')} &nbsp;·&nbsp;
            {len(exp.get('factors',[]))} factors &nbsp;·&nbsp;
            {len(exp.get('responses',[]))} responses
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Factors", len(exp.get("factors", [])))
    c2.metric("Responses", len(exp.get("responses", [])))
    runs = len(exp["design_matrix"]) if has_design else "—"
    c3.metric("Experimental Runs", runs)
    c4.metric("Models Fitted", len(st.session_state.get("models", {})))

    # Progress pipeline
    st.markdown("#### Experiment Progress")
    pipeline = [
        ("Design Generated", has_design, "🔬 Design Wizard"),
        ("Data Entered", exp.get("response_data") is not None, "📊 Analysis"),
        ("Models Fitted", has_models, "📊 Analysis"),
        ("Optimized", has_opt, "🎯 Optimization"),
    ]
    pcols = st.columns(4)
    for i, (label, done, target_page) in enumerate(pipeline):
        with pcols[i]:
            if done:
                st.markdown(f"""
                <div style="border:1px solid #134e30; border-top:2px solid #10b981;
                            border-radius:5px; padding:10px 12px; text-align:center;
                            background:#052e16;">
                    <div style="font-size:1.0rem;">●</div>
                    <div style="font-weight:600; font-size:0.8rem; color:#6ee7b7; margin-top:3px;">
                        {label}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="border:1px solid #1e2d45; border-top:2px solid #1e3a6e;
                            border-radius:5px; padding:10px 12px; text-align:center;
                            background:#0a1628;">
                    <div style="font-size:1.0rem; color:#334155;">○</div>
                    <div style="font-weight:500; font-size:0.8rem; color:#475569; margin-top:3px;">
                        {label}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Start →", key=f"go_{i}", use_container_width=True):
                    st.session_state.page = target_page
                    st.rerun()

    if has_models:
        st.divider()
        st.markdown("#### Model Summary")
        rows = []
        for rname, m in st.session_state.models.items():
            r2, q2 = m["r2"], m["q2"]
            rows.append({
                "Response": rname, "Type": m["type"],
                "R²": f"{r2:.3f}", "R²adj": f"{m['r2_adj']:.3f}",
                "Q²": f"{q2:.3f}",
                "RMSE": f"{m['rmse']:.4f}" if not np.isnan(m["rmse"]) else "—",
                "Quality": "✅ Good" if r2 > 0.8 and q2 > 0.5 else ("⚠️ Check" if r2 > 0.5 else "❌ Poor"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if has_opt:
        opt = st.session_state.optimization
        d = opt.get("desirability", 0)
        st.divider()
        st.markdown("#### Optimization Result")
        c1, c2 = st.columns([1, 2])
        with c1:
            d_color = "#059669" if d > 0.7 else ("#f59e0b" if d > 0.4 else "#ef4444")
            d_label = "🟢 Excellent (≥ 0.7)" if d >= 0.7 else ("🟡 Acceptable (≥ 0.4)" if d >= 0.4 else "🔴 Low — review limits")
            st.markdown(f"""
            <div style="text-align:center; padding:18px 14px;
                        background:#0a1628; border-radius:6px;
                        border:1px solid #1e2d45; border-top:2px solid {d_color};">
                <div style="font-size:2.4rem; font-weight:800; color:{d_color}; line-height:1;">{d:.2f}</div>
                <div style="font-size:0.72rem; font-weight:600; color:{d_color}; text-transform:uppercase;
                            letter-spacing:1px; margin-top:5px;">Overall Desirability</div>
                <div style="font-size:0.72rem; color:#64748b; margin-top:6px;">{d_label}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            opt_nat = opt.get("optimal_natural", {})
            for fname, val in opt_nat.items():
                unit = next((f.get("unit","") for f in exp.get("factors",[]) if f["name"]==fname), "")
                st.markdown(f"**{fname}:** &nbsp; `{val:.4f}` {unit}")
            st.markdown("**Predicted responses:**")
            for rname, val in opt.get("predicted_responses", {}).items():
                rconf = next((r for r in exp.get("responses",[]) if r["name"]==rname), {})
                st.markdown(f"&nbsp;&nbsp;{rname}: `{val:.4f}` {rconf.get('unit','')}")


# ─── Export ───────────────────────────────────────────────────────────────────
def render_export():
    st.title("📤 Export Results")
    exp = st.session_state.experiment
    models = st.session_state.get("models", {})
    optimization = st.session_state.get("optimization", {})

    if not exp.get("name"):
        st.warning("No experiment to export.")
        return

    import io
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        wb = writer.book
        header_fmt = wb.add_format({"bold": True, "bg_color": "#1e40af", "font_color": "white"})

        # Sheet 1: Info
        pd.DataFrame({
            "Field": ["Name", "Objective", "Design", "Factors", "Responses", "Runs"],
            "Value": [
                exp.get("name",""), exp.get("objective",""),
                exp.get("design_type_name",""),
                len(exp.get("factors",[])), len(exp.get("responses",[])),
                len(exp.get("design_matrix",[])) if exp.get("design_matrix") is not None else 0,
            ]
        }).to_excel(writer, sheet_name="Experiment Info", index=False)

        # Sheet 2: Design Matrix + responses
        nat = exp.get("natural_matrix")
        if nat is not None:
            resp_data = exp.get("response_data")
            if resp_data is not None:
                nat = nat.copy()
                for rc in [r["name"] for r in exp["responses"]]:
                    if rc in resp_data.columns:
                        nat[rc] = resp_data[rc].values
            nat.to_excel(writer, sheet_name="Design Matrix")

        # Sheet 3: Model Summary
        if models:
            pd.DataFrame([{
                "Response": rname, "Type": m["type"],
                "R²": m["r2"], "R²adj": m["r2_adj"], "Q²": m["q2"], "RMSE": m["rmse"],
            } for rname, m in models.items()]).to_excel(writer, sheet_name="Model Summary", index=False)
            for rname, m in models.items():
                if m["type"] == "MLR":
                    m["coefficients"].to_excel(writer, sheet_name=f"Coeff_{rname[:24]}", index=False)

        # Sheet 4: Optimization
        if optimization:
            rows = []
            for k, v in optimization.get("optimal_natural", {}).items():
                rows.append({"Type": "Factor", "Name": k, "Value": v})
            for k, v in optimization.get("predicted_responses", {}).items():
                rows.append({"Type": "Response (predicted)", "Name": k, "Value": v})
            rows.append({"Type": "Overall", "Name": "Desirability",
                         "Value": optimization.get("desirability","")})
            pd.DataFrame(rows).to_excel(writer, sheet_name="Optimization", index=False)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download Full Results (Excel)", data=buf.getvalue(),
            file_name=f"{exp['name'].replace(' ','_')}_FactorLab_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary", use_container_width=True,
        )
    with col2:
        nat_csv = exp.get("natural_matrix")
        if nat_csv is not None:
            st.download_button("⬇️ Design Matrix (CSV)", data=nat_csv.to_csv(),
                               file_name="design_matrix.csv", mime="text/csv",
                               use_container_width=True)

    st.divider()
    st.markdown("""
**Excel file contains:**
- **Experiment Info** — design type, factors, responses
- **Design Matrix** — all runs with measured responses
- **Model Summary** — R², Q², RMSE for all fitted models
- **Coeff_[Response]** — MLR coefficient table per response
- **Optimization** — optimal settings and predicted responses
""")


# ─── Router ───────────────────────────────────────────────────────────────────
page = st.session_state.get("page", "🏠 Dashboard")

if   page == "🏠 Dashboard":   render_dashboard()
elif page == "🔬 Design Wizard": render_design_wizard()
elif page == "📊 Analysis":    render_analysis()
elif page == "🎯 Optimization":  render_optimization()
elif page == "📈 Visualization": render_visualization()
elif page == "🤖 AI Insights":  render_ai_insights()
elif page == "📤 Export":      render_export()
