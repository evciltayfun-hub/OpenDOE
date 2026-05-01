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

# ─── Custom CSS — AQUIONIS-inspired Dark Theme ───────────────────────────────
st.markdown("""
<style>
  /* ══════════════════════════════════════════════
     FACTORLAB — AQUIONIS-inspired Dark Theme
     Page:    #0b0f14  |  Sidebar: #0d1117
     Card:    #161b22  |  Border:  #21262d
     Accent:  #00c9a7  |  Text:    #e6edf3
  ══════════════════════════════════════════════ */

  /* Hide Streamlit chrome */
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  [data-testid="stStatusWidget"],
  #MainMenu, footer,
  header[data-testid="stHeader"]          { display: none !important; }
  [data-testid="stSidebarCollapseButton"],
  [data-testid="collapsedControl"]        { display: none !important; }

  /* Page */
  .stApp                    { background: #0b0f14; color: #e6edf3; }
  .block-container          { background: transparent; padding-top: 16px !important; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d !important;
    min-width: 220px !important;
  }
  [data-testid="stSidebar"] * { color: #7d8590 !important; }
  [data-testid="stSidebar"] hr { border-color: #21262d !important; margin: 6px 0 !important; }

  /* Sidebar section labels */
  .sb-section {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
    color: #444d56 !important;
    padding: 10px 14px 4px 14px !important;
    display: block !important;
  }

  /* Sidebar nav buttons */
  [data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    border-left: 2px solid transparent !important;
    border-radius: 0 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 7px 14px !important;
    width: 100% !important;
    font-size: 0.84rem !important;
    font-weight: 400 !important;
    color: #7d8590 !important;
    transition: all 0.1s !important;
    box-shadow: none !important;
    margin: 0 !important;
    height: auto !important;
    min-height: 0 !important;
  }
  [data-testid="stSidebar"] .stButton > button:hover {
    background: #161b22 !important;
    color: #e6edf3 !important;
    border-left-color: #30363d !important;
  }
  [data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #1c2128 !important;
    border-left-color: #00c9a7 !important;
    color: #e6edf3 !important;
    font-weight: 500 !important;
  }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-top: 2px solid #00c9a7;
    border-radius: 6px;
    padding: 10px 14px;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #00c9a7 !important;
    font-weight: 700;
    font-size: 1.5rem !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #7d8590 !important;
    font-size: 0.73rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  /* Panels */
  .fl-panel {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 12px;
  }

  /* Workflow step cards */
  .wf-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 18px 14px 16px;
    height: 100%;
    position: relative;
  }
  .wf-card.wf-active  { border-color: #00c9a7; }
  .wf-card.wf-done    { border-color: #238636; }
  .wf-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 30px; height: 30px; border-radius: 50%;
    font-size: 0.82rem; font-weight: 700; margin-bottom: 10px;
  }
  .wf-num.active  { background: #00c9a7; color: #0b0f14 !important; }
  .wf-num.done    { background: #238636; color: #fff !important; }
  .wf-num.locked  { background: #21262d; color: #444d56 !important; }
  .wf-title       { font-size: 0.9rem; font-weight: 600; color: #e6edf3; margin-bottom: 5px; }
  .wf-title.locked{ color: #444d56; }
  .wf-desc        { font-size: 0.77rem; color: #7d8590; line-height: 1.45; }
  .wf-arrow {
    display: flex; align-items: center; justify-content: center;
    color: #30363d; font-size: 1.1rem; padding-top: 30px;
  }
  .wf-action {
    display: inline-block; margin-top: 10px;
    font-size: 0.75rem; color: #00c9a7; font-weight: 500;
  }

  /* Hero */
  .fl-hero {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #00c9a7;
    border-radius: 8px;
    padding: 22px 28px;
    margin-bottom: 18px;
  }
  .fl-hero h1 { color: #e6edf3 !important; font-size: 1.6rem !important; margin: 0; font-weight: 700; }
  .fl-hero p  { color: #7d8590; margin: 5px 0 0 0; font-size: 0.87rem; }

  /* Badge */
  .fl-badge {
    display: inline-block;
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 3px;
    padding: 2px 8px;
    font-size: 0.72rem;
    color: #7d8590;
    font-weight: 500;
    margin: 2px 2px 2px 0;
  }

  /* Compare table */
  .fl-compare td { padding: 6px 12px; font-size: 0.8rem; border-bottom: 1px solid #21262d; color: #7d8590; }
  .fl-compare th { background: #161b22; padding: 7px 12px; font-size: 0.75rem;
                   color: #7d8590; font-weight: 600; border-bottom: 1px solid #21262d;
                   text-transform: uppercase; letter-spacing: 0.5px; }

  /* Buttons (main area) */
  .stButton > button[kind="primary"] {
    background: #00c9a7 !important;
    border: 1px solid #00c9a7 !important;
    border-radius: 5px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    color: #0b0f14 !important;
  }
  .stButton > button[kind="primary"]:hover {
    background: #00b396 !important;
    border-color: #00b396 !important;
  }
  .stButton > button:not([kind="primary"]) {
    border: 1px solid #21262d !important;
    color: #7d8590 !important;
    background: #161b22 !important;
    border-radius: 5px !important;
    font-size: 0.84rem !important;
  }
  .stButton > button:not([kind="primary"]):hover {
    border-color: #00c9a7 !important;
    color: #e6edf3 !important;
  }
  /* Disabled buttons in sidebar should not get main area styling */
  [data-testid="stSidebar"] .stButton > button[kind="primary"] {
    color: #e6edf3 !important;
    background: #1c2128 !important;
    border-left-color: #00c9a7 !important;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 1px; background: #0d1117; border-radius: 5px; padding: 3px;
    border: 1px solid #21262d;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 4px; padding: 6px 16px;
    color: #7d8590 !important; font-weight: 400; font-size: 0.83rem;
  }
  .stTabs [aria-selected="true"] {
    background: #161b22 !important;
    color: #e6edf3 !important;
    font-weight: 600 !important;
  }

  /* Expanders */
  [data-testid="stExpander"] {
    border: 1px solid #21262d !important;
    border-radius: 5px !important;
    background: #161b22 !important;
  }
  [data-testid="stExpander"] summary { color: #7d8590 !important; font-size: 0.84rem !important; }

  /* Data tables */
  [data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 5px; }

  /* Progress */
  [data-testid="stProgress"] > div > div { background: #00c9a7 !important; }

  /* Typography */
  h1 { color: #e6edf3 !important; font-weight: 700; font-size: 1.4rem !important; }
  h2 { color: #e6edf3 !important; font-weight: 600; font-size: 1.15rem !important; }
  h3 { color: #c9d1d9 !important; font-weight: 600; font-size: 1.0rem !important; }
  h4 { color: #7d8590 !important; font-weight: 600; font-size: 0.82rem !important;
       text-transform: uppercase; letter-spacing: 0.5px; }
  hr { border-color: #21262d !important; margin: 12px 0; }
  p  { font-size: 0.86rem; color: #c9d1d9; }

  /* Inputs */
  [data-testid="stTextInput"] > div > div > input,
  [data-testid="stNumberInput"] > div > div > input {
    background: #0d1117 !important;
    border-radius: 4px !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    font-size: 0.86rem !important;
  }
  [data-testid="stTextInput"] > div > div > input:focus,
  [data-testid="stNumberInput"] > div > div > input:focus {
    border-color: #00c9a7 !important;
    box-shadow: 0 0 0 2px rgba(0,201,167,0.12) !important;
  }
  [data-testid="stSelectbox"] > div > div {
    background: #0d1117 !important;
    border-radius: 4px !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    font-size: 0.86rem !important;
  }
  [data-testid="stTextArea"] textarea {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    font-size: 0.86rem !important;
    border-radius: 4px !important;
  }
  [data-baseweb="select"] * { color: #e6edf3 !important; }
  [data-baseweb="popover"]  { background: #161b22 !important; border: 1px solid #21262d !important; }
  [data-baseweb="option"]:hover { background: #1c2128 !important; }

  /* Widget labels */
  [data-testid="stWidgetLabel"] p,
  label[data-testid] p { font-size: 0.8rem !important; color: #7d8590 !important; font-weight: 500; }

  /* Alerts */
  .stAlert { border-radius: 5px; font-size: 0.84rem; }

  /* Page header breadcrumb */
  .fl-breadcrumb {
    font-size: 0.75rem; color: #444d56; text-transform: uppercase;
    letter-spacing: 1px; font-weight: 600; margin-bottom: 6px;
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
    # Logo
    st.markdown(f"""
    <div style="padding:16px 14px 10px 14px; border-bottom:1px solid #21262d;">
      <div style="display:flex; align-items:center; gap:10px;">
        <div style="background:#1c2128; border:1px solid #21262d; border-left:2px solid #00c9a7;
                    border-radius:5px; width:32px; height:32px;
                    display:flex; align-items:center; justify-content:center; font-size:1.1rem;">🔬</div>
        <div>
          <div style="font-size:1.05rem; font-weight:700; color:#e6edf3; letter-spacing:-0.2px;">{APP_NAME}</div>
          <div style="font-size:0.6rem; color:#00c9a7; letter-spacing:1.5px; text-transform:uppercase; font-weight:500;">{APP_TAGLINE}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    current_page = st.session_state.get("page", "🏠 Dashboard")

    def _nav_btn(label, page_key, icon=""):
        is_active = current_page == page_key
        btn_type = "primary" if is_active else "secondary"
        if st.button(f"{icon} {label}" if icon else label,
                     key=f"nav_{page_key}", use_container_width=True, type=btn_type):
            st.session_state.page = page_key
            st.rerun()

    st.markdown('<span class="sb-section">MAIN</span>', unsafe_allow_html=True)
    _nav_btn("Dashboard",    "🏠 Dashboard",    "⊞")

    st.markdown('<span class="sb-section">WORKFLOW</span>', unsafe_allow_html=True)
    _nav_btn("Design Wizard","🔬 Design Wizard","①")
    _nav_btn("Analysis",     "📊 Analysis",     "②")
    _nav_btn("Optimization", "🎯 Optimization", "③")
    _nav_btn("Visualization","📈 Visualization","④")

    st.markdown('<span class="sb-section">TOOLS</span>', unsafe_allow_html=True)
    _nav_btn("AI Insights",  "🤖 AI Insights",  "◈")
    _nav_btn("Export",       "📤 Export",        "↓")

    # Experiment status
    exp = st.session_state.experiment
    if exp.get("name"):
        st.markdown('<div style="border-top:1px solid #21262d; padding:10px 14px 0 14px; margin-top:8px;">', unsafe_allow_html=True)
        has_design = exp.get("design_matrix") is not None
        has_data   = exp.get("response_data") is not None
        has_models = bool(st.session_state.get("models"))
        has_opt    = bool(st.session_state.get("optimization"))
        st.markdown(f"""
        <div style="font-size:0.78rem; line-height:1.6;">
          <div style="color:#e6edf3; font-weight:600; font-size:0.82rem; margin-bottom:2px;
                      white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{exp['name']}</div>
          <div style="color:#00c9a7; font-size:0.73rem; margin-bottom:6px;">{exp.get('design_type_name','—')}</div>
          {"".join(f'<div style="color:{"#238636" if done else "#30363d"}; font-size:0.75rem;">{"●" if done else "○"} {label}</div>'
                   for label, done in [("Design",has_design),("Data",has_data),("Models",has_models),("Optimized",has_opt)])}
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="border-top:1px solid #21262d; padding:10px 14px; margin-top:auto;">', unsafe_allow_html=True)
    st.markdown(f'<span style="color:#444d56; font-size:0.7rem;">v{APP_VERSION} · Open Source · Free</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    if st.button("⊘ New Experiment", use_container_width=True):
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
        <p>{APP_TAGLINE} — Professional Design of Experiments platform, free and open source.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, _, _ = st.columns([1.4, 1.4, 1, 1])
    with c1:
        if st.button("→ Start New Experiment", type="primary", use_container_width=True):
            st.session_state.page = "🔬 Design Wizard"
            st.rerun()
    with c2:
        if st.button("⚡ Load Demo Experiment", use_container_width=True):
            _load_demo()
            st.rerun()

    st.markdown("---")

    # Workflow cards — AQUIONIS style
    st.markdown("#### Workflow")
    steps_data = [
        ("Design Wizard",  "Define factors & responses. Choose a design type (CCD, Factorial…). Download your experiment template.", "→ Start here", "🔬 Design Wizard"),
        ("Run Experiments","Perform the experimental runs in the lab or plant. Fill in measured response values.", None, None),
        ("Analysis",       "Upload data. Fit MLR or PLS models. Review R², Q², ANOVA and coefficient plots.", None, "📊 Analysis"),
        ("Optimization",   "Set goals (maximize / minimize / target). Run multi-objective optimizer.", None, "🎯 Optimization"),
        ("Visualization",  "Explore 3D response surfaces, contour maps, sweet spot and interaction charts.", None, "📈 Visualization"),
        ("AI Insights",    "Claude analyzes your results and delivers a plain-language expert interpretation.", None, "🤖 AI Insights"),
    ]

    # Row 1: steps 1-3
    cols = st.columns([1, 0.12, 1, 0.12, 1])
    for idx, col_i in enumerate([0, 2, 4]):
        i = idx
        s = steps_data[i]
        num_cls = "active" if i == 0 else "locked"
        card_cls = "wf-card wf-active" if i == 0 else "wf-card"
        action_html = f'<span class="wf-action">{s[2]}</span>' if s[2] else ""
        title_cls = "wf-title" if i == 0 else "wf-title locked"
        with cols[col_i]:
            st.markdown(f"""
            <div class="{card_cls}">
              <div class="wf-num {num_cls}">{i+1}</div>
              <div class="{title_cls}">{s[0]}</div>
              <div class="wf-desc">{s[1]}</div>
              {action_html}
            </div>""", unsafe_allow_html=True)
        if col_i in [0, 2]:
            with cols[col_i + 1]:
                st.markdown('<div class="wf-arrow">›</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: steps 4-6
    cols2 = st.columns([1, 0.12, 1, 0.12, 1])
    for idx, col_i in enumerate([0, 2, 4]):
        i = idx + 3
        s = steps_data[i]
        num_cls = "locked"
        with cols2[col_i]:
            st.markdown(f"""
            <div class="wf-card">
              <div class="wf-num {num_cls}">{i+1}</div>
              <div class="wf-title locked">{s[0]}</div>
              <div class="wf-desc">{s[1]}</div>
            </div>""", unsafe_allow_html=True)
        if col_i in [0, 2]:
            with cols2[col_i + 1]:
                st.markdown('<div class="wf-arrow">›</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Feature / comparison
    col_feat, col_cmp = st.columns([3, 2])

    with col_feat:
        st.markdown("#### Design Types")
        design_info = [
            ("2^k", "Full Factorial", "2–6 factors", "All factor combinations. Complete information."),
            ("½", "Fractional Factorial", "3–15 factors", "Efficient screening; some interactions aliased."),
            ("PB", "Plackett-Burman", "2–23 factors", "Maximum screening power, minimum runs."),
            ("◎", "Central Composite (CCD)", "2–8 factors", "Gold standard for response surface modeling."),
            ("◇", "Box-Behnken (BBD)", "3–7 factors", "No corner points — ideal for constrained ranges."),
        ]
        for sym, name, factors, desc in design_info:
            st.markdown(f"""
            <div style="display:flex; align-items:flex-start; gap:10px; padding:7px 0; border-bottom:1px solid #21262d;">
              <div style="background:#1c2128; border:1px solid #21262d; border-radius:4px;
                          width:28px; height:28px; display:flex; align-items:center; justify-content:center;
                          font-size:0.72rem; font-weight:700; color:#00c9a7; flex-shrink:0;">{sym}</div>
              <div>
                <b style="font-size:0.84rem; color:#e6edf3;">{name}</b>
                <span style="color:#00c9a7; font-size:0.75rem; margin-left:8px;">{factors}</span><br>
                <span style="color:#7d8590; font-size:0.76rem;">{desc}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    with col_cmp:
        st.markdown("#### FactorLab vs MODDE")
        st.markdown("""
<table class="fl-compare" style="width:100%; border-collapse:collapse; border:1px solid #21262d; border-radius:6px; overflow:hidden; background:#161b22;">
<tr><th>Feature</th><th>FactorLab</th><th>MODDE</th></tr>
<tr><td>Price</td><td style="color:#238636;">✓ Free</td><td style="color:#444d56;">€3,000+/yr</td></tr>
<tr><td>Platform</td><td style="color:#238636;">✓ Web</td><td style="color:#444d56;">Windows only</td></tr>
<tr><td>AI Insights</td><td style="color:#238636;">✓ Claude AI</td><td style="color:#444d56;">—</td></tr>
<tr><td>Open Source</td><td style="color:#238636;">✓ GitHub</td><td style="color:#444d56;">—</td></tr>
<tr><td>MLR / PLS</td><td style="color:#238636;">✓</td><td style="color:#00c9a7;">✓</td></tr>
<tr><td>3D Surface</td><td style="color:#238636;">✓ Interactive</td><td style="color:#00c9a7;">✓ Static</td></tr>
<tr><td>Sweet Spot</td><td style="color:#238636;">✓ + Zone Box</td><td style="color:#00c9a7;">✓</td></tr>
<tr><td>Collaboration</td><td style="color:#238636;">✓ Multi-user</td><td style="color:#444d56;">—</td></tr>
</table>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Export formats:**")
        for badge in ["Excel (.xlsx)", "CSV", "Plotly Interactive"]:
            st.markdown(f'<span class="fl-badge">{badge}</span>', unsafe_allow_html=True)


def _render_experiment_dashboard(exp):
    has_design = exp.get("design_matrix") is not None
    has_models = bool(st.session_state.get("models"))
    has_opt    = bool(st.session_state.get("optimization"))

    st.markdown(f"""
    <div style="background:#161b22; border:1px solid #21262d; border-left:3px solid #00c9a7;
                border-radius:6px; padding:14px 20px; margin-bottom:16px;">
        <div style="font-size:1.1rem; font-weight:700; color:#e6edf3;">🔬 {exp['name']}</div>
        <div style="color:#00c9a7; font-size:0.8rem; margin-top:4px;">
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
                <div style="background:#0d1117; border:1px solid #21262d; border-top:2px solid #238636;
                            border-radius:5px; padding:10px 12px; text-align:center;">
                    <div style="font-size:1.0rem; color:#3fb950;">●</div>
                    <div style="font-weight:600; font-size:0.8rem; color:#3fb950; margin-top:3px;">
                        {label}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#161b22; border:1px solid #21262d; border-top:2px solid #30363d;
                            border-radius:5px; padding:10px 12px; text-align:center;">
                    <div style="font-size:1.0rem; color:#7d8590;">○</div>
                    <div style="font-weight:500; font-size:0.8rem; color:#7d8590; margin-top:3px;">
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
            d_color = "#00c9a7" if d > 0.7 else ("#f59e0b" if d > 0.4 else "#ef4444")
            d_label = "Excellent (≥ 0.7)" if d >= 0.7 else ("Acceptable (≥ 0.4)" if d >= 0.4 else "Low — review limits")
            st.markdown(f"""
            <div style="text-align:center; padding:18px 14px;
                        background:#161b22; border-radius:6px;
                        border:1px solid #21262d; border-top:2px solid {d_color};">
                <div style="font-size:2.4rem; font-weight:800; color:{d_color}; line-height:1;">{d:.2f}</div>
                <div style="font-size:0.72rem; font-weight:600; color:{d_color}; text-transform:uppercase;
                            letter-spacing:1px; margin-top:5px;">Overall Desirability</div>
                <div style="font-size:0.72rem; color:#7d8590; margin-top:6px;">{d_label}</div>
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
