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

# ─── Custom CSS — Theme: Arctic Lab (Option B) ────────────────────────────────
st.markdown("""
<style>
  /* ════════════════════════════════════════════════════
     ARCTIC LAB — Clean Scientific
     Sidebar: #1e3a5f navy  |  Content: white / #f8fafc
     Accent:  #0891b2 teal  |  Secondary: #059669 green
  ════════════════════════════════════════════════════ */

  /* ── Page background ───────────────────────────────── */
  .stApp { background-color: #f0f5fa; }
  .block-container { background: transparent; }

  /* ── Sidebar ───────────────────────────────────────── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #132b4a 0%, #1a3a5c 55%, #1e4474 100%);
    border-right: 1px solid #14324f;
  }
  [data-testid="stSidebar"] * { color: #cfe2f0 !important; }
  [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.10) !important; }

  /* Nav items */
  [data-testid="stSidebar"] .stRadio > label { display: none; }
  [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: transparent;
    border-radius: 8px;
    padding: 9px 14px;
    margin-bottom: 2px;
    display: block;
    transition: background 0.15s;
    font-size: 0.92rem;
    border-left: 3px solid transparent;
    color: #a8c4d8 !important;
  }
  [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(14,165,233,0.12);
    color: #7dd3fc !important;
  }
  [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] {
    background: rgba(14,165,233,0.18);
    border-left: 3px solid #0ea5e9;
    color: #7dd3fc !important;
  }

  /* ── Main content card wrapper ─────────────────────── */
  [data-testid="stMainBlockContainer"] > div > div > div {
    background: transparent;
  }

  /* ── Metric cards ──────────────────────────────────── */
  [data-testid="metric-container"] {
    background: white;
    border: 1px solid #dce8f2;
    border-top: 3px solid #0891b2;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 1px 6px rgba(8,145,178,0.07);
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0369a1 !important;
    font-weight: 800;
  }

  /* ── Section panels ────────────────────────────────── */
  .fl-panel {
    background: white;
    border: 1px solid #dce8f2;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    margin-bottom: 16px;
  }

  /* ── Step cards ────────────────────────────────────── */
  .fl-step-card {
    background: white;
    border: 1px solid #dce8f2;
    border-radius: 14px;
    padding: 24px 18px 20px 18px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(8,145,178,0.06);
    height: 100%;
    transition: box-shadow 0.2s, transform 0.2s, border-color 0.2s;
  }
  .fl-step-card:hover {
    box-shadow: 0 8px 24px rgba(8,145,178,0.15);
    border-color: #7dd3fc;
    transform: translateY(-3px);
  }
  .fl-step-icon { font-size: 2.2rem; margin-bottom: 10px; }
  .fl-step-num {
    display: inline-block;
    background: linear-gradient(135deg, #0369a1, #0891b2);
    color: white !important;
    border-radius: 50%;
    width: 28px; height: 28px; line-height: 28px;
    font-size: 0.78rem; font-weight: 800; margin-bottom: 10px;
  }
  .fl-step-title { font-weight: 700; font-size: 1.0rem; color: #0c2d4e; margin-bottom: 6px; }
  .fl-step-desc  { font-size: 0.82rem; color: #64748b; line-height: 1.5; }
  .fl-arrow {
    font-size: 1.6rem; color: #bae6fd;
    display: flex; align-items: center; justify-content: center;
    height: 100%; padding-top: 40px;
  }

  /* ── Hero banner ───────────────────────────────────── */
  .fl-hero {
    background: linear-gradient(135deg, #0c2d4e 0%, #1e3a5f 45%, #0c6e8a 100%);
    border-radius: 16px;
    padding: 38px 44px;
    color: white;
    margin-bottom: 28px;
    border: 1px solid #1e4d6b;
    box-shadow: 0 4px 24px rgba(12,45,78,0.25);
  }
  .fl-hero h1 { color: #f0f9ff !important; font-size: 2.5rem !important; margin: 0; font-weight: 800; }
  .fl-hero p  { color: #7dd3fc; margin: 10px 0 0 0; font-size: 1.05rem; letter-spacing: 0.2px; }

  /* ── Feature badge ─────────────────────────────────── */
  .fl-badge {
    display: inline-block;
    background: #e0f7fa;
    border: 1px solid #b2ebf2;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: #006064;
    font-weight: 600;
    margin: 3px 3px 3px 0;
  }

  /* ── Compare table ─────────────────────────────────── */
  .fl-compare td { padding: 7px 14px; font-size: 0.87rem; border-bottom: 1px solid #f1f5f9; }
  .fl-compare th { background: #f0f9ff; padding: 9px 14px; font-size: 0.82rem;
                   color: #0369a1; font-weight: 700; border-bottom: 2px solid #bae6fd; }

  /* ── Primary button ────────────────────────────────── */
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0369a1 0%, #0891b2 60%, #059669 100%);
    border: none; border-radius: 8px; font-weight: 600;
    color: white !important;
    box-shadow: 0 2px 8px rgba(8,145,178,0.3);
    transition: all 0.2s;
  }
  .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0284c7 0%, #06b6d4 60%, #10b981 100%);
    box-shadow: 0 4px 14px rgba(8,145,178,0.4);
    transform: translateY(-1px);
  }
  /* Secondary / default button */
  .stButton > button:not([kind="primary"]) {
    border: 1px solid #bae6fd !important;
    color: #0369a1 !important;
    background: white !important;
    border-radius: 8px;
    font-weight: 500;
  }
  .stButton > button:not([kind="primary"]):hover {
    background: #f0f9ff !important;
    border-color: #0891b2 !important;
  }

  /* ── Tabs ──────────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #e8f4fb; border-radius: 10px; padding: 4px;
    border: 1px solid #c8e3f0;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px; padding: 8px 20px;
    color: #5b8fa8 !important; font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    background: white;
    box-shadow: 0 1px 6px rgba(8,145,178,0.12);
    color: #0369a1 !important;
    font-weight: 700;
  }

  /* ── Expanders ─────────────────────────────────────── */
  [data-testid="stExpander"] {
    border: 1px solid #dce8f2 !important;
    border-radius: 10px !important;
    background: white;
  }

  /* ── Data tables ───────────────────────────────────── */
  [data-testid="stDataFrame"] {
    border: 1px solid #dce8f2;
    border-radius: 8px;
    overflow: hidden;
  }

  /* ── Progress & sliders ────────────────────────────── */
  [data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #0891b2, #059669) !important;
  }

  /* ── Alerts ────────────────────────────────────────── */
  .stAlert { border-radius: 10px; }
  .stSuccess {
    background: #f0fdf4 !important;
    border-left: 4px solid #059669 !important;
    color: #14532d !important;
  }
  .stInfo {
    background: #f0f9ff !important;
    border-left: 4px solid #0891b2 !important;
    color: #0c4a6e !important;
  }
  .stWarning {
    background: #fffbeb !important;
    border-left: 4px solid #f59e0b !important;
  }

  /* ── Typography ─────────────────────────────────────── */
  h1 { color: #0c2d4e !important; font-weight: 800; }
  h2 { color: #0c4a6e !important; font-weight: 700; }
  h3 { color: #075985 !important; font-weight: 600; }
  hr { border-color: #dce8f2 !important; margin: 20px 0; }

  /* ── Input fields ──────────────────────────────────── */
  [data-testid="stTextInput"] > div > div > input,
  [data-testid="stNumberInput"] > div > div > input {
    border-radius: 7px !important;
    border: 1px solid #c8e3f0 !important;
  }
  [data-testid="stTextInput"] > div > div > input:focus,
  [data-testid="stNumberInput"] > div > div > input:focus {
    border-color: #0891b2 !important;
    box-shadow: 0 0 0 3px rgba(8,145,178,0.12) !important;
  }
  [data-testid="stSelectbox"] > div > div {
    border-radius: 7px !important;
    border: 1px solid #c8e3f0 !important;
  }

  /* ── Divider ───────────────────────────────────────── */
  .fl-divider {
    height: 2px;
    background: linear-gradient(90deg, #0891b2, #059669, transparent);
    border-radius: 2px; margin: 24px 0;
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
    <div style="padding: 20px 6px 12px 6px;">
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="background:linear-gradient(135deg,#0891b2,#059669);
                        border-radius:10px; width:42px; height:42px;
                        display:flex; align-items:center; justify-content:center;
                        font-size:1.4rem; flex-shrink:0;">🔬</div>
            <div>
                <div style="font-size:1.4rem; font-weight:800; letter-spacing:-0.5px; color:#f0f9ff;">
                    {APP_NAME}
                </div>
                <div style="font-size:0.65rem; color:#7dd3fc; letter-spacing:2px;
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
        <div style="font-size:0.84rem; color:#cbd5e1; line-height:1.6;">
          <b style="color:#f1f5f9; font-size:0.95rem;">{exp['name']}</b><br>
          <span style="color:#7dd3fc;">{exp.get('design_type_name', '—')}</span><br>
          {n_f} factors &nbsp;·&nbsp; {n_r} responses
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:10px;'>", unsafe_allow_html=True)
        steps = [("Design", has_design), ("Data", has_data),
                 ("Models", has_models), ("Optimized", has_opt)]
        for label, done in steps:
            icon  = "✅" if done else "◻️"
            color = "#4ade80" if done else "#475569"
            st.markdown(f"<span style='color:{color}; font-size:0.82rem;'>{icon} {label}</span>",
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"<span style='color:#5b8fa8; font-size:0.74rem;'>v{APP_VERSION} &nbsp;·&nbsp; Open Source &nbsp;·&nbsp; Free Forever</span>",
                unsafe_allow_html=True)
    if st.button("🗑️ New Experiment", use_container_width=True):
        for k in ["experiment", "models", "optimization", "ai_history", "wizard_step"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()


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
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        if st.button("🚀 Start New Experiment", type="primary", use_container_width=True):
            st.session_state.page = "🔬 Design Wizard"
            st.rerun()
    with c2:
        if st.button("📖 How It Works", use_container_width=True):
            st.session_state.show_guide = not st.session_state.get("show_guide", False)
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
            <div style="display:flex; align-items:flex-start; gap:12px; padding:10px 0;
                        border-bottom:1px solid #f1f5f9;">
                <span style="font-size:1.4rem; margin-top:2px;">{icon}</span>
                <div>
                    <b style="font-size:0.93rem;">{name}</b>
                    <span style="color:#64748b; font-size:0.82rem; margin-left:8px;">{factors}</span><br>
                    <span style="color:#64748b; font-size:0.82rem;">{desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_cmp:
        st.markdown("### FactorLab vs MODDE")
        st.markdown("""
<table class="fl-compare" style="width:100%; border-collapse:collapse; border-radius:10px; overflow:hidden; border:1px solid #e2e8f0;">
<tr><th>Feature</th><th>FactorLab</th><th>MODDE</th></tr>
<tr><td>Price</td><td>✅ <b>Free</b></td><td>💰 €3,000+/yr</td></tr>
<tr><td>Platform</td><td>✅ <b>Web</b></td><td>Windows only</td></tr>
<tr><td>AI Interpretation</td><td>✅ <b>Claude AI</b></td><td>❌</td></tr>
<tr><td>Open Source</td><td>✅ GitHub</td><td>❌</td></tr>
<tr><td>MLR / PLS</td><td>✅</td><td>✅</td></tr>
<tr><td>Response Surface</td><td>✅ Interactive</td><td>✅ Static</td></tr>
<tr><td>Sweet Spot Plot</td><td>✅ + Zone Box</td><td>✅</td></tr>
<tr><td>Collaboration</td><td>✅ Multi-user</td><td>❌</td></tr>
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
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                border: 1px solid #bae6fd; border-left: 5px solid #0891b2;
                border-radius: 12px; padding: 20px 28px; margin-bottom: 22px;
                box-shadow: 0 2px 10px rgba(8,145,178,0.08);">
        <div style="font-size:1.4rem; font-weight:800; color:#0c2d4e;">🔬 {exp['name']}</div>
        <div style="color:#0369a1; font-size:0.88rem; margin-top:5px; font-weight:500;">
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
                <div style="border:1px solid #a7f3d0; border-top:3px solid #059669;
                            border-radius:10px; padding:14px; text-align:center;
                            background:#f0fdf4; box-shadow:0 1px 4px rgba(5,150,105,0.08);">
                    <div style="font-size:1.3rem;">✅</div>
                    <div style="font-weight:700; font-size:0.83rem; color:#065f46; margin-top:4px;">
                        {label}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="border:1px solid #bae6fd; border-top:3px solid #0891b2;
                            border-radius:10px; padding:14px; text-align:center;
                            background:white; box-shadow:0 1px 4px rgba(8,145,178,0.06);">
                    <div style="font-size:1.3rem;">◻️</div>
                    <div style="font-weight:600; font-size:0.83rem; color:#0369a1; margin-top:4px;">
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
            <div style="text-align:center; padding:26px 16px;
                        background:{d_color}10; border-radius:14px;
                        border:2px solid {d_color}40;
                        box-shadow:0 2px 14px {d_color}20;">
                <div style="font-size:3rem; font-weight:900; color:{d_color}; line-height:1;">{d:.2f}</div>
                <div style="font-size:0.78rem; font-weight:700; color:{d_color}; text-transform:uppercase;
                            letter-spacing:1px; margin-top:6px;">Overall Desirability</div>
                <div style="font-size:0.75rem; color:#64748b; margin-top:8px;">{d_label}</div>
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
