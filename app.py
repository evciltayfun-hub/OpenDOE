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

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Sidebar ──────────────────────────────────────── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1424 0%, #102040 60%, #0e3052 100%);
  }
  [data-testid="stSidebar"] * { color: #dde6f0 !important; }
  [data-testid="stSidebar"] .stRadio > label { display: none; }
  [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: rgba(255,255,255,0.04);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 3px;
    display: block;
    transition: background 0.15s;
    font-size: 0.93rem;
    border-left: 3px solid transparent;
  }
  [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.10);
  }

  /* ── Metrics ──────────────────────────────────────── */
  [data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
  }

  /* ── Step cards ───────────────────────────────────── */
  .fl-step-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 22px 20px 18px 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    height: 100%;
    transition: box-shadow 0.2s, transform 0.2s;
  }
  .fl-step-card:hover {
    box-shadow: 0 6px 18px rgba(30,64,175,0.13);
    transform: translateY(-2px);
  }
  .fl-step-icon {
    font-size: 2.2rem;
    margin-bottom: 10px;
  }
  .fl-step-num {
    display: inline-block;
    background: #1e40af;
    color: white !important;
    border-radius: 50%;
    width: 26px; height: 26px;
    line-height: 26px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 8px;
  }
  .fl-step-title {
    font-weight: 700;
    font-size: 1.0rem;
    color: #0f172a;
    margin-bottom: 6px;
  }
  .fl-step-desc {
    font-size: 0.82rem;
    color: #64748b;
    line-height: 1.45;
  }
  .fl-arrow {
    font-size: 1.5rem;
    color: #cbd5e1;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    padding-top: 36px;
  }

  /* ── Hero banner ──────────────────────────────────── */
  .fl-hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #1e40af 50%, #0e7490 100%);
    border-radius: 16px;
    padding: 36px 40px;
    color: white;
    margin-bottom: 28px;
  }
  .fl-hero h1 { color: white !important; font-size: 2.4rem !important; margin: 0; }
  .fl-hero p  { color: #bfdbfe; margin: 8px 0 0 0; font-size: 1.05rem; }

  /* ── Feature badge ────────────────────────────────── */
  .fl-badge {
    display: inline-block;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: #1e40af;
    font-weight: 600;
    margin: 3px 3px 3px 0;
  }

  /* ── Compare table ────────────────────────────────── */
  .fl-compare td { padding: 6px 14px; font-size: 0.87rem; }
  .fl-compare th { background: #f1f5f9; padding: 8px 14px; font-size: 0.83rem; color: #475569; }

  /* ── Buttons ──────────────────────────────────────── */
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1e40af, #0e7490);
    border: none; border-radius: 8px; font-weight: 600;
  }
  .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1d4ed8, #0891b2);
    transform: translateY(-1px);
  }

  /* ── Tabs ─────────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #f1f5f9; border-radius: 10px; padding: 4px;
  }
  .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 18px; }
  .stTabs [aria-selected="true"] { background: white; box-shadow: 0 1px 4px rgba(0,0,0,0.10); }

  h1 { color: #0f172a; }
  h2, h3 { color: #1e3a5f; }
  .stAlert { border-radius: 8px; }
  .stSuccess { border-left: 4px solid #10b981; }
  hr { border-color: #e2e8f0; }
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
    <div style="padding: 18px 4px 10px 4px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <span style="font-size:2rem;">🔬</span>
            <div>
                <div style="font-size:1.45rem; font-weight:800; letter-spacing:-0.5px; color:#f1f5f9;">
                    {APP_NAME}
                </div>
                <div style="font-size:0.72rem; color:#64a3c8; letter-spacing:1.5px; text-transform:uppercase;">
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
    st.markdown(f"<span style='color:#475569; font-size:0.75rem;'>v{APP_VERSION} · Open Source · Free Forever</span>",
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
    <div style="background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
                border: 1px solid #bae6fd; border-radius: 14px; padding: 20px 28px; margin-bottom: 20px;">
        <div style="font-size:1.4rem; font-weight:800; color:#0c4a6e;">🔬 {exp['name']}</div>
        <div style="color:#0369a1; font-size:0.9rem; margin-top:4px;">
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
            color = "#10b981" if done else "#e2e8f0"
            icon  = "✅" if done else "○"
            st.markdown(f"""
            <div style="border:2px solid {color}; border-radius:10px; padding:14px;
                        text-align:center; background:{'#f0fdf4' if done else '#fafafa'};">
                <div style="font-size:1.1rem;">{icon}</div>
                <div style="font-weight:600; font-size:0.85rem; color:{'#065f46' if done else '#94a3b8'};">
                    {label}</div>
            </div>
            """, unsafe_allow_html=True)
            if not done:
                if st.button(f"Go →", key=f"go_{i}", use_container_width=True):
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
            color = "#10b981" if d > 0.7 else ("#f59e0b" if d > 0.4 else "#ef4444")
            st.markdown(f"""
            <div style="text-align:center; padding:24px 16px;
                        background:{color}18; border-radius:14px; border:2px solid {color};">
                <div style="font-size:2.8rem; font-weight:900; color:{color};">{d:.2f}</div>
                <div style="font-size:0.85rem; font-weight:600; color:{color};">Overall Desirability</div>
                <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">
                    {'Excellent ≥ 0.7' if d>=0.7 else ('Acceptable ≥ 0.4' if d>=0.4 else 'Low — review limits')}</div>
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
