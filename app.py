"""OpenDOE — Open-source Design of Experiments Platform."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from modules.design_wizard import render_design_wizard
from modules.analysis_wizard import render_analysis
from modules.optimization import render_optimization
from modules.visualization import render_visualization
from modules.ai_insights import render_ai_insights

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OpenDOE",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e3a5f 100%);
  }
  [data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
  }
  [data-testid="stSidebar"] .stRadio > label { display: none; }
  [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 4px;
    display: block;
    transition: background 0.2s;
    font-size: 0.95rem;
    cursor: pointer;
  }
  [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.12);
  }
  [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] {
    background: rgba(59,130,246,0.35);
  }

  /* Metrics */
  [data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 16px;
  }

  /* Cards */
  .doe-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }

  /* Primary button tweak */
  .stButton > button[kind="primary"] {
    background: #1e40af;
    border: none;
    border-radius: 8px;
  }
  .stButton > button[kind="primary"]:hover {
    background: #1d4ed8;
  }

  /* Tab style */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #f1f5f9;
    border-radius: 10px;
    padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
  }
  .stTabs [aria-selected="true"] {
    background: white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }

  /* Title */
  h1 { color: #0f172a; }
  h2, h3 { color: #1e3a5f; }
  .stAlert { border-radius: 8px; }
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
    st.markdown("""
    <div style="padding: 16px 0 8px 0;">
        <div style="font-size: 1.8rem; font-weight: 800; letter-spacing: -0.5px;">
            ⚗️ OpenDOE
        </div>
        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">
            Design of Experiments Platform
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
        has_data = exp.get("response_data") is not None
        has_models = bool(st.session_state.get("models"))
        has_opt = bool(st.session_state.get("optimization"))

        st.markdown(f"""
        <div style="font-size:0.85rem; color:#cbd5e1;">
          <b style="color:#e2e8f0; font-size:0.95rem;">{exp['name']}</b><br>
          {exp.get('design_type_name', '—')}<br>
          {n_f} factors · {n_r} responses
        </div>
        """, unsafe_allow_html=True)

        status_items = [
            ("Design", has_design),
            ("Data", has_data),
            ("Models", has_models),
            ("Optimized", has_opt),
        ]
        for label, done in status_items:
            icon = "✅" if done else "○"
            col_str = "#10b981" if done else "#64748b"
            st.markdown(f"<span style='color:{col_str}; font-size:0.82rem;'>{icon} {label}</span>",
                        unsafe_allow_html=True)

    st.divider()
    st.caption("v1.0 · Open Source · Free Forever")
    if st.button("🗑️ Reset Experiment", use_container_width=True):
        for k in ["experiment", "models", "optimization", "ai_history", "wizard_step"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()


# ─── Dashboard ────────────────────────────────────────────────────────────────
def render_dashboard():
    st.title("Welcome to OpenDOE")
    st.markdown("*The free, web-based Design of Experiments platform — built to exceed MODDE.*")

    exp = st.session_state.experiment

    if exp.get("name"):
        # Existing experiment summary
        has_design = exp.get("design_matrix") is not None
        has_models = bool(st.session_state.get("models"))
        has_opt = bool(st.session_state.get("optimization"))

        st.success(f"**Current experiment:** {exp['name']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Factors", len(exp.get("factors", [])))
        c2.metric("Responses", len(exp.get("responses", [])))
        c3.metric("Runs", len(exp.get("design_matrix", [])) if has_design else "—")
        c4.metric("Models fitted", len(st.session_state.get("models", {})))

        if has_models:
            st.divider()
            st.markdown("### Model Summary")
            rows = []
            for rname, m in st.session_state.models.items():
                rows.append({
                    "Response": rname, "Type": m["type"],
                    "R²": f"{m['r2']:.3f}", "Q²": f"{m['q2']:.3f}",
                    "Status": "✅" if m["r2"] > 0.8 and m["q2"] > 0.5 else "⚠️",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if has_opt:
            opt = st.session_state.optimization
            d = opt.get("desirability", 0)
            st.divider()
            st.markdown("### Optimization Result")
            c1, c2 = st.columns([1, 2])
            with c1:
                color = "#10b981" if d > 0.7 else ("#f59e0b" if d > 0.4 else "#ef4444")
                st.markdown(f"""
                <div style="text-align:center; padding:20px;
                            background:{color}22; border-radius:12px;
                            border: 2px solid {color};">
                    <div style="font-size:2.5rem; font-weight:800; color:{color};">{d:.2f}</div>
                    <div style="color:{color}; font-weight:600;">Overall Desirability</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                opt_nat = opt.get("optimal_natural", {})
                for fname, val in opt_nat.items():
                    unit = next((f.get("unit", "") for f in exp.get("factors", [])
                                 if f["name"] == fname), "")
                    st.markdown(f"**{fname}:** `{val:.4f}` {unit}")

    else:
        # Welcome screen
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("""
### What is OpenDOE?

OpenDOE is a professional **Design of Experiments** platform that helps you:

- **Design** systematic experiments with minimal runs
- **Analyze** data with MLR and PLS models
- **Optimize** multiple responses simultaneously
- **Visualize** response surfaces and design spaces
- **Interpret** results with AI-powered analysis (Claude)

### Supported Design Types
| Design | Runs | Best For |
|--------|------|----------|
| Full Factorial (2^k) | 4–64 | ≤5 factors, complete information |
| Fractional Factorial | 8–32 | Screening many factors |
| Plackett-Burman | 4–24 | Maximum factor screening |
| Central Composite (CCD) | 13–90 | Response surface, optimization |
| Box-Behnken (BBD) | 15–62 | RSM without extreme corners |
""")

        with c2:
            st.markdown("""
### vs. MODDE (Sartorius)

| Feature | OpenDOE | MODDE |
|---------|---------|-------|
| Cost | **Free** | €3,000+/yr |
| Platform | **Web** | Windows only |
| AI Insights | **✅ Claude** | ❌ |
| Open Source | **✅** | ❌ |
| Design types | **5 types** | 10+ |
| PLS / MLR | **✅** | ✅ |
| 3D Visualization | **✅** | ✅ |
| Collaboration | **✅** | ❌ |
""")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Start New Experiment", type="primary", use_container_width=True):
                st.session_state.page = "🔬 Design Wizard"
                st.rerun()
        with col2:
            if st.button("📖 Quick Start Guide", use_container_width=True):
                st.session_state.show_guide = True

        if st.session_state.get("show_guide"):
            st.markdown("""
---
### Quick Start Guide

1. **🔬 Design Wizard** — Enter your factors (inputs) and responses (outputs), choose a design type, and download your experimental template.
2. **Run your experiments** — Fill in the measured response values in the downloaded Excel file.
3. **📊 Analysis** — Upload your data and fit MLR or PLS models. Review R², Q², and coefficient plots.
4. **🎯 Optimization** — Set goals for each response (maximize/minimize/target) and find optimal conditions.
5. **📈 Visualization** — Explore response surfaces, contour plots, and interaction plots interactively.
6. **🤖 AI Insights** — Get expert interpretation from Claude in plain language.
7. **📤 Export** — Download results in Excel or PDF format.
""")


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
        bold = wb.add_format({"bold": True})

        # Sheet 1: Experiment Info
        info_data = {
            "Field": ["Name", "Objective", "Design", "Factors", "Responses", "Runs"],
            "Value": [
                exp.get("name", ""),
                exp.get("objective", ""),
                exp.get("design_type_name", ""),
                len(exp.get("factors", [])),
                len(exp.get("responses", [])),
                len(exp.get("design_matrix", [])) if exp.get("design_matrix") is not None else 0,
            ]
        }
        pd.DataFrame(info_data).to_excel(writer, sheet_name="Experiment Info", index=False)

        # Sheet 2: Design Matrix (natural)
        nat = exp.get("natural_matrix")
        if nat is not None:
            resp_data = exp.get("response_data")
            if resp_data is not None:
                resp_cols = [r["name"] for r in exp["responses"]]
                for rc in resp_cols:
                    if rc in resp_data.columns:
                        nat = nat.copy()
                        nat[rc] = resp_data[rc].values
            nat.to_excel(writer, sheet_name="Design Matrix")

        # Sheet 3: Model Results
        if models:
            model_rows = []
            for rname, m in models.items():
                model_rows.append({
                    "Response": rname, "Type": m["type"],
                    "R²": m["r2"], "R²adj": m["r2_adj"],
                    "Q²": m["q2"], "RMSE": m["rmse"],
                })
            pd.DataFrame(model_rows).to_excel(writer, sheet_name="Model Summary", index=False)

            for rname, m in models.items():
                if m["type"] == "MLR":
                    safe = rname[:25]
                    m["coefficients"].to_excel(writer, sheet_name=f"Coeffs_{safe}", index=False)

        # Sheet 4: Optimization
        if optimization:
            opt_rows = []
            for k, v in optimization.get("optimal_natural", {}).items():
                opt_rows.append({"Type": "Factor", "Name": k, "Optimal Value": v})
            for k, v in optimization.get("predicted_responses", {}).items():
                opt_rows.append({"Type": "Response", "Name": k, "Predicted": v})
            opt_rows.append({"Type": "Overall", "Name": "Desirability",
                             "Optimal Value": optimization.get("desirability", "")})
            pd.DataFrame(opt_rows).to_excel(writer, sheet_name="Optimization", index=False)

    st.download_button(
        "⬇️ Download Full Results (Excel)",
        data=buf.getvalue(),
        file_name=f"{exp['name'].replace(' ', '_')}_OpenDOE_Results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

    st.divider()
    st.markdown("### What's included in the export:")
    st.markdown("""
- **Experiment Info** — design type, factors, responses
- **Design Matrix** — all experimental runs with measured responses
- **Model Summary** — R², Q², RMSE for all fitted models
- **Coefficients** — MLR coefficient tables (one sheet per response)
- **Optimization** — optimal factor settings and predicted responses
""")


# ─── Router ───────────────────────────────────────────────────────────────────
page = st.session_state.get("page", "🏠 Dashboard")

if page == "🏠 Dashboard":
    render_dashboard()
elif page == "🔬 Design Wizard":
    render_design_wizard()
elif page == "📊 Analysis":
    render_analysis()
elif page == "🎯 Optimization":
    render_optimization()
elif page == "📈 Visualization":
    render_visualization()
elif page == "🤖 AI Insights":
    render_ai_insights()
elif page == "📤 Export":
    render_export()
