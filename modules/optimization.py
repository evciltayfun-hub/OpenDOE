"""Optimization — multi-response desirability optimization."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.stats import run_optimization, perturbation_analysis, predict_response
from utils.doe_engine import coded_to_natural

PLOTLY_TEMPLATE = "plotly_dark"
PRIMARY = "#3b82f6"
SUCCESS = "#10b981"
WARN = "#f59e0b"


def _nav(page):
    st.session_state.page = page
    st.rerun()


def render_optimization():
    st.title("🎯 Optimization")

    exp = st.session_state.experiment
    if not exp.get("name"):
        st.warning("No experiment loaded.")
        if st.button("← Go to Design Wizard", key="opt_nodsgn"):
            _nav("🔬 Design Wizard")
        return
    if not st.session_state.get("models"):
        st.warning("No fitted models — please complete the Analysis step first.")
        if st.button("← Go to Analysis", key="opt_nomdl"):
            _nav("📊 Analysis")
        return

    # Navigation row
    c_back, c_info, c_fwd = st.columns([1, 4, 1])
    with c_back:
        if st.button("← Analysis", use_container_width=True):
            _nav("📊 Analysis")
    with c_info:
        has_opt = bool(st.session_state.get("optimization"))
        st.markdown(
            f"<div style='font-size:0.82rem; color:#64748b; padding-top:6px;'>"
            f"<b style='color:#e2e8f0;'>{exp['name']}</b> &nbsp;·&nbsp; "
            f"{len(st.session_state.get('models', {}))} models fitted"
            f"</div>", unsafe_allow_html=True)
    with c_fwd:
        has_opt = bool(st.session_state.get("optimization"))
        if st.button("Visualization →", use_container_width=True, disabled=not has_opt):
            _nav("📈 Visualization")

    st.divider()
    tab1, tab2, tab3 = st.tabs(["⚙️ Configure & Run", "📍 Optimal Conditions", "📉 Perturbation"])

    with tab1:
        _tab_configure(exp)
    with tab2:
        _tab_results(exp)
    with tab3:
        _tab_perturbation(exp)


# ─── Tab 1: Configure ─────────────────────────────────────────────────────────

def _tab_configure(exp):
    st.subheader("Configure Desirability Functions")
    st.markdown("Define how each response contributes to the overall optimum.")

    responses = exp.get("responses", [])
    models = st.session_state.models
    model_rnames = list(models.keys())
    resp_map = {r["name"]: r for r in responses}

    updated_configs = {}
    updated_importances = {}

    for rname in model_rnames:
        rconf = resp_map.get(rname, {})
        st.markdown(f"**{rname}**")
        c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 2])

        with c1:
            default_goal = rconf.get("goal", "Maximize")
            goal = st.selectbox("Goal", ["Maximize", "Minimize", "Target"],
                                index=["Maximize", "Minimize", "Target"].index(default_goal),
                                key=f"opt_goal_{rname}")
        with c2:
            def_lo = rconf.get("lower_limit", None)
            lo_str = st.text_input("Lower Limit", value=str(def_lo) if def_lo is not None else "",
                                   key=f"opt_lo_{rname}", placeholder="min acceptable")
        with c3:
            def_hi = rconf.get("upper_limit", None)
            hi_str = st.text_input("Upper Limit", value=str(def_hi) if def_hi is not None else "",
                                   key=f"opt_hi_{rname}", placeholder="max acceptable")
        with c4:
            if goal == "Target":
                def_t = rconf.get("target", None)
                t_str = st.text_input("Target", value=str(def_t) if def_t is not None else "",
                                      key=f"opt_t_{rname}", placeholder="ideal value")
            else:
                t_str = ""
                st.empty()
        with c5:
            imp = st.slider("Importance", 1, 5,
                            value=int(rconf.get("importance", 3)),
                            key=f"opt_imp_{rname}")

        def _f(s):
            try:
                return float(s)
            except (ValueError, TypeError):
                return None

        updated_configs[rname] = {
            "goal": goal.lower(),
            "lower_limit": _f(lo_str),
            "upper_limit": _f(hi_str),
            "target": _f(t_str) if goal == "Target" else None,
            "weight": 1.0,
        }
        updated_importances[rname] = imp
        st.divider()

    st.markdown("**Factor Constraints**")
    factors = exp["factors"]
    constraints = {}
    cols = st.columns(min(4, len(factors)))
    for i, f in enumerate(factors):
        with cols[i % len(cols)]:
            st.markdown(f"*{f['name']}*")
            c_lo = st.number_input(f"Low", value=float(f["low"]),
                                   key=f"constr_lo_{i}", label_visibility="collapsed")
            c_hi = st.number_input(f"High", value=float(f["high"]),
                                   key=f"constr_hi_{i}", label_visibility="collapsed")
            constraints[f["name"]] = (c_lo, c_hi)

    if st.button("🚀 Run Optimization", type="primary"):
        factor_ranges = [constraints[f["name"]] for f in factors]
        fname_list = [f["name"] for f in factors]
        with st.spinner("Running global optimization (differential evolution)..."):
            try:
                result = run_optimization(
                    models=models,
                    factor_names=fname_list,
                    factor_ranges=factor_ranges,
                    response_configs=updated_configs,
                    importances=updated_importances,
                    n_restarts=5,
                )
                result["response_configs"] = updated_configs
                result["importances"] = updated_importances
                result["factor_constraints"] = constraints
                st.session_state.optimization = result
                d = result["desirability"]
                st.success(f"✅ Optimization complete! Overall desirability: **{d:.3f}**")
                if d < 0.3:
                    st.warning("Low desirability — check that your limits are achievable within the design space.")
                if st.button("→ View in Visualization", type="primary", key="goto_viz"):
                    _nav("📈 Visualization")
            except Exception as e:
                st.error(f"Optimization failed: {e}")


# ─── Tab 2: Results ───────────────────────────────────────────────────────────

def _tab_results(exp):
    st.subheader("Optimal Conditions")
    opt = st.session_state.get("optimization", {})
    if not opt:
        st.info("Run optimization in the **Configure & Run** tab first.")
        return


    d = opt["desirability"]
    d_color = SUCCESS if d > 0.7 else (WARN if d > 0.4 else "#ef4444")

    # Overall desirability gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=d,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Overall Desirability"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": d_color},
            "steps": [
                {"range": [0, 0.4], "color": "#fee2e2"},
                {"range": [0.4, 0.7], "color": "#fef9c3"},
                {"range": [0.7, 1.0], "color": "#d1fae5"},
            ],
            "threshold": {"line": {"color": PRIMARY, "width": 3}, "value": 0.7},
        },
    ))
    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()
    factors = exp["factors"]
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ⚗️ Optimal Factor Settings")
        factor_rows = []
        opt_natural = opt.get("optimal_natural", {})
        for f in factors:
            n = f["name"]
            val = opt_natural.get(n, "—")
            unit = f.get("unit", "")
            factor_rows.append({
                "Factor": n,
                "Optimal Value": f"{val:.4f}" if isinstance(val, float) else val,
                "Unit": unit,
                "Low": f["low"],
                "High": f["high"],
            })
        fdf = pd.DataFrame(factor_rows)
        st.dataframe(fdf, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### 📊 Predicted Responses at Optimum")
        pred_rows = []
        pred_resp = opt.get("predicted_responses", {})
        resp_map = {r["name"]: r for r in exp.get("responses", [])}
        for rname, val in pred_resp.items():
            rconf = resp_map.get(rname, {})
            pred_rows.append({
                "Response": rname,
                "Predicted": f"{val:.4f}",
                "Unit": rconf.get("unit", ""),
                "Goal": rconf.get("goal", "—"),
            })
        st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)

    # Factor setting bar chart
    st.divider()
    st.markdown("### Factor Settings Relative to Range")
    factor_names = [f["name"] for f in factors]
    opt_coded = opt.get("optimal_coded", {})
    bars = [{"Factor": n, "Coded Value": opt_coded.get(n, 0)} for n in factor_names]
    bars_df = pd.DataFrame(bars)
    fig_bars = go.Figure(go.Bar(
        x=bars_df["Factor"],
        y=bars_df["Coded Value"],
        marker_color=[SUCCESS if v >= 0 else PRIMARY for v in bars_df["Coded Value"]],
        hovertemplate="%{x}<br>Coded: %{y:.3f}<extra></extra>",
    ))
    fig_bars.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
    fig_bars.update_layout(template=PLOTLY_TEMPLATE, height=320,
                           yaxis=dict(range=[-1.2, 1.2], title="Coded Value (−1=Low, +1=High)"),
                           title="Optimal Factor Settings in Coded Space")
    st.plotly_chart(fig_bars, use_container_width=True)

    # Export button
    st.divider()
    export_data = []
    for f in factors:
        export_data.append({"Type": "Factor", "Name": f["name"],
                            "Optimal": opt_natural.get(f["name"], ""), "Unit": f.get("unit", "")})
    for rname, val in pred_resp.items():
        rconf = resp_map.get(rname, {})
        export_data.append({"Type": "Response", "Name": rname,
                            "Optimal": val, "Unit": rconf.get("unit", "")})
    export_df = pd.DataFrame(export_data)
    csv = export_df.to_csv(index=False)
    st.download_button("⬇️ Download Optimal Conditions (CSV)", data=csv,
                       file_name="optimal_conditions.csv", mime="text/csv")


# ─── Tab 3: Perturbation ──────────────────────────────────────────────────────

def _tab_perturbation(exp):
    st.subheader("Perturbation Analysis")
    st.markdown("Shows how each response changes as individual factors deviate from the optimum.")

    opt = st.session_state.get("optimization", {})
    if not opt:
        st.info("Run optimization first.")
        return

    models = st.session_state.models
    factors = exp["factors"]
    fname_list = [f["name"] for f in factors]
    x_opt_coded = opt.get("optimal_coded", {})

    delta = st.slider("Deviation Range (coded units)", 0.05, 0.5, 0.2, 0.05)

    with st.spinner("Computing perturbation..."):
        pert = perturbation_analysis(models, x_opt_coded, fname_list, delta=delta)

    for rname in models:
        st.markdown(f"**{rname}**")
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for fi, fname in enumerate(fname_list):
            steps = pert[fname]["steps"]
            preds = pert[fname]["predictions"].get(rname, [])
            fig.add_trace(go.Scatter(
                x=steps, y=preds, mode="lines+markers",
                name=fname, line=dict(color=colors[fi % len(colors)], width=2),
                hovertemplate=f"{fname}<br>Δ=%{{x:.3f}}<br>{rname}=%{{y:.4f}}<extra></extra>",
            ))
        fig.add_vline(x=0, line_dash="dash", line_color="#374151")
        fig.update_layout(template=PLOTLY_TEMPLATE, height=350,
                          xaxis_title="Deviation from Optimum (coded)", yaxis_title=rname,
                          legend_title="Factor")
        st.plotly_chart(fig, use_container_width=True)
