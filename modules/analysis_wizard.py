"""Analysis Wizard — model fitting, ANOVA, diagnostics."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

from utils.stats import fit_mlr, fit_pls

PLOTLY_TEMPLATE = "plotly_dark"
PRIMARY = "#3b82f6"
SUCCESS = "#10b981"
DANGER = "#ef4444"
WARN = "#f59e0b"


def _nav(page):
    st.session_state.page = page
    st.rerun()


def render_analysis():
    st.title("📊 Analysis")

    exp = st.session_state.experiment
    if not exp.get("name"):
        st.warning("No experiment loaded.")
        if st.button("← Go to Design Wizard", key="ana_nodsgn"):
            _nav("🔬 Design Wizard")
        return
    if exp.get("design_matrix") is None:
        st.warning("No design matrix. Please complete the Design Wizard first.")
        if st.button("← Go to Design Wizard", key="ana_nomat"):
            _nav("🔬 Design Wizard")
        return

    # Navigation row
    c_back, c_info, c_fwd = st.columns([1, 4, 1])
    with c_back:
        if st.button("← Design Wizard", use_container_width=True):
            _nav("🔬 Design Wizard")
    with c_info:
        st.markdown(
            f"<div style='font-size:0.82rem; color:#64748b; padding-top:6px;'>"
            f"<b style='color:#e2e8f0;'>{exp['name']}</b> &nbsp;·&nbsp; "
            f"{exp.get('design_type_name','—')} &nbsp;·&nbsp; "
            f"{len(exp['factors'])} factors &nbsp;·&nbsp; {len(exp['responses'])} responses"
            f"</div>", unsafe_allow_html=True)
    with c_fwd:
        if st.button("Optimization →", use_container_width=True,
                     disabled=not bool(st.session_state.get("models"))):
            _nav("🎯 Optimization")

    st.divider()
    tab1, tab2, tab3 = st.tabs(["📥 Enter Data", "⚙️ Fit Models", "📈 Diagnostics"])

    with tab1:
        _tab_enter_data(exp)
    with tab2:
        _tab_fit_models(exp)
    with tab3:
        _tab_diagnostics(exp)


# ─── Tab 1: Data Entry ────────────────────────────────────────────────────────

def _tab_enter_data(exp):
    st.subheader("Enter Response Data")

    natural_df = exp.get("natural_matrix")
    responses = exp.get("responses", [])
    rnames = [r["name"] for r in responses]

    method = st.radio("How would you like to enter data?",
                      ["📝 Edit in table", "📂 Upload Excel/CSV"],
                      horizontal=True, label_visibility="collapsed")

    if method == "📝 Edit in table":
        existing = exp.get("response_data")
        if existing is not None:
            base_df = existing.copy()
        else:
            base_df = natural_df.copy()
            for r in rnames:
                if r not in base_df.columns:
                    base_df[r] = np.nan

        factor_cols = [f["name"] for f in exp["factors"]]
        edit_cols = rnames
        display_df = base_df[[c for c in factor_cols if c in base_df.columns] + edit_cols].copy()

        st.markdown("Fill in the response values (factor columns are read-only reference):")
        col_config = {}
        for r in rnames:
            col_config[r] = st.column_config.NumberColumn(r, required=False)
        for fc in factor_cols:
            if fc in display_df.columns:
                col_config[fc] = st.column_config.NumberColumn(fc, disabled=True)

        edited = st.data_editor(display_df, column_config=col_config,
                                use_container_width=True, key="data_editor")

        if st.button("💾 Save Response Data", type="primary"):
            merged = natural_df.copy()
            for r in rnames:
                if r in edited.columns:
                    merged[r] = edited[r].values
            st.session_state.experiment["response_data"] = merged
            st.session_state.models = {}  # invalidate models
            st.success("Response data saved! Proceed to **Fit Models**.")

    else:
        uploaded = st.file_uploader("Upload your filled design file",
                                    type=["xlsx", "csv", "xls"])
        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    df_up = pd.read_csv(uploaded, index_col=0)
                else:
                    df_up = pd.read_excel(uploaded, index_col=0)

                st.success(f"Loaded {len(df_up)} rows × {len(df_up.columns)} columns")
                st.dataframe(df_up, use_container_width=True)

                if st.button("✅ Use This Data", type="primary"):
                    merged = natural_df.copy()
                    for r in rnames:
                        if r in df_up.columns:
                            merged[r] = df_up[r].values[:len(merged)]
                    st.session_state.experiment["response_data"] = merged
                    st.session_state.models = {}
                    st.success("Data loaded! Proceed to **Fit Models**.")
            except Exception as e:
                st.error(f"Failed to read file: {e}")

    current = exp.get("response_data")
    if current is not None:
        st.divider()
        st.markdown("**Current response data summary:**")
        summary_rows = []
        for r in rnames:
            if r in current.columns and current[r].notna().any():
                vals = current[r].dropna()
                summary_rows.append({
                    "Response": r,
                    "N": len(vals),
                    "Mean": round(vals.mean(), 4),
                    "Std": round(vals.std(), 4),
                    "Min": round(vals.min(), 4),
                    "Max": round(vals.max(), 4),
                })
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ─── Tab 2: Fit Models ────────────────────────────────────────────────────────

def _tab_fit_models(exp):
    st.subheader("Fit Statistical Models")

    resp_data = exp.get("response_data")
    if resp_data is None:
        st.info("Please enter response data in the **Enter Data** tab first.")
        return

    coded_df = exp.get("design_matrix")
    responses = exp.get("responses", [])
    factors = exp.get("factors", [])
    rnames = [r["name"] for r in responses]
    fname_list = [f["name"] for f in factors]

    c1, c2 = st.columns(2)
    with c1:
        model_type = st.selectbox("Model Type", ["MLR (Multiple Linear Regression)",
                                                  "PLS (Partial Least Squares)"],
                                  index=0)
    with c2:
        design_key = exp.get("design_key", "")
        default_degree = "quadratic" if design_key in ("ccd", "box_behnken") else "2fi"
        if "MLR" in model_type:
            degree = st.selectbox("Model Degree",
                                  ["linear", "2fi (Linear + Interactions)", "quadratic (Full RSM)"],
                                  index=["linear", "2fi", "quadratic"].index(default_degree)
                                  if default_degree in ["linear", "2fi", "quadratic"] else 1)
            degree = degree.split()[0]
        else:
            degree = "linear"
            st.info("PLS handles interactions internally via latent variables.")

    target_responses = st.multiselect("Responses to model", rnames, default=rnames)

    if st.button("🚀 Fit Models", type="primary"):
        X_coded = coded_df[fname_list].values
        models = {}
        progress = st.progress(0)
        for i, rname in enumerate(target_responses):
            if rname not in resp_data.columns:
                st.warning(f"No data for {rname}, skipping.")
                continue
            y = resp_data[rname].dropna().values
            X_sub = X_coded[:len(y)]
            if len(y) < 3:
                st.warning(f"{rname}: need at least 3 observations.")
                continue
            try:
                with st.spinner(f"Fitting {rname}..."):
                    if "MLR" in model_type:
                        m = fit_mlr(X_sub, y, degree=degree, factor_names=fname_list)
                    else:
                        m = fit_pls(X_sub, y, factor_names=fname_list)
                models[rname] = m
                st.success(f"✓ {rname}: R²={m['r2']:.3f}, Q²={m['q2']:.3f}")
            except Exception as e:
                st.error(f"{rname}: {e}")
            progress.progress((i + 1) / len(target_responses))
        if models:
            st.session_state.models = models
            st.session_state.optimization = {}
            st.success("✅ All models fitted successfully!")
            if st.button("→ Continue to Optimization", type="primary", key="goto_opt"):
                _nav("🎯 Optimization")

    # Show existing model summary
    if st.session_state.models:
        st.divider()
        st.markdown("### Model Summary")
        rows = []
        for rname, m in st.session_state.models.items():
            r2 = m["r2"]
            q2 = m["q2"]
            rows.append({
                "Response": rname,
                "Type": m["type"],
                "R²": f"{r2:.3f}",
                "R² adj": f"{m['r2_adj']:.3f}",
                "Q²": f"{q2:.3f}",
                "RMSE": f"{m['rmse']:.4f}" if not np.isnan(m["rmse"]) else "—",
                "Status": "✅ Good" if r2 > 0.8 and q2 > 0.5 else
                          ("⚠️ Moderate" if r2 > 0.6 else "❌ Poor"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Model quality thresholds
        with st.expander("ℹ️ Model Quality Guidelines"):
            st.markdown("""
| Metric | Acceptable | Good | Excellent |
|--------|-----------|------|-----------|
| R² | > 0.60 | > 0.80 | > 0.95 |
| Q² | > 0.40 | > 0.60 | > 0.80 |
| R² − Q² | < 0.3 (large gap indicates overfitting) | | |
""")


# ─── Tab 3: Diagnostics ───────────────────────────────────────────────────────

def _tab_diagnostics(exp):
    st.subheader("Model Diagnostics")

    if not st.session_state.models:
        st.info("No models fitted yet — go to the **Fit Models** tab first.")
        return

    rnames = list(st.session_state.models.keys())
    selected_r = st.selectbox("Select Response", rnames)
    m = st.session_state.models[selected_r]

    # Coefficients / VIP
    if m["type"] == "MLR":
        _plot_coefficients(m, selected_r)
    else:
        _plot_vip(m, selected_r)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        _plot_obs_vs_pred(m, selected_r)
    with col2:
        _plot_residuals(m, selected_r)

    st.divider()
    _show_anova(m, selected_r)


def _plot_coefficients(m, rname):
    st.markdown(f"#### Coefficients — {rname}")
    coef = m["coefficients"].copy()
    coef_plot = coef[coef["Term"] != "Intercept"].copy()
    coef_plot = coef_plot.sort_values("Coefficient", key=abs, ascending=True)
    colors = [SUCCESS if sig else "#9ca3af" for sig in coef_plot["Significant"]]

    fig = go.Figure(go.Bar(
        x=coef_plot["Coefficient"],
        y=coef_plot["Term"],
        orientation="h",
        marker_color=colors,
        error_x=dict(type="data", array=1.96 * coef_plot["Std Error"].values,
                     visible=True, color="#6b7280"),
        hovertemplate="%{y}<br>Coeff: %{x:.4f}<br><extra></extra>",
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=max(300, 40 * len(coef_plot)),
                      xaxis_title="Coefficient (coded)", yaxis_title="",
                      margin=dict(l=10, r=10, t=30, b=40),
                      title=f"Coefficients (green = p < 0.05)")
    fig.add_vline(x=0, line_dash="dash", line_color="#374151", line_width=1)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Coefficient table"):
        st.dataframe(coef[coef["Term"] != "Intercept"].style.format({
            "Coefficient": "{:.4f}", "Std Error": "{:.4f}",
            "t-value": "{:.3f}", "p-value": "{:.4f}",
        }), use_container_width=True, hide_index=True)


def _plot_vip(m, rname):
    st.markdown(f"#### VIP Scores — {rname}")
    vip = m["vip"].sort_values("VIP", ascending=True)
    colors = [SUCCESS if v >= 1.0 else "#9ca3af" for v in vip["VIP"]]
    fig = go.Figure(go.Bar(
        x=vip["VIP"], y=vip["Factor"], orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: VIP=%{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=1.0, line_dash="dash", line_color=DANGER,
                  annotation_text="VIP=1.0 threshold")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=max(300, 40 * len(vip)),
                      xaxis_title="VIP Score", title="VIP ≥ 1.0 = important factor")
    st.plotly_chart(fig, use_container_width=True)


def _plot_obs_vs_pred(m, rname):
    y_obs = m["y_obs"]
    y_pred = m["y_pred"]
    lo, hi = min(y_obs.min(), y_pred.min()), max(y_obs.max(), y_pred.max())
    margin = (hi - lo) * 0.05
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_obs, y=y_pred, mode="markers",
                             marker=dict(color=PRIMARY, size=8, opacity=0.8),
                             hovertemplate="Obs: %{x:.3f}<br>Pred: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[lo - margin, hi + margin], y=[lo - margin, hi + margin],
                             mode="lines", line=dict(color="#9ca3af", dash="dash"),
                             showlegend=False))
    fig.update_layout(template=PLOTLY_TEMPLATE,
                      title=f"Observed vs Predicted (R²={m['r2']:.3f})",
                      xaxis_title="Observed", yaxis_title="Predicted",
                      height=380)
    st.plotly_chart(fig, use_container_width=True)


def _plot_residuals(m, rname):
    residuals = m["residuals"]
    y_pred = m["y_pred"]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Residuals vs Fitted", "Normal Q-Q"))

    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                             marker=dict(color=PRIMARY, size=7, opacity=0.8),
                             hovertemplate="Fitted: %{x:.3f}<br>Residual: %{y:.4f}<extra></extra>"),
                  row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af", row=1, col=1)

    sorted_res = np.sort(residuals)
    n = len(sorted_res)
    theoretical = np.array([np.percentile(np.random.randn(10000), 100 * (i - 0.5) / n)
                             for i in range(1, n + 1)])
    fig.add_trace(go.Scatter(x=theoretical, y=sorted_res, mode="markers",
                             marker=dict(color=SUCCESS, size=7, opacity=0.8),
                             hovertemplate="Theoretical: %{x:.3f}<br>Sample: %{y:.4f}<extra></extra>"),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=[theoretical[0], theoretical[-1]],
                             y=[theoretical[0], theoretical[-1]],
                             mode="lines", line=dict(color="#9ca3af", dash="dash"),
                             showlegend=False), row=1, col=2)

    fig.update_layout(template=PLOTLY_TEMPLATE, height=380, showlegend=False,
                      title="Residual Diagnostics")
    st.plotly_chart(fig, use_container_width=True)


def _show_anova(m, rname):
    st.markdown(f"#### ANOVA Table — {rname}")
    anova = m["anova"].copy()
    for col in ["SS", "MS", "F"]:
        anova[col] = anova[col].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    anova["p-value"] = anova["p-value"].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    anova["df"] = anova["df"].apply(lambda v: str(int(v)) if pd.notna(v) else "—")
    st.dataframe(anova, use_container_width=True, hide_index=True)
