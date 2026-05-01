"""Visualization — response surface, contour, sweet spot, main effects."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.stats import predict_response, build_model_matrix
from utils.doe_engine import coded_to_natural

PLOTLY_TEMPLATE = "plotly_white"
PRIMARY = "#1e40af"
COLORSCALE = "RdYlGn"
N_GRID = 40


def render_visualization():
    st.title("📈 Visualization")

    exp = st.session_state.experiment
    if not exp.get("name"):
        st.warning("No experiment found. Please complete the Design Wizard.")
        return
    if not st.session_state.get("models"):
        st.warning("No fitted models. Complete the Analysis step first.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 Response Surface", "🗺️ Contour / Sweet Spot",
        "📊 Main Effects", "🔗 Interaction Plots"
    ])
    with tab1:
        _tab_surface(exp)
    with tab2:
        _tab_contour(exp)
    with tab3:
        _tab_main_effects(exp)
    with tab4:
        _tab_interactions(exp)


def _factor_sliders(exp, exclude1=None, exclude2=None, suffix=""):
    """Render sliders for all factors except excluded ones. Return dict of coded values."""
    factors = exp["factors"]
    vals = {}
    fixed = [f for f in factors if f["name"] not in (exclude1, exclude2)]
    if not fixed:
        return vals
    st.markdown("**Fix remaining factors:**")
    cols = st.columns(min(4, len(fixed)))
    for i, f in enumerate(fixed):
        with cols[i % len(cols)]:
            lo, hi = f["low"], f["high"]
            mid = (lo + hi) / 2.0
            nat_val = st.slider(f"{f['name']}", lo, hi, mid, key=f"fix_{f['name']}_{suffix}")
            coded = 2 * (nat_val - lo) / (hi - lo) - 1
            vals[f["name"]] = coded
    return vals


def _build_grid_X(exp, xname, yname, fixed_coded, n=N_GRID):
    """Build a grid of coded X values varying xname and yname."""
    factors = exp["factors"]
    fname_list = [f["name"] for f in factors]
    x_vals = np.linspace(-1, 1, n)
    y_vals = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(x_vals, y_vals)
    n2 = n * n
    X_grid = np.zeros((n2, len(fname_list)))
    for i, fname in enumerate(fname_list):
        if fname == xname:
            X_grid[:, i] = xx.ravel()
        elif fname == yname:
            X_grid[:, i] = yy.ravel()
        else:
            X_grid[:, i] = fixed_coded.get(fname, 0.0)
    return xx, yy, X_grid


def _natural_ticks(factor, n=5):
    lo, hi = factor["low"], factor["high"]
    coded_ticks = np.linspace(-1, 1, n)
    nat_ticks = lo + (coded_ticks + 1) / 2 * (hi - lo)
    return coded_ticks.tolist(), [f"{v:.3g}" for v in nat_ticks]


# ─── Response Surface ─────────────────────────────────────────────────────────

def _tab_surface(exp):
    st.subheader("3D Response Surface")
    factors = exp["factors"]
    fname_list = [f["name"] for f in factors]
    models = st.session_state.models

    c1, c2, c3 = st.columns(3)
    with c1:
        rname = st.selectbox("Response", list(models.keys()), key="surf_resp")
    with c2:
        xf = st.selectbox("X-axis Factor", fname_list, index=0, key="surf_x")
    with c3:
        yfnames = [f for f in fname_list if f != xf]
        yf = st.selectbox("Y-axis Factor", yfnames, index=0, key="surf_y")

    fixed = _factor_sliders(exp, exclude1=xf, exclude2=yf, suffix="surf")

    model = models[rname]
    xx, yy, X_grid = _build_grid_X(exp, xf, yf, fixed)
    try:
        zz = predict_response(model, X_grid).reshape(N_GRID, N_GRID)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return

    xfactor = next(f for f in factors if f["name"] == xf)
    yfactor = next(f for f in factors if f["name"] == yf)
    x_coded_ticks, x_nat_labels = _natural_ticks(xfactor)
    y_coded_ticks, y_nat_labels = _natural_ticks(yfactor)

    fig = go.Figure(go.Surface(
        x=np.linspace(-1, 1, N_GRID),
        y=np.linspace(-1, 1, N_GRID),
        z=zz,
        colorscale=COLORSCALE,
        colorbar=dict(title=rname),
        hovertemplate=(f"{xf}: %{{x:.3f}}<br>{yf}: %{{y:.3f}}<br>"
                       f"{rname}: %{{z:.4f}}<extra></extra>"),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=550,
        scene=dict(
            xaxis=dict(title=xf, tickvals=x_coded_ticks, ticktext=x_nat_labels),
            yaxis=dict(title=yf, tickvals=y_coded_ticks, ticktext=y_nat_labels),
            zaxis=dict(title=rname),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Contour / Sweet Spot ─────────────────────────────────────────────────────

def _tab_contour(exp):
    st.subheader("Contour & Sweet Spot Plots")
    factors = exp["factors"]
    fname_list = [f["name"] for f in factors]
    models = st.session_state.models

    c1, c2 = st.columns(2)
    with c1:
        xf = st.selectbox("X-axis Factor", fname_list, index=0, key="cont_x")
    with c2:
        yfnames = [f for f in fname_list if f != xf]
        yf = st.selectbox("Y-axis Factor", yfnames, index=0, key="cont_y")

    fixed = _factor_sliders(exp, exclude1=xf, exclude2=yf, suffix="cont")

    plot_type = st.radio("Plot type", ["Individual Responses", "Sweet Spot (overlay)"],
                         horizontal=True, label_visibility="collapsed")

    xfactor = next(f for f in factors if f["name"] == xf)
    yfactor = next(f for f in factors if f["name"] == yf)
    x_nat = np.linspace(xfactor["low"], xfactor["high"], N_GRID)
    y_nat = np.linspace(yfactor["low"], yfactor["high"], N_GRID)

    xx, yy, X_grid = _build_grid_X(exp, xf, yf, fixed)

    if plot_type == "Individual Responses":
        sel_responses = st.multiselect("Show responses", list(models.keys()),
                                       default=list(models.keys())[:2])
        if not sel_responses:
            return
        ncols = min(2, len(sel_responses))
        nrows = (len(sel_responses) + ncols - 1) // ncols
        fig = make_subplots(rows=nrows, cols=ncols,
                            subplot_titles=sel_responses)
        for idx, rname in enumerate(sel_responses):
            row, col = divmod(idx, ncols)
            zz = predict_response(models[rname], X_grid).reshape(N_GRID, N_GRID)
            fig.add_trace(go.Contour(
                x=x_nat, y=y_nat, z=zz,
                colorscale=COLORSCALE, showscale=True,
                colorbar=dict(title=rname, x=1 + col * 0.05),
                hovertemplate=f"{xf}: %{{x:.3g}}<br>{yf}: %{{y:.3g}}<br>{rname}: %{{z:.4f}}<extra></extra>",
            ), row=row + 1, col=col + 1)
        fig.update_layout(template=PLOTLY_TEMPLATE, height=420 * nrows)
        for i in range(1, nrows + 1):
            for j in range(1, ncols + 1):
                fig.update_xaxes(title_text=xf, row=i, col=j)
                fig.update_yaxes(title_text=yf, row=i, col=j)
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Sweet spot: overlay all responses within acceptable limits
        opt = st.session_state.get("optimization", {})
        resp_configs = opt.get("response_configs", {})
        if not resp_configs:
            st.info("Configure desirability limits in **Optimization → Configure & Run** to see sweet spot.")
            return

        from utils.stats import single_desirability, overall_desirability
        overall = np.zeros((N_GRID, N_GRID))
        for i in range(N_GRID * N_GRID):
            y_vals = {rname: float(predict_response(m, X_grid[i:i+1]))
                      for rname, m in models.items() if rname in resp_configs}
            imp = {r: exp_r.get("importance", 3)
                   for r in resp_configs
                   for exp_r in exp.get("responses", [])
                   if exp_r["name"] == r}
            overall.ravel()[i] = overall_desirability(y_vals, resp_configs, imp)
        overall = overall.reshape(N_GRID, N_GRID)

        fig = go.Figure(go.Contour(
            x=x_nat, y=y_nat, z=overall,
            colorscale="RdYlGn", zmin=0, zmax=1,
            contours=dict(showlabels=True, labelfont=dict(size=11)),
            colorbar=dict(title="Desirability"),
            hovertemplate=f"{xf}: %{{x:.3g}}<br>{yf}: %{{y:.3g}}<br>D=%{{z:.3f}}<extra></extra>",
        ))
        # Mark optimal if available
        opt_nat = opt.get("optimal_natural", {})
        if xf in opt_nat and yf in opt_nat:
            fig.add_trace(go.Scatter(
                x=[opt_nat[xf]], y=[opt_nat[yf]], mode="markers",
                marker=dict(color="white", size=14, symbol="star",
                            line=dict(color=PRIMARY, width=2)),
                name="Optimum", showlegend=True,
            ))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=480,
                          xaxis_title=xf, yaxis_title=yf,
                          title="Sweet Spot — Overall Desirability")
        st.plotly_chart(fig, use_container_width=True)


# ─── Main Effects ─────────────────────────────────────────────────────────────

def _tab_main_effects(exp):
    st.subheader("Main Effects Plots")
    st.markdown("Shows how each factor independently affects each response.")

    factors = exp["factors"]
    models = st.session_state.models
    fname_list = [f["name"] for f in factors]

    rname = st.selectbox("Response", list(models.keys()), key="me_resp")
    model = models[rname]

    n_pts = 30
    ncols = min(3, len(factors))
    nrows = (len(factors) + ncols - 1) // ncols
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=fname_list)

    colors = px.colors.qualitative.Plotly
    for idx, f in enumerate(factors):
        row, col = divmod(idx, ncols)
        coded_range = np.linspace(-1, 1, n_pts)
        X_base = np.zeros((n_pts, len(factors)))
        fi = fname_list.index(f["name"])
        X_base[:, fi] = coded_range
        y_pred = predict_response(model, X_base)
        nat_range = f["low"] + (coded_range + 1) / 2 * (f["high"] - f["low"])
        fig.add_trace(go.Scatter(
            x=nat_range, y=y_pred, mode="lines",
            line=dict(color=colors[idx % len(colors)], width=2),
            hovertemplate=f"{f['name']}: %{{x:.3g}}<br>{rname}: %{{y:.4f}}<extra></extra>",
            showlegend=False,
        ), row=row + 1, col=col + 1)

    fig.update_layout(template=PLOTLY_TEMPLATE,
                      height=350 * nrows, title=f"Main Effects — {rname}")
    for i in range(1, nrows + 1):
        for j in range(1, ncols + 1):
            fig.update_yaxes(title_text=rname, row=i, col=j)
    st.plotly_chart(fig, use_container_width=True)


# ─── Interaction Plots ────────────────────────────────────────────────────────

def _tab_interactions(exp):
    st.subheader("Interaction Plots")
    st.markdown("Shows how pairs of factors interact in their effect on the response.")

    factors = exp["factors"]
    models = st.session_state.models
    fname_list = [f["name"] for f in factors]

    if len(factors) < 2:
        st.info("Need at least 2 factors for interaction plots.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        rname = st.selectbox("Response", list(models.keys()), key="int_resp")
    with c2:
        xf = st.selectbox("Factor 1 (X-axis)", fname_list, index=0, key="int_x")
    with c3:
        yfnames = [f for f in fname_list if f != xf]
        yf = st.selectbox("Factor 2 (Lines)", yfnames, index=0, key="int_y")

    model = models[rname]
    xfactor = next(f for f in factors if f["name"] == xf)
    yfactor = next(f for f in factors if f["name"] == yf)
    fi_x = fname_list.index(xf)
    fi_y = fname_list.index(yf)

    n_pts = 40
    coded_x = np.linspace(-1, 1, n_pts)
    nat_x = xfactor["low"] + (coded_x + 1) / 2 * (xfactor["high"] - xfactor["low"])

    fig = go.Figure()
    levels = [-1, 0, 1]
    nat_levels = [yfactor["low"], (yfactor["low"] + yfactor["high"]) / 2, yfactor["high"]]
    labels = [f"{yf} = {v:.3g}" for v in nat_levels]
    colors = [px.colors.qualitative.Plotly[0], px.colors.qualitative.Plotly[1],
              px.colors.qualitative.Plotly[2]]

    for level, label, color in zip(levels, labels, colors):
        X_int = np.zeros((n_pts, len(factors)))
        X_int[:, fi_x] = coded_x
        X_int[:, fi_y] = level
        y_pred = predict_response(model, X_int)
        fig.add_trace(go.Scatter(
            x=nat_x, y=y_pred, mode="lines",
            name=label, line=dict(color=color, width=2),
            hovertemplate=f"{xf}: %{{x:.3g}}<br>{rname}: %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(template=PLOTLY_TEMPLATE, height=420,
                      xaxis_title=xf, yaxis_title=rname,
                      legend_title=yf,
                      title=f"Interaction: {xf} × {yf} on {rname}")
    st.plotly_chart(fig, use_container_width=True)

    # Interaction significance note
    if "MLR" in st.session_state.models.get(rname, {}).get("type", ""):
        coef = st.session_state.models[rname]["coefficients"]
        int_term = coef[coef["Term"].str.contains(f"{xf}.*{yf}|{yf}.*{xf}", regex=True)]
        if not int_term.empty:
            p = float(int_term["p-value"].values[0])
            sig = "✅ Significant" if p < 0.05 else "⚪ Not significant"
            st.caption(f"Interaction term p-value: {p:.4f} — {sig}")
