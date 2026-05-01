"""Visualization — response surface, contour, sweet spot, main effects.
   Chart style inspired by MODDE-style DOE reports.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.stats import predict_response
from utils.doe_engine import coded_to_natural

PLOTLY_TEMPLATE = "plotly_white"
# MODDE-style: blue-cyan-green-yellow-orange-red
MODDE_COLORSCALE = [
    [0.00, "#1a3c8f"],
    [0.15, "#2f6abf"],
    [0.30, "#3ea4d4"],
    [0.45, "#70c9a0"],
    [0.60, "#c8e04a"],
    [0.75, "#f0b030"],
    [0.88, "#e85010"],
    [1.00, "#c01000"],
]
N_GRID = 50


def render_visualization():
    st.title("📈 Visualization")
    st.markdown("Interactive response surface plots, contour maps, and design space exploration.")

    exp = st.session_state.experiment
    if not exp.get("name"):
        st.warning("No experiment found. Please complete the Design Wizard.")
        return
    if not st.session_state.get("models"):
        st.warning("No fitted models. Complete the Analysis step first.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 Response Surface (3D)",
        "🗺️ Contour / Sweet Spot",
        "📊 Main Effects",
        "🔗 Interaction Plots",
    ])
    with tab1:
        _tab_surface(exp)
    with tab2:
        _tab_contour(exp)
    with tab3:
        _tab_main_effects(exp)
    with tab4:
        _tab_interactions(exp)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _factor_sliders(exp, exclude1=None, exclude2=None, suffix=""):
    """Render sliders for fixed factors. Return dict of coded values."""
    factors = exp["factors"]
    fixed_factors = [f for f in factors if f["name"] not in (exclude1, exclude2)]
    if not fixed_factors:
        return {}
    with st.expander(f"📌 Fix remaining {len(fixed_factors)} factor(s)", expanded=True):
        vals = {}
        cols = st.columns(min(4, len(fixed_factors)))
        for i, f in enumerate(fixed_factors):
            lo, hi = f["low"], f["high"]
            mid = (lo + hi) / 2.0
            unit_label = f" ({f['unit']})" if f.get("unit") else ""
            with cols[i % len(cols)]:
                nat_val = st.slider(f"{f['name']}{unit_label}", lo, hi, mid,
                                    key=f"fix_{f['name']}_{suffix}",
                                    format="%.3g")
            coded = 2 * (nat_val - lo) / (hi - lo) - 1
            vals[f["name"]] = coded
    return vals


def _build_grid_X(exp, xname, yname, fixed_coded, n=N_GRID):
    factors = exp["factors"]
    fname_list = [f["name"] for f in factors]
    x_coded = np.linspace(-1, 1, n)
    y_coded = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(x_coded, y_coded)
    X_grid = np.zeros((n * n, len(fname_list)))
    for i, fname in enumerate(fname_list):
        if fname == xname:
            X_grid[:, i] = xx.ravel()
        elif fname == yname:
            X_grid[:, i] = yy.ravel()
        else:
            X_grid[:, i] = fixed_coded.get(fname, 0.0)
    return xx, yy, X_grid


def _nat_axis(factor, n=6):
    lo, hi = factor["low"], factor["high"]
    coded_ticks = np.linspace(-1, 1, n)
    nat_ticks   = lo + (coded_ticks + 1) / 2 * (hi - lo)
    unit_label  = f" ({factor['unit']})" if factor.get("unit") else ""
    return (coded_ticks.tolist(),
            [f"{v:.3g}" for v in nat_ticks],
            f"{factor['name']}{unit_label}")


def _nat_range(factor, n=N_GRID):
    return np.linspace(factor["low"], factor["high"], n)


# ─── Tab 1: 3D Response Surface ───────────────────────────────────────────────

def _tab_surface(exp):
    st.subheader("3D Response Surface")
    factors    = exp["factors"]
    fname_list = [f["name"] for f in factors]
    models     = st.session_state.models

    c1, c2, c3 = st.columns(3)
    with c1: rname = st.selectbox("Response", list(models.keys()), key="surf_resp")
    with c2: xf = st.selectbox("X-axis Factor", fname_list, index=0, key="surf_x")
    with c3:
        yf_opts = [f for f in fname_list if f != xf]
        yf = st.selectbox("Y-axis Factor", yf_opts, index=0, key="surf_y")

    fixed = _factor_sliders(exp, exclude1=xf, exclude2=yf, suffix="surf")

    xfac = next(f for f in factors if f["name"] == xf)
    yfac = next(f for f in factors if f["name"] == yf)
    xt, xl, xlab = _nat_axis(xfac)
    yt, yl, ylab = _nat_axis(yfac)

    xx, yy, X_grid = _build_grid_X(exp, xf, yf, fixed)
    try:
        zz = predict_response(models[rname], X_grid).reshape(N_GRID, N_GRID)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return

    # Convert coded axes to natural
    x_nat = np.linspace(xfac["low"], xfac["high"], N_GRID)
    y_nat = np.linspace(yfac["low"], yfac["high"], N_GRID)

    fig = go.Figure(go.Surface(
        x=x_nat, y=y_nat, z=zz,
        colorscale=MODDE_COLORSCALE,
        colorbar=dict(title=rname, thickness=18, len=0.7),
        contours=dict(
            x=dict(show=True, color="white", width=1, highlightwidth=2),
            y=dict(show=True, color="white", width=1, highlightwidth=2),
            z=dict(show=True, color="white", width=1),
        ),
        hovertemplate=(f"{xlab}: %{{x:.3g}}<br>{ylab}: %{{y:.3g}}<br>"
                       f"{rname}: %{{z:.4f}}<extra></extra>"),
        opacity=0.95,
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=580,
        scene=dict(
            xaxis=dict(title=xlab, backgroundcolor="#f8fafc"),
            yaxis=dict(title=ylab, backgroundcolor="#f8fafc"),
            zaxis=dict(title=rname, backgroundcolor="#f0f9ff"),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=f"Response Surface — {rname}", font=dict(size=15, color="#0f172a")),
    )
    st.plotly_chart(fig, use_container_width=True)

    # MODDE-style info box
    zmin, zmax = float(zz.min()), float(zz.max())
    st.markdown(f"""
    <div style="background:#f0f9ff; border:1px solid #bae6fd; border-radius:8px;
                padding:12px 18px; font-size:0.85rem; color:#0c4a6e; margin-top:-8px;">
        📊 &nbsp;<b>{rname}</b> range on this surface:
        &nbsp;Min = <b>{zmin:.3f}</b> &nbsp;|&nbsp; Max = <b>{zmax:.3f}</b>
        &nbsp;|&nbsp; Range = <b>{zmax-zmin:.3f}</b>
    </div>
    """, unsafe_allow_html=True)


# ─── Tab 2: Contour / Sweet Spot ──────────────────────────────────────────────

def _tab_contour(exp):
    st.subheader("Contour Plots & Sweet Spot")
    factors    = exp["factors"]
    fname_list = [f["name"] for f in factors]
    models     = st.session_state.models

    c1, c2 = st.columns(2)
    with c1: xf = st.selectbox("X-axis Factor", fname_list, index=0, key="cont_x")
    with c2:
        yf_opts = [f for f in fname_list if f != xf]
        yf = st.selectbox("Y-axis Factor", yf_opts, index=0, key="cont_y")

    fixed = _factor_sliders(exp, exclude1=xf, exclude2=yf, suffix="cont")

    xfac = next(f for f in factors if f["name"] == xf)
    yfac = next(f for f in factors if f["name"] == yf)
    x_nat = _nat_range(xfac)
    y_nat = _nat_range(yfac)
    xlab  = f"{xf}" + (f" ({xfac['unit']})" if xfac.get("unit") else "")
    ylab  = f"{yf}" + (f" ({yfac['unit']})" if yfac.get("unit") else "")

    xx, yy, X_grid = _build_grid_X(exp, xf, yf, fixed)

    plot_type = st.radio("Plot type",
                         ["📉 Individual response contours", "🌟 Sweet Spot (desirability overlay)"],
                         horizontal=True, label_visibility="collapsed")

    # ── Ideal Zone Box ────────────────────────────────────────────────────────
    with st.expander("📦 Ideal Zone Box (optional — draws an operating range on the plot)"):
        st.markdown("Define the acceptable operating range for the X and Y axes to overlay a zone box.")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            box_x0 = st.number_input(f"{xf} — Min", value=xfac["low"],
                                      min_value=xfac["low"], max_value=xfac["high"], key="box_x0")
        with col_b:
            box_x1 = st.number_input(f"{xf} — Max", value=xfac["high"],
                                      min_value=xfac["low"], max_value=xfac["high"], key="box_x1")
        with col_c:
            box_y0 = st.number_input(f"{yf} — Min", value=yfac["low"],
                                      min_value=yfac["low"], max_value=yfac["high"], key="box_y0")
        with col_d:
            box_y1 = st.number_input(f"{yf} — Max", value=yfac["high"],
                                      min_value=yfac["low"], max_value=yfac["high"], key="box_y1")
        show_box  = st.checkbox("Show zone box on plot", value=False, key="show_box")
        box_color = st.color_picker("Box color", "#ff6600", key="box_color")

    def _add_zone_box(fig, row=None, col=None):
        if not show_box or box_x0 >= box_x1 or box_y0 >= box_y1:
            return
        kw = dict(row=row, col=col) if row else {}
        fig.add_shape(type="rect",
                      x0=box_x0, x1=box_x1, y0=box_y0, y1=box_y1,
                      line=dict(color=box_color, width=2.5, dash="dash"),
                      fillcolor=box_color, opacity=0.08, **kw)
        fig.add_annotation(x=(box_x0 + box_x1) / 2, y=box_y1,
                           text="Ideal Zone", showarrow=False,
                           font=dict(size=11, color=box_color, family="Arial Black"),
                           yshift=10, **(dict(row=row, col=col) if row else {}))

    # ── Individual contour plots ───────────────────────────────────────────────
    if "Individual" in plot_type:
        sel_r = st.multiselect("Responses to plot", list(models.keys()),
                               default=list(models.keys()), key="cont_sel")
        if not sel_r:
            return

        n_cols = min(2, len(sel_r))
        n_rows = (len(sel_r) + n_cols - 1) // n_cols
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=sel_r,
                            horizontal_spacing=0.12, vertical_spacing=0.14)

        for idx, rname in enumerate(sel_r):
            r, c = divmod(idx, n_cols)
            zz = predict_response(models[rname], X_grid).reshape(N_GRID, N_GRID)
            fig.add_trace(go.Contour(
                x=x_nat, y=y_nat, z=zz,
                colorscale=MODDE_COLORSCALE,
                showscale=(idx == 0),
                colorbar=dict(title=rname, thickness=16, len=0.8 / n_rows,
                              y=1 - (r + 0.5) / n_rows),
                contours=dict(
                    showlabels=True,
                    labelfont=dict(size=10, color="white", family="Arial"),
                    coloring="heatmap",
                ),
                line=dict(width=1.2, color="rgba(255,255,255,0.6)"),
                hovertemplate=f"{xlab}: %{{x:.3g}}<br>{ylab}: %{{y:.3g}}<br>{rname}: %{{z:.4f}}<extra></extra>",
            ), row=r + 1, col=c + 1)
            _add_zone_box(fig, row=r + 1, col=c + 1)

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=400 * n_rows,
            title=dict(text="Contour Plots", font=dict(size=15)),
        )
        for row in range(1, n_rows + 1):
            for col in range(1, n_cols + 1):
                fig.update_xaxes(title_text=xlab, row=row, col=col)
                fig.update_yaxes(title_text=ylab, row=row, col=col)
        st.plotly_chart(fig, use_container_width=True)

    # ── Sweet Spot ────────────────────────────────────────────────────────────
    else:
        opt = st.session_state.get("optimization", {})
        resp_configs = opt.get("response_configs", {})
        if not resp_configs:
            st.info("💡 Set desirability goals in **Optimization → Configure & Run** first "
                    "to generate the Sweet Spot map.")
            return

        from utils.stats import overall_desirability
        overall = np.zeros(N_GRID * N_GRID)
        resp_list = {rn: models[rn] for rn in resp_configs if rn in models}
        imp_map = {r["name"]: r.get("importance", 3) for r in exp.get("responses", [])}

        for i in range(N_GRID * N_GRID):
            y_vals = {}
            for rname, m in resp_list.items():
                try:
                    raw = predict_response(m, X_grid[i:i+1])
                    y_vals[rname] = float(np.asarray(raw).ravel()[0])
                except Exception:
                    y_vals[rname] = np.nan
            overall[i] = overall_desirability(y_vals, resp_configs, imp_map)
        overall = overall.reshape(N_GRID, N_GRID)

        # Sweet spot = area where desirability > threshold
        threshold = st.slider("Highlight threshold (desirability ≥)", 0.0, 1.0, 0.6, 0.05)

        fig = go.Figure()
        fig.add_trace(go.Contour(
            x=x_nat, y=y_nat, z=overall,
            colorscale=[
                [0.0,  "#f0f4f8"], [0.3,  "#fde68a"],
                [0.6,  "#34d399"], [0.8,  "#059669"],
                [1.0,  "#065f46"],
            ],
            zmin=0, zmax=1,
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color="#0f172a"),
                start=0, end=1, size=0.1,
            ),
            colorbar=dict(title="Desirability", thickness=18),
            hovertemplate=f"{xlab}: %{{x:.3g}}<br>{ylab}: %{{y:.3g}}<br>D=%{{z:.3f}}<extra></extra>",
        ))

        # Highlight sweet spot zone as filled contour above threshold
        sweet = np.where(overall >= threshold, overall, np.nan)
        if not np.all(np.isnan(sweet)):
            fig.add_trace(go.Contour(
                x=x_nat, y=y_nat, z=sweet,
                colorscale=[[0, "rgba(5,150,105,0.25)"], [1, "rgba(5,150,105,0.45)"]],
                showscale=False, showlegend=True, name=f"D ≥ {threshold}",
                contours=dict(type="constraint", operation=">=", value=threshold),
                line=dict(color="#065f46", width=2),
                hoverinfo="skip",
            ))

        # Mark optimum if available
        opt_nat = opt.get("optimal_natural", {})
        if xf in opt_nat and yf in opt_nat:
            fig.add_trace(go.Scatter(
                x=[opt_nat[xf]], y=[opt_nat[yf]], mode="markers+text",
                marker=dict(color="white", size=16, symbol="star",
                            line=dict(color="#1e40af", width=2.5)),
                text=["Optimum"], textposition="top center",
                textfont=dict(color="#1e40af", size=11, family="Arial Black"),
                name="Optimum",
            ))

        _add_zone_box(fig)

        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=520,
            xaxis_title=xlab, yaxis_title=ylab,
            title=dict(text="Sweet Spot — Overall Desirability Map",
                       font=dict(size=15, color="#0f172a")),
            legend=dict(orientation="h", y=-0.12),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        pct_sweet = 100 * np.mean(overall >= threshold)
        st.markdown(f"""
        <div style="background:#f0fdf4; border:1px solid #86efac; border-radius:8px;
                    padding:12px 18px; font-size:0.85rem; color:#14532d;">
            🟢 &nbsp;<b>{pct_sweet:.1f}%</b> of the design space meets
            desirability ≥ {threshold:.2f} — the highlighted region above.
        </div>
        """, unsafe_allow_html=True)


# ─── Tab 3: Main Effects ──────────────────────────────────────────────────────

def _tab_main_effects(exp):
    st.subheader("Main Effects Plots")
    st.markdown("Effect of each factor on the response when all other factors are held at center.")

    factors    = exp["factors"]
    fname_list = [f["name"] for f in factors]
    models     = st.session_state.models

    c1, c2 = st.columns(2)
    with c1:
        rname = st.selectbox("Response", list(models.keys()), key="me_resp")
    with c2:
        sort_by = st.selectbox("Sort by", ["Factor order", "Effect size (absolute)"], key="me_sort")

    model = models[rname]
    n_pts = 40
    effects = {}
    for f in factors:
        coded_range = np.linspace(-1, 1, n_pts)
        X_base = np.zeros((n_pts, len(factors)))
        fi = fname_list.index(f["name"])
        X_base[:, fi] = coded_range
        y_pred = np.asarray(predict_response(model, X_base)).ravel()
        effects[f["name"]] = {"nat": _nat_range(f, n_pts), "pred": y_pred,
                               "delta": float(y_pred.max() - y_pred.min()),
                               "unit": f.get("unit", "")}

    if sort_by == "Effect size (absolute)":
        sorted_factors = sorted(effects.items(), key=lambda kv: kv[1]["delta"], reverse=True)
    else:
        sorted_factors = list(effects.items())

    n_cols = min(3, len(sorted_factors))
    n_rows = (len(sorted_factors) + n_cols - 1) // n_cols
    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=[f"{fn} (Δ={v['delta']:.3f})" for fn, v in sorted_factors],
                        vertical_spacing=0.12, horizontal_spacing=0.1)

    colors = px.colors.qualitative.Set2
    for idx, (fname, v) in enumerate(sorted_factors):
        r, c = divmod(idx, n_cols)
        unit_label = f" ({v['unit']})" if v["unit"] else ""
        fig.add_trace(go.Scatter(
            x=v["nat"], y=v["pred"], mode="lines",
            line=dict(color=colors[idx % len(colors)], width=2.5),
            fill="tozeroy", fillcolor=f"rgba({','.join(str(int(c, 16)) for c in [colors[idx % len(colors)][1:3], colors[idx % len(colors)][3:5], colors[idx % len(colors)][5:7]])},0.08)",
            hovertemplate=f"{fname}: %{{x:.3g}}{unit_label}<br>{rname}: %{{y:.4f}}<extra></extra>",
            showlegend=False,
        ), row=r + 1, col=c + 1)
        # Mark center
        mid_val = (v["nat"][0] + v["nat"][-1]) / 2
        fig.add_vline(x=mid_val, line_dash="dot", line_color="#9ca3af",
                      line_width=1, row=r + 1, col=c + 1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=300 * n_rows,
        title=dict(text=f"Main Effects on {rname}", font=dict(size=15)),
    )
    for row in range(1, n_rows + 1):
        for col in range(1, n_cols + 1):
            fig.update_yaxes(title_text=rname, row=row, col=col)
    st.plotly_chart(fig, use_container_width=True)

    # Effect ranking bar
    st.markdown("#### Effect Size Ranking")
    eff_df = pd.DataFrame([{"Factor": fn, "Effect (Δ)": v["delta"]}
                            for fn, v in sorted_factors]).sort_values("Effect (Δ)", ascending=True)
    fig2 = go.Figure(go.Bar(
        x=eff_df["Effect (Δ)"], y=eff_df["Factor"], orientation="h",
        marker=dict(
            color=eff_df["Effect (Δ)"],
            colorscale=MODDE_COLORSCALE,
            showscale=False,
        ),
        hovertemplate="%{y}: Δ=%{x:.4f}<extra></extra>",
        text=[f"{v:.4f}" for v in eff_df["Effect (Δ)"]],
        textposition="outside",
    ))
    fig2.update_layout(template=PLOTLY_TEMPLATE, height=max(250, 50 * len(eff_df)),
                       xaxis_title=f"Effect Size (max−min of {rname})",
                       title="Factors sorted by absolute effect")
    st.plotly_chart(fig2, use_container_width=True)


# ─── Tab 4: Interaction Plots ─────────────────────────────────────────────────

def _tab_interactions(exp):
    st.subheader("Interaction Plots")
    st.markdown("How two factors interact in their combined effect on the response.")

    factors    = exp["factors"]
    fname_list = [f["name"] for f in factors]
    models     = st.session_state.models

    if len(factors) < 2:
        st.info("Need at least 2 factors.")
        return

    c1, c2, c3 = st.columns(3)
    with c1: rname = st.selectbox("Response", list(models.keys()), key="int_resp")
    with c2: xf = st.selectbox("Factor 1 (X-axis)", fname_list, index=0, key="int_x")
    with c3:
        yf_opts = [f for f in fname_list if f != xf]
        yf = st.selectbox("Factor 2 (Lines)", yf_opts, index=0, key="int_y")

    model  = models[rname]
    xfac   = next(f for f in factors if f["name"] == xf)
    yfac   = next(f for f in factors if f["name"] == yf)
    fi_x   = fname_list.index(xf)
    fi_y   = fname_list.index(yf)
    n_pts  = 50
    coded_x = np.linspace(-1, 1, n_pts)
    nat_x   = _nat_range(xfac, n_pts)
    xlab    = f"{xf}" + (f" ({xfac['unit']})" if xfac.get("unit") else "")

    levels     = [-1, -0.5, 0, 0.5, 1]
    nat_levels = [yfac["low"] + (l + 1) / 2 * (yfac["high"] - yfac["low"]) for l in levels]
    labels     = [f"{yf} = {v:.3g}" + (f" {yfac['unit']}" if yfac.get("unit") else "")
                  for v in nat_levels]
    colors     = ["#1e3a8a", "#2563eb", "#0ea5e9", "#10b981", "#f59e0b"]

    fig = go.Figure()
    for level, label, color in zip(levels, labels, colors):
        X_int = np.zeros((n_pts, len(factors)))
        X_int[:, fi_x] = coded_x
        X_int[:, fi_y] = level
        y_pred = np.asarray(predict_response(model, X_int)).ravel()
        fig.add_trace(go.Scatter(
            x=nat_x, y=y_pred, mode="lines", name=label,
            line=dict(color=color, width=2.2),
            hovertemplate=f"{xlab}: %{{x:.3g}}<br>{rname}: %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=440,
        xaxis_title=xlab, yaxis_title=rname,
        legend_title=yf,
        title=dict(text=f"Interaction: {xf} × {yf}  →  {rname}",
                   font=dict(size=15, color="#0f172a")),
        legend=dict(bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e2e8f0", borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Significance note for MLR
    m = st.session_state.models.get(rname, {})
    if m.get("type") == "MLR":
        coef = m["coefficients"]
        int_term = coef[coef["Term"].str.contains(
            f"{xf}.*{yf}|{yf}.*{xf}", regex=True, na=False)]
        if not int_term.empty:
            p = float(int_term["p-value"].values[0])
            c_val = float(int_term["Coefficient"].values[0])
            sig   = p < 0.05
            color = "#065f46" if sig else "#92400e"
            bg    = "#f0fdf4" if sig else "#fffbeb"
            border = "#86efac" if sig else "#fde68a"
            msg = f"✅ Significant (p = {p:.4f}) — coefficient: {c_val:+.4f}" if sig else \
                  f"⚪ Not significant (p = {p:.4f}) — parallel lines indicate weak interaction"
            st.markdown(f"""
            <div style="background:{bg}; border:1px solid {border}; border-radius:8px;
                        padding:10px 16px; font-size:0.85rem; color:{color}; margin-top:6px;">
                <b>Interaction term {xf} × {yf}:</b> {msg}
            </div>
            """, unsafe_allow_html=True)
