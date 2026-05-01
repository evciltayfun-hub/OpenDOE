"""Design Wizard — step-by-step DOE design creation."""
import streamlit as st
import pandas as pd
import numpy as np
from utils.doe_engine import (
    DESIGN_CATALOG, generate_design, build_design_dataframe,
    design_to_excel, get_run_count, get_design_recommendations
)

STEPS = ["Experiment Setup", "Define Factors", "Define Responses",
         "Select Design", "Review & Generate"]


def _step_badge(current: int, total: int = len(STEPS)) -> None:
    st.caption(f"Step {current} of {total}")


def render_design_wizard():
    st.title("🔬 Design Wizard")
    st.markdown("Build your experimental design in 5 guided steps.")

    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 1

    # Step indicator
    cols = st.columns(len(STEPS))
    for i, (col, label) in enumerate(zip(cols, STEPS), 1):
        with col:
            if i < st.session_state.wizard_step:
                st.success(f"✓ {label}")
            elif i == st.session_state.wizard_step:
                st.info(f"▶ {label}")
            else:
                st.markdown(f"<span style='color:#9ca3af'>○ {label}</span>", unsafe_allow_html=True)

    st.divider()

    step = st.session_state.wizard_step

    if step == 1:
        _step_experiment_setup()
    elif step == 2:
        _step_define_factors()
    elif step == 3:
        _step_define_responses()
    elif step == 4:
        _step_select_design()
    elif step == 5:
        _step_review_generate()


def _nav(back=True, next_label="Next →", back_label="← Back"):
    c1, _, c2 = st.columns([1, 4, 1])
    with c1:
        if back and st.button(back_label, use_container_width=True):
            st.session_state.wizard_step -= 1
            st.rerun()
    with c2:
        return st.button(next_label, type="primary", use_container_width=True)


# ─── Step 1 ───────────────────────────────────────────────────────────────────

def _step_experiment_setup():
    _step_badge(1)
    st.subheader("Experiment Setup")
    exp = st.session_state.experiment

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Experiment Name *", value=exp.get("name", ""),
                             placeholder="e.g., Reaction Yield Optimization")
    with col2:
        objective = st.selectbox("Primary Objective",
                                 ["Optimization (find best settings)",
                                  "Screening (identify important factors)"],
                                 index=0 if exp.get("objective", "").startswith("O") else 1)

    description = st.text_area("Description (optional)", value=exp.get("description", ""),
                               placeholder="Briefly describe what you are optimizing and why.")

    n_factors = st.number_input("Number of Factors (inputs)", min_value=2, max_value=15,
                                value=max(2, len(exp.get("factors", [])) or 3))
    n_responses = st.number_input("Number of Responses (outputs)", min_value=1, max_value=10,
                                  value=max(1, len(exp.get("responses", [])) or 2))

    if _nav(back=False, next_label="Next →"):
        if not name.strip():
            st.error("Please provide an experiment name.")
            return
        st.session_state.experiment.update({
            "name": name.strip(),
            "objective": objective,
            "description": description,
            "_n_factors": int(n_factors),
            "_n_responses": int(n_responses),
        })
        # Reset factor/response lists if count changed
        cur_f = st.session_state.experiment.get("factors", [])
        cur_r = st.session_state.experiment.get("responses", [])
        if len(cur_f) != int(n_factors):
            st.session_state.experiment["factors"] = []
        if len(cur_r) != int(n_responses):
            st.session_state.experiment["responses"] = []
        st.session_state.wizard_step = 2
        st.rerun()


# ─── Step 2 ───────────────────────────────────────────────────────────────────

def _step_define_factors():
    _step_badge(2)
    st.subheader("Define Factors")
    st.markdown("Specify the range and unit of each controllable factor.")

    exp = st.session_state.experiment
    n = exp.get("_n_factors", 3)
    factors = exp.get("factors", [])
    while len(factors) < n:
        factors.append({"name": f"Factor {len(factors)+1}", "low": 0.0, "high": 1.0,
                        "unit": "", "type": "Continuous"})

    updated = []
    for i in range(n):
        st.markdown(f"**Factor {i+1}**")
        c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 2])
        with c1:
            fname = st.text_input("Name", value=factors[i]["name"],
                                  key=f"fname_{i}", label_visibility="collapsed",
                                  placeholder="Factor name")
        with c2:
            lo = st.number_input("Low", value=float(factors[i]["low"]),
                                 key=f"flo_{i}", label_visibility="collapsed")
        with c3:
            hi = st.number_input("High", value=float(factors[i]["high"]),
                                 key=f"fhi_{i}", label_visibility="collapsed")
        with c4:
            unit = st.text_input("Unit", value=factors[i]["unit"],
                                 key=f"funit_{i}", label_visibility="collapsed",
                                 placeholder="°C, bar, ...")
        with c5:
            ftype = st.selectbox("Type", ["Continuous", "Categorical"],
                                 index=0 if factors[i].get("type", "Continuous") == "Continuous" else 1,
                                 key=f"ftype_{i}", label_visibility="collapsed")
        updated.append({"name": fname, "low": lo, "high": hi, "unit": unit, "type": ftype})

    if _nav(next_label="Next →"):
        errors = []
        for i, f in enumerate(updated):
            if not f["name"].strip():
                errors.append(f"Factor {i+1}: name is required.")
            if f["low"] >= f["high"]:
                errors.append(f"Factor {i+1} ({f['name']}): Low must be < High.")
        if errors:
            for e in errors:
                st.error(e)
            return
        st.session_state.experiment["factors"] = updated
        st.session_state.wizard_step = 3
        st.rerun()


# ─── Step 3 ───────────────────────────────────────────────────────────────────

def _step_define_responses():
    _step_badge(3)
    st.subheader("Define Responses")
    st.markdown("Specify what you measure and what you want to achieve.")

    exp = st.session_state.experiment
    n = exp.get("_n_responses", 2)
    responses = exp.get("responses", [])
    while len(responses) < n:
        responses.append({"name": f"Response {len(responses)+1}", "unit": "",
                          "goal": "Maximize", "lower_limit": None, "upper_limit": None,
                          "target": None, "importance": 3})

    updated = []
    for i in range(n):
        st.markdown(f"**Response {i+1}**")
        c1, c2, c3 = st.columns([3, 2, 2])
        with c1:
            rname = st.text_input("Name", value=responses[i]["name"],
                                  key=f"rname_{i}", label_visibility="collapsed",
                                  placeholder="Response name")
        with c2:
            unit = st.text_input("Unit", value=responses[i].get("unit", ""),
                                 key=f"runit_{i}", label_visibility="collapsed",
                                 placeholder="%, g/L, ...")
        with c3:
            goal = st.selectbox("Goal", ["Maximize", "Minimize", "Target"],
                                index=["Maximize", "Minimize", "Target"].index(
                                    responses[i].get("goal", "Maximize")),
                                key=f"rgoal_{i}", label_visibility="collapsed")

        c4, c5, c6, c7 = st.columns(4)
        with c4:
            lo_def = responses[i].get("lower_limit") or ""
            lo_str = st.text_input("Lower Limit", value=str(lo_def) if lo_def != "" else "",
                                   key=f"rlo_{i}", placeholder="Optional")
        with c5:
            hi_def = responses[i].get("upper_limit") or ""
            hi_str = st.text_input("Upper Limit", value=str(hi_def) if hi_def != "" else "",
                                   key=f"rhi_{i}", placeholder="Optional")
        with c6:
            if goal == "Target":
                t_def = responses[i].get("target") or ""
                t_str = st.text_input("Target Value", value=str(t_def) if t_def != "" else "",
                                      key=f"rt_{i}", placeholder="Required")
            else:
                t_str = ""
                st.empty()
        with c7:
            imp = st.slider("Importance", 1, 5,
                            value=int(responses[i].get("importance", 3)),
                            key=f"rimp_{i}")

        def _to_float(s):
            try:
                return float(s)
            except (ValueError, TypeError):
                return None

        updated.append({
            "name": rname, "unit": unit, "goal": goal,
            "lower_limit": _to_float(lo_str),
            "upper_limit": _to_float(hi_str),
            "target": _to_float(t_str) if goal == "Target" else None,
            "importance": imp,
        })

        st.divider() if i < n - 1 else None

    if _nav(next_label="Next →"):
        errors = []
        for i, r in enumerate(updated):
            if not r["name"].strip():
                errors.append(f"Response {i+1}: name is required.")
            if r["goal"] == "Target" and r["target"] is None:
                errors.append(f"Response {i+1} ({r['name']}): target value required.")
        if errors:
            for e in errors:
                st.error(e)
            return
        st.session_state.experiment["responses"] = updated
        st.session_state.wizard_step = 4
        st.rerun()


# ─── Step 4 ───────────────────────────────────────────────────────────────────

def _step_select_design():
    _step_badge(4)
    st.subheader("Select Design Type")

    exp = st.session_state.experiment
    n_factors = len(exp["factors"])
    objective_raw = exp.get("objective", "Optimization")
    objective = "optimization" if "Optimiz" in objective_raw else "screening"

    recommended = get_design_recommendations(n_factors, objective)
    st.info(f"💡 For **{n_factors} factors** and **{objective}** objective, "
            f"we recommend: {' or '.join(recommended)}")

    design_choice = st.session_state.experiment.get("design_type_name", recommended[0])

    available = {k: v for k, v in DESIGN_CATALOG.items()
                 if v["min_factors"] <= n_factors <= v["max_factors"]}

    cols = st.columns(min(3, len(available)))
    selected = design_choice

    for idx, (dname, dinfo) in enumerate(available.items()):
        col = cols[idx % len(cols)]
        with col:
            run_count = get_run_count(dinfo["key"], n_factors)
            border = "2px solid #1e40af" if dname == selected else "1px solid #e5e7eb"
            bg = "#eff6ff" if dname == selected else "#ffffff"
            rec_badge = " ⭐" if dname in recommended else ""
            st.markdown(f"""
            <div style="border:{border}; border-radius:8px; padding:14px; background:{bg};
                        margin-bottom:8px; cursor:pointer;">
                <b>{dinfo['icon']} {dname}{rec_badge}</b><br>
                <span style="color:#6b7280; font-size:0.85em;">{dinfo['type']}</span><br>
                <span style="font-size:0.9em;">🔢 <b>{run_count}</b> runs</span><br>
                <span style="font-size:0.85em; color:#374151;">{dinfo['description']}</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Select", key=f"sel_{dname}", use_container_width=True):
                selected = dname
                st.session_state.experiment["design_type_name"] = dname
                st.rerun()

    st.divider()
    if selected:
        dinfo = DESIGN_CATALOG[selected]
        c1, c2 = st.columns(2)
        with c1:
            center_pts = st.number_input("Center Points", min_value=0, max_value=10, value=3)
        with c2:
            randomize = st.checkbox("Randomize Run Order", value=True)

        ccd_face = "ccf"
        if dinfo["key"] == "ccd":
            ccd_face = st.selectbox("CCD Variant",
                                    ["ccf (Face-Centered)", "ccc (Circumscribed)", "cci (Inscribed)"],
                                    index=0).split()[0]

    if _nav(next_label="Generate Design →"):
        if not selected:
            st.error("Please select a design type.")
            return
        st.session_state.experiment.update({
            "design_type_name": selected,
            "design_key": DESIGN_CATALOG[selected]["key"],
            "center_points": int(center_pts),
            "randomize": randomize,
            "ccd_face": ccd_face if dinfo["key"] == "ccd" else "ccf",
        })
        # Generate design
        try:
            factors = st.session_state.experiment["factors"]
            factor_ranges = [(f["low"], f["high"]) for f in factors]
            factor_names = [f["name"] for f in factors]
            design = generate_design(
                DESIGN_CATALOG[selected]["key"],
                n_factors=len(factors),
                center_points=int(center_pts),
                ccd_face=ccd_face if dinfo["key"] == "ccd" else "ccf",
                randomize=randomize,
            )
            coded_df, natural_df = build_design_dataframe(design, factor_names, factor_ranges)
            st.session_state.experiment["design_matrix"] = coded_df
            st.session_state.experiment["natural_matrix"] = natural_df
            # Reset downstream
            st.session_state.experiment["response_data"] = None
            st.session_state.models = {}
            st.session_state.optimization = {}
            st.session_state.wizard_step = 5
            st.rerun()
        except Exception as e:
            st.error(f"Design generation failed: {e}")


# ─── Step 5 ───────────────────────────────────────────────────────────────────

def _step_review_generate():
    _step_badge(5)
    exp = st.session_state.experiment
    st.subheader(f"✅ Design Ready: {exp['name']}")

    natural_df = exp.get("natural_matrix")
    if natural_df is None:
        st.error("Design matrix not found. Please go back and regenerate.")
        return

    factors = exp["factors"]
    responses = exp["responses"]
    n_runs = len(natural_df)
    n_factors = len(factors)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Design", exp.get("design_type_name", "—"))
    c2.metric("Experimental Runs", n_runs)
    c3.metric("Factors", n_factors)
    c4.metric("Responses", len(responses))

    st.divider()

    tab1, tab2 = st.tabs(["📋 Design Matrix", "📊 Factor Summary"])
    with tab1:
        st.markdown("**Experimental runs to perform (natural units):**")
        display_df = natural_df.copy()
        for f in factors:
            if f["unit"]:
                display_df.rename(columns={f["name"]: f"{f['name']} ({f['unit']})"}, inplace=True)
        st.dataframe(display_df, use_container_width=True, height=400)

    with tab2:
        fdata = pd.DataFrame([{
            "Factor": f["name"], "Low": f["low"], "High": f["high"],
            "Unit": f["unit"], "Type": f["type"]
        } for f in factors])
        st.dataframe(fdata, use_container_width=True, hide_index=True)

    st.divider()
    response_names = [r["name"] for r in responses]
    excel_bytes = design_to_excel(natural_df, response_names)
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        st.download_button("⬇️ Download Excel Template", data=excel_bytes,
                           file_name=f"{exp['name'].replace(' ', '_')}_design.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    with c2:
        csv = natural_df.to_csv(index=True)
        st.download_button("⬇️ Download CSV", data=csv,
                           file_name=f"{exp['name'].replace(' ', '_')}_design.csv",
                           mime="text/csv", use_container_width=True)
    with c3:
        if st.button("▶ Go to Analysis", type="primary", use_container_width=True):
            st.session_state.page = "📊 Analysis"
            st.rerun()

    if st.button("← Back", use_container_width=False):
        st.session_state.wizard_step = 4
        st.rerun()

    st.info("💡 **Next:** Fill in your response measurements in the Excel file, "
            "then upload it in the **Analysis** tab.")
