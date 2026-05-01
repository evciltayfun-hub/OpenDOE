"""AI Insights — Claude-powered experiment interpretation."""
import streamlit as st
import os

SYSTEM_PROMPT = """You are an expert DOE statistician and process optimization consultant with 20+ years experience in pharmaceutical, chemical, and food industries. You analyze Design of Experiments results and provide clear, actionable insights.

Your responses should be:
- Direct and actionable (engineer-friendly, not academic)
- Structured with clear sections
- Honest about limitations and data quality
- Specific — mention factor names, response names, actual numbers

Format your response in markdown with sections."""

def _build_context(exp, models, optimization) -> str:
    """Build a rich context string from the experiment state."""
    lines = []
    lines.append(f"# Experiment: {exp.get('name', 'Unnamed')}")
    lines.append(f"Objective: {exp.get('objective', 'Not specified')}")
    lines.append(f"Design: {exp.get('design_type_name', 'Unknown')}")
    lines.append("")

    # Factors
    lines.append("## Factors")
    for f in exp.get("factors", []):
        lines.append(f"- **{f['name']}**: {f['low']} to {f['high']} {f.get('unit', '')}")

    # Responses
    lines.append("\n## Response Goals")
    for r in exp.get("responses", []):
        goal = r.get("goal", "—")
        limits = f"[{r.get('lower_limit', '—')}, {r.get('upper_limit', '—')}]"
        lines.append(f"- **{r['name']}**: {goal} | Limits: {limits} | Importance: {r.get('importance', 3)}/5")

    # Models
    lines.append("\n## Model Results")
    for rname, m in models.items():
        lines.append(f"\n### {rname} ({m['type']})")
        lines.append(f"- R² = {m['r2']:.3f}, R²adj = {m['r2_adj']:.3f}, Q² = {m['q2']:.3f}")
        lines.append(f"- RMSE = {m['rmse']:.4f}")
        if m["type"] == "MLR":
            coef = m["coefficients"]
            sig = coef[coef["Significant"] & (coef["Term"] != "Intercept")]
            insig = coef[~coef["Significant"] & (coef["Term"] != "Intercept")]
            if not sig.empty:
                sig_list = [f"{row['Term']} (coeff={row['Coefficient']:.3f}, p={row['p-value']:.4f})"
                            for _, row in sig.iterrows()]
                lines.append(f"- **Significant terms (p<0.05):** {', '.join(sig_list)}")
            if not insig.empty:
                insig_list = [row["Term"] for _, row in insig.iterrows()]
                lines.append(f"- *Non-significant terms:* {', '.join(insig_list)}")
        else:
            vip = m.get("vip")
            if vip is not None:
                imp_factors = vip[vip["VIP"] >= 1.0]["Factor"].tolist()
                lines.append(f"- **Important factors (VIP≥1):** {', '.join(imp_factors) or 'None'}")
            lines.append(f"- PLS components: {m.get('n_components', '—')}")

    # Optimization
    if optimization:
        lines.append("\n## Optimization Results")
        d = optimization.get("desirability", 0)
        lines.append(f"- Overall desirability: {d:.3f}")
        opt_nat = optimization.get("optimal_natural", {})
        lines.append("- Optimal factor settings:")
        for name, val in opt_nat.items():
            lines.append(f"  - {name}: {val:.4f}")
        pred = optimization.get("predicted_responses", {})
        if pred:
            lines.append("- Predicted responses at optimum:")
            for name, val in pred.items():
                lines.append(f"  - {name}: {val:.4f}")

    return "\n".join(lines)


def _nav(page):
    st.session_state.page = page
    st.rerun()


def render_ai_insights():
    st.title("🤖 AI Insights")

    exp = st.session_state.experiment
    models = st.session_state.get("models", {})
    optimization = st.session_state.get("optimization", {})

    if not exp.get("name"):
        st.warning("No experiment loaded.")
        if st.button("← Go to Design Wizard", key="ai_nodsgn"):
            _nav("🔬 Design Wizard")
        return

    # Navigation row
    c_back, c_info = st.columns([1, 5])
    with c_back:
        if st.button("← Visualization", use_container_width=True):
            _nav("📈 Visualization")
    with c_info:
        model_status = f"{len(models)} models fitted" if models else "No models yet"
        opt_status = "✓ Optimized" if optimization else "Not optimized"
        st.markdown(
            f"<div style='font-size:0.82rem; color:#64748b; padding-top:6px;'>"
            f"<b style='color:#e2e8f0;'>{exp['name']}</b> &nbsp;·&nbsp; "
            f"{model_status} &nbsp;·&nbsp; {opt_status}"
            f"</div>", unsafe_allow_html=True)

    if not models:
        st.info("AI Insights works best after fitting models. Go to **Analysis** first.")
        if st.button("← Go to Analysis", key="ai_nomdl"):
            _nav("📊 Analysis")

    st.divider()

    # API Key
    with st.expander("🔑 API Key Setup", expanded=not bool(os.environ.get("ANTHROPIC_API_KEY"))):
        st.markdown("Enter your [Anthropic API key](https://console.anthropic.com/) "
                    "to use AI Insights. Your key is stored only in this session.")
        api_key_input = st.text_input("Anthropic API Key", type="password",
                                      placeholder="sk-ant-...",
                                      value=st.session_state.get("anthropic_api_key", ""))
        if api_key_input:
            st.session_state.anthropic_api_key = api_key_input

    api_key = st.session_state.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY", "")

    # Context preview
    context = _build_context(exp, models, optimization)
    with st.expander("📋 Data being sent to Claude"):
        st.text(context)

    # Analysis options
    st.divider()
    analysis_type = st.selectbox("What would you like Claude to analyze?", [
        "Full experiment interpretation (recommended)",
        "Model quality and reliability assessment",
        "Which factors matter most — and why",
        "Optimal conditions interpretation and practical advice",
        "Suggestions for follow-up experiments",
        "Custom question",
    ])

    custom_q = ""
    if analysis_type == "Custom question":
        custom_q = st.text_area("Your question", placeholder="Ask anything about your experiment...")

    if st.button("🔍 Generate AI Analysis", type="primary", disabled=not api_key):
        if not api_key:
            st.error("Please enter your Anthropic API key above.")
            return

        if not models:
            st.warning("No fitted models yet — AI analysis is most useful after fitting models.")

        prompts = {
            "Full experiment interpretation (recommended)":
                "Provide a complete interpretation of this DOE experiment. Cover: "
                "1) Executive summary of findings, "
                "2) Model quality assessment (is the data reliable?), "
                "3) Key significant factors and their effects, "
                "4) Optimal conditions and practical recommendations, "
                "5) Any concerns about the experimental data, "
                "6) Suggested next steps.",

            "Model quality and reliability assessment":
                "Assess the statistical quality of these models. "
                "Interpret R², Q², and RMSE values. "
                "Flag any concerns about model reliability. "
                "Is the model good enough to make process decisions?",

            "Which factors matter most — and why":
                "Analyze which factors have the most significant effects on each response. "
                "Explain the mechanism if you can infer it. "
                "Rank factors by importance and explain practical implications.",

            "Optimal conditions interpretation and practical advice":
                "Interpret the optimal conditions found by the optimizer. "
                "Are these conditions practically achievable? "
                "What is the overall desirability and what does it mean? "
                "Give specific advice for implementing these conditions in production.",

            "Suggestions for follow-up experiments":
                "Based on these results, what follow-up experiments would you recommend? "
                "Should the design space be expanded or narrowed? "
                "Are there missing factor interactions to investigate? "
                "What confirmation runs should be performed?",

            "Custom question": custom_q,
        }

        user_prompt = prompts.get(analysis_type, custom_q)
        if not user_prompt.strip():
            st.error("Please enter a question.")
            return

        full_prompt = f"{context}\n\n---\n\n{user_prompt}"

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)

            with st.spinner("Claude is analyzing your experiment..."):
                response_box = st.empty()
                full_text = ""

                with client.messages.stream(
                    model="claude-opus-4-7",
                    max_tokens=2000,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": full_prompt}],
                ) as stream:
                    for text_chunk in stream.text_stream:
                        full_text += text_chunk
                        response_box.markdown(full_text + "▌")

                response_box.markdown(full_text)

            # Save to history
            if "ai_history" not in st.session_state:
                st.session_state.ai_history = []
            st.session_state.ai_history.append({
                "type": analysis_type,
                "response": full_text,
            })

            st.divider()
            st.download_button("⬇️ Download Analysis", data=full_text,
                               file_name="ai_analysis.md", mime="text/markdown")

        except ImportError:
            st.error("anthropic package not installed. Run: pip install anthropic")
        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                st.error("Invalid API key. Please check your Anthropic API key.")
            else:
                st.error(f"API error: {e}")

    # History
    history = st.session_state.get("ai_history", [])
    if history:
        st.divider()
        with st.expander(f"📚 Previous analyses ({len(history)})"):
            for i, item in enumerate(reversed(history[-5:])):
                st.markdown(f"**{item['type']}**")
                st.markdown(item["response"])
                if i < len(history) - 1:
                    st.divider()
