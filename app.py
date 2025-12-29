"""Streamlit dashboard for Cookie Cats A/B test analysis."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Import analysis modules
from src.utils import load_data, create_binary_treatment, download_data
from src.classical_ab import difference_in_means, proportions_test, check_sample_ratio_mismatch
from src.causal_methods import (
    regression_adjustment, propensity_score_weighting,
    cuped_estimator, doubly_robust_estimator
)
from src.hte_analysis import subgroup_analysis


st.set_page_config(
    page_title="Cookie Cats A/B Analysis",
    page_icon="🐱",
    layout="wide"
)

st.title("🐱 Cookie Cats A/B Test Analysis")
st.markdown("""
Analysis of the Cookie Cats mobile game A/B test using classical and causal inference methods.
The experiment tested moving the first gate from level 30 to level 40.
""")


@st.cache_data
def get_data():
    try:
        return load_data()
    except FileNotFoundError:
        return None


df = get_data()

if df is None:
    st.error("Dataset not found. Please run `python -m src.utils` to download.")
    st.stop()

df = create_binary_treatment(df)

# Sidebar
st.sidebar.header("Analysis Options")
outcome = st.sidebar.selectbox("Outcome Variable", ["retention_1", "retention_7"])
outcome_label = "1-Day Retention" if outcome == "retention_1" else "7-Day Retention"

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "🔬 Classical A/B", "🎯 Causal Methods", "📈 HTE Analysis"
])

with tab1:
    st.header("Data Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Users", f"{len(df):,}")
    with col2:
        st.metric("Control (gate_30)", f"{(df['version']=='gate_30').sum():,}")
    with col3:
        st.metric("Treatment (gate_40)", f"{(df['version']=='gate_40').sum():,}")
    with col4:
        st.metric(f"{outcome_label}", f"{df[outcome].mean():.2%}")

    st.subheader("Retention by Group")
    retention_by_group = df.groupby("version")[["retention_1", "retention_7"]].mean()

    fig = px.bar(
        retention_by_group.reset_index(),
        x="version",
        y=[outcome],
        barmode="group",
        title=f"{outcome_label} by Treatment Group"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Game Rounds Distribution")
    cap = df["sum_gamerounds"].quantile(0.99)
    df_capped = df[df["sum_gamerounds"] <= cap]

    fig = px.histogram(
        df_capped,
        x="sum_gamerounds",
        color="version",
        nbins=50,
        opacity=0.7,
        title="Distribution of Game Rounds (capped at 99th percentile)"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Classical A/B Test Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Difference in Means")
        dim = difference_in_means(df, outcome)

        st.metric("Average Treatment Effect", f"{dim['ate']:.4f}")
        st.metric("Relative Effect", f"{dim['relative_effect']:.2%}")
        st.write(f"**95% CI:** [{dim['ci_lower']:.4f}, {dim['ci_upper']:.4f}]")

        # Visualize
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Control (gate_30)", "Treatment (gate_40)"],
            y=[dim["mean_control"], dim["mean_treatment"]],
            error_y=dict(type="constant", value=dim["se"])
        ))
        fig.update_layout(title=f"Mean {outcome_label} by Group")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Statistical Test")
        prop_test = proportions_test(df, outcome)

        st.metric("Z-Statistic", f"{prop_test['z_statistic']:.4f}")
        st.metric("P-Value", f"{prop_test['p_value']:.4f}")

        if prop_test["significant_at_05"]:
            st.success("✅ Statistically significant at α = 0.05")
        else:
            st.warning("⚠️ Not statistically significant at α = 0.05")

        st.subheader("Sample Ratio Mismatch")
        srm = check_sample_ratio_mismatch(df)
        st.write(f"Observed ratio: {srm['observed_ratio']:.4f}")
        st.write(f"Expected ratio: {srm['expected_ratio']:.4f}")

        if srm["srm_detected"]:
            st.error("🚨 SRM detected!")
        else:
            st.success("✅ No SRM detected")

with tab3:
    st.header("Causal Inference Methods")

    methods = {
        "Regression Adjustment": regression_adjustment(df, outcome),
        "IPW": propensity_score_weighting(df, outcome),
        "CUPED": cuped_estimator(df, outcome),
        "Doubly Robust": doubly_robust_estimator(df, outcome),
    }

    # Summary table
    summary_data = []
    for method, result in methods.items():
        summary_data.append({
            "Method": method,
            "ATE": result["ate"],
            "SE": result["se"],
            "CI Lower": result["ci_lower"],
            "CI Upper": result["ci_upper"],
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df.style.format({
        "ATE": "{:.6f}",
        "SE": "{:.6f}",
        "CI Lower": "{:.6f}",
        "CI Upper": "{:.6f}",
    }))

    # Forest plot
    fig = go.Figure()
    for i, (method, result) in enumerate(methods.items()):
        fig.add_trace(go.Scatter(
            x=[result["ate"]],
            y=[method],
            error_x=dict(
                type="data",
                array=[result["ci_upper"] - result["ate"]],
                arrayminus=[result["ate"] - result["ci_lower"]]
            ),
            mode="markers",
            marker=dict(size=12),
            name=method
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(
        title="Forest Plot: ATE Estimates by Method",
        xaxis_title="Average Treatment Effect",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # CUPED variance reduction
    cuped = methods["CUPED"]
    st.subheader("CUPED Variance Reduction")
    st.metric("Variance Reduction", f"{cuped['variance_reduction']:.1%}")

with tab4:
    st.header("Heterogeneous Treatment Effects")

    st.subheader("Treatment Effect by Engagement Level")
    hte_df = subgroup_analysis(df, outcome)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hte_df["bucket"].astype(str),
        y=hte_df["ate"],
        error_y=dict(
            type="data",
            array=1.96 * hte_df["se"],
            arrayminus=1.96 * hte_df["se"]
        ),
        mode="markers+lines",
        marker=dict(size=12)
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(
        title=f"HTE by Game Rounds Bucket ({outcome_label})",
        xaxis_title="Game Rounds Bucket",
        yaxis_title="Treatment Effect"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(hte_df.style.format({
        "control_mean": "{:.4f}",
        "treatment_mean": "{:.4f}",
        "ate": "{:.4f}",
        "se": "{:.4f}",
        "ci_lower": "{:.4f}",
        "ci_upper": "{:.4f}",
    }))

# Footer
st.markdown("---")
st.markdown("""
**Methods Implemented:**
- Classical: Difference in means, Z-test, Power analysis, SRM detection
- Causal: Regression adjustment, IPW, CUPED, Doubly robust
- HTE: Subgroup analysis, Causal forests, Meta-learners (run via CLI)
- Sensitivity: Peeking simulation, Multiple testing correction, Robustness bounds
""")
