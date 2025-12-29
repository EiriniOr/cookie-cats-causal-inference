"""Heterogeneous Treatment Effects (HTE) analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils import load_data, create_binary_treatment

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def subgroup_analysis(
    df: pd.DataFrame, outcome: str = "retention_1"
) -> pd.DataFrame:
    """Analyze treatment effects across subgroups defined by game rounds."""
    df = create_binary_treatment(df).copy()

    # Create engagement buckets
    df["engagement_bucket"] = pd.cut(
        df["sum_gamerounds"],
        bins=[0, 10, 50, 100, 500, float("inf")],
        labels=["0-10", "11-50", "51-100", "101-500", "500+"],
    )

    results = []
    for bucket in df["engagement_bucket"].dropna().unique():
        subset = df[df["engagement_bucket"] == bucket]

        control = subset[subset["treatment"] == 0][outcome]
        treatment = subset[subset["treatment"] == 1][outcome]

        ate = treatment.mean() - control.mean()
        se = np.sqrt(control.var() / len(control) + treatment.var() / len(treatment))

        results.append({
            "bucket": bucket,
            "n": len(subset),
            "n_control": len(control),
            "n_treatment": len(treatment),
            "control_mean": control.mean(),
            "treatment_mean": treatment.mean(),
            "ate": ate,
            "se": se,
            "ci_lower": ate - 1.96 * se,
            "ci_upper": ate + 1.96 * se,
        })

    return pd.DataFrame(results)


def plot_hte_by_engagement(df: pd.DataFrame, outcome: str = "retention_1", save: bool = True):
    """Plot heterogeneous treatment effects by engagement level."""
    hte_df = subgroup_analysis(df, outcome)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(hte_df))
    ax.errorbar(
        x, hte_df["ate"],
        yerr=1.96 * hte_df["se"],
        fmt="o", capsize=5, capthick=2, markersize=8
    )

    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(hte_df["bucket"])
    ax.set_xlabel("Game Rounds Bucket")
    ax.set_ylabel(f"Treatment Effect on {outcome}")
    ax.set_title(f"Heterogeneous Treatment Effects by Engagement Level\n({outcome})")

    # Add sample sizes as annotations
    for i, row in hte_df.iterrows():
        ax.annotate(f"n={row['n']:,}", (i, row["ate"] + 1.96 * row["se"] + 0.005),
                   ha="center", fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / f"hte_engagement_{outcome}.png", dpi=150)
    plt.show()


def causal_forest_analysis(
    df: pd.DataFrame, outcome: str = "retention_1"
) -> dict:
    """
    Causal forest for heterogeneous treatment effect estimation.
    Uses EconML's CausalForestDML.
    """
    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    except ImportError:
        print("EconML not installed. Run: pip install econml")
        return {}

    df = create_binary_treatment(df).copy()

    # Prepare features (cap extreme values)
    cap = df["sum_gamerounds"].quantile(0.99)
    df["sum_gamerounds_capped"] = df["sum_gamerounds"].clip(upper=cap)

    X = df[["sum_gamerounds_capped"]].values
    T = df["treatment"].values
    Y = df[outcome].values

    # Fit causal forest
    cf = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=42),
        model_t=RandomForestClassifier(n_estimators=100, random_state=42),
        n_estimators=100,
        random_state=42,
    )

    cf.fit(Y, T, X=X)

    # Get treatment effects
    te = cf.effect(X)
    te_lower, te_upper = cf.effect_interval(X, alpha=0.05)

    # Average treatment effect
    ate = te.mean()
    ate_se = te.std() / np.sqrt(len(te))

    return {
        "outcome": outcome,
        "ate": ate,
        "ate_se": ate_se,
        "te_mean": te.mean(),
        "te_std": te.std(),
        "te_min": te.min(),
        "te_max": te.max(),
        "te_median": np.median(te),
        "individual_effects": te,
        "ci_lower": te_lower,
        "ci_upper": te_upper,
    }


def meta_learner_analysis(
    df: pd.DataFrame, outcome: str = "retention_1"
) -> dict:
    """
    T-Learner and X-Learner for HTE estimation.
    """
    try:
        from causalml.inference.meta import BaseTClassifier, BaseXClassifier
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        print("causalml not installed. Run: pip install causalml")
        return {}

    df = create_binary_treatment(df).copy()

    cap = df["sum_gamerounds"].quantile(0.99)
    df["sum_gamerounds_capped"] = df["sum_gamerounds"].clip(upper=cap)

    X = df[["sum_gamerounds_capped"]].values
    treatment = df["treatment"].values
    y = df[outcome].values

    results = {}

    # T-Learner
    t_learner = BaseTClassifier(
        learner=RandomForestClassifier(n_estimators=100, random_state=42)
    )
    te_t = t_learner.fit_predict(X, treatment, y)

    results["t_learner"] = {
        "ate": te_t.mean(),
        "te_std": te_t.std(),
        "te_min": te_t.min(),
        "te_max": te_t.max(),
    }

    # X-Learner
    x_learner = BaseXClassifier(
        learner=RandomForestClassifier(n_estimators=100, random_state=42)
    )
    te_x = x_learner.fit_predict(X, treatment, y)

    results["x_learner"] = {
        "ate": te_x.mean(),
        "te_std": te_x.std(),
        "te_min": te_x.min(),
        "te_max": te_x.max(),
    }

    return results


def run_hte_analysis() -> dict:
    """Run complete HTE analysis pipeline."""
    print("Loading data...")
    df = load_data()

    results = {}

    for outcome in ["retention_1", "retention_7"]:
        print(f"\n{'='*50}")
        print(f"=== HTE Analysis for {outcome} ===")
        print("=" * 50)

        print("\n--- Subgroup Analysis ---")
        subgroup_df = subgroup_analysis(df, outcome)
        print(subgroup_df.to_string(index=False))

        print("\n--- Plotting HTE by Engagement ---")
        plot_hte_by_engagement(df, outcome)

        print("\n--- Causal Forest ---")
        cf_results = causal_forest_analysis(df, outcome)
        if cf_results:
            for k, v in cf_results.items():
                if isinstance(v, (float, np.floating)):
                    print(f"{k}: {v:.6f}")
                elif not isinstance(v, np.ndarray):
                    print(f"{k}: {v}")

        print("\n--- Meta-Learners ---")
        meta_results = meta_learner_analysis(df, outcome)
        if meta_results:
            for learner, metrics in meta_results.items():
                print(f"\n{learner}:")
                for k, v in metrics.items():
                    print(f"  {k}: {v:.6f}")

        results[outcome] = {
            "subgroup": subgroup_df.to_dict("records"),
            "causal_forest": cf_results,
            "meta_learners": meta_results,
        }

    return results


if __name__ == "__main__":
    run_hte_analysis()
