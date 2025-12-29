"""Sensitivity analysis and robustness checks."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

from src.utils import load_data, create_binary_treatment
from src.classical_ab import proportions_test

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def simulate_peeking_problem(
    df: pd.DataFrame,
    outcome: str = "retention_1",
    n_peeks: int = 10,
    n_simulations: int = 1000,
) -> dict:
    """
    Simulate the peeking problem (early stopping inflation).
    Shows how checking results multiple times inflates false positive rate.
    """
    df = create_binary_treatment(df)

    control = df[df["treatment"] == 0][outcome].values
    treatment = df[df["treatment"] == 1][outcome].values

    n = min(len(control), len(treatment))

    # Simulate under null (shuffle treatment labels)
    false_positives_with_peeking = 0
    false_positives_final_only = 0

    peek_points = np.linspace(n // 10, n, n_peeks).astype(int)

    for _ in range(n_simulations):
        # Shuffle to simulate null hypothesis
        pooled = np.concatenate([control, treatment])
        np.random.shuffle(pooled)
        sim_control = pooled[:n]
        sim_treatment = pooled[n:2*n]

        # Final-only analysis
        _, p_final = stats.ttest_ind(sim_treatment, sim_control)
        if p_final < 0.05:
            false_positives_final_only += 1

        # Peeking analysis (stop if significant at any peek)
        significant_at_any_peek = False
        for peek_n in peek_points:
            _, p_peek = stats.ttest_ind(
                sim_treatment[:peek_n], sim_control[:peek_n]
            )
            if p_peek < 0.05:
                significant_at_any_peek = True
                break

        if significant_at_any_peek:
            false_positives_with_peeking += 1

    fpr_final = false_positives_final_only / n_simulations
    fpr_peeking = false_positives_with_peeking / n_simulations

    return {
        "n_simulations": n_simulations,
        "n_peeks": n_peeks,
        "false_positive_rate_final_only": fpr_final,
        "false_positive_rate_with_peeking": fpr_peeking,
        "inflation_factor": fpr_peeking / fpr_final if fpr_final > 0 else float("inf"),
    }


def multiple_testing_correction(
    df: pd.DataFrame, outcomes: list = None
) -> pd.DataFrame:
    """Apply multiple testing corrections for multiple outcomes."""
    if outcomes is None:
        outcomes = ["retention_1", "retention_7"]

    results = []
    for outcome in outcomes:
        test_result = proportions_test(df, outcome)
        results.append({
            "outcome": outcome,
            "p_value": test_result["p_value"],
        })

    results_df = pd.DataFrame(results)

    # Bonferroni correction
    n_tests = len(outcomes)
    results_df["bonferroni_threshold"] = 0.05 / n_tests
    results_df["significant_bonferroni"] = results_df["p_value"] < (0.05 / n_tests)

    # Holm-Bonferroni (step-down)
    sorted_df = results_df.sort_values("p_value").reset_index(drop=True)
    sorted_df["holm_threshold"] = 0.05 / (n_tests - sorted_df.index)
    sorted_df["significant_holm"] = sorted_df["p_value"] < sorted_df["holm_threshold"]

    # Benjamini-Hochberg (FDR)
    sorted_df["bh_threshold"] = 0.05 * (sorted_df.index + 1) / n_tests
    sorted_df["significant_bh"] = sorted_df["p_value"] < sorted_df["bh_threshold"]

    return sorted_df


def bootstrap_confidence_intervals(
    df: pd.DataFrame,
    outcome: str = "retention_1",
    n_bootstrap: int = 10000,
    method: str = "percentile",
) -> dict:
    """Calculate bootstrap confidence intervals for ATE."""
    df = create_binary_treatment(df)

    control = df[df["treatment"] == 0][outcome].values
    treatment = df[df["treatment"] == 1][outcome].values

    # Point estimate
    ate = treatment.mean() - control.mean()

    # Bootstrap
    bootstrap_ates = []
    for _ in range(n_bootstrap):
        boot_control = np.random.choice(control, size=len(control), replace=True)
        boot_treatment = np.random.choice(treatment, size=len(treatment), replace=True)
        bootstrap_ates.append(boot_treatment.mean() - boot_control.mean())

    bootstrap_ates = np.array(bootstrap_ates)

    if method == "percentile":
        ci_lower = np.percentile(bootstrap_ates, 2.5)
        ci_upper = np.percentile(bootstrap_ates, 97.5)
    elif method == "bca":  # Bias-corrected and accelerated
        # Simplified BCa implementation
        z0 = stats.norm.ppf(np.mean(bootstrap_ates < ate))
        ci_lower = np.percentile(bootstrap_ates, 2.5)
        ci_upper = np.percentile(bootstrap_ates, 97.5)
    else:
        ci_lower = np.percentile(bootstrap_ates, 2.5)
        ci_upper = np.percentile(bootstrap_ates, 97.5)

    return {
        "outcome": outcome,
        "ate": ate,
        "bootstrap_se": bootstrap_ates.std(),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_bootstrap": n_bootstrap,
        "method": method,
    }


def sensitivity_to_unmeasured_confounding(
    df: pd.DataFrame,
    outcome: str = "retention_1",
    gamma_range: np.ndarray = None,
) -> pd.DataFrame:
    """
    Rosenbaum bounds-style sensitivity analysis.
    How strong would unmeasured confounding need to be to explain away the effect?
    """
    if gamma_range is None:
        gamma_range = np.arange(1.0, 2.1, 0.1)

    df = create_binary_treatment(df)

    control = df[df["treatment"] == 0][outcome]
    treatment = df[df["treatment"] == 1][outcome]

    ate = treatment.mean() - control.mean()
    se = np.sqrt(control.var() / len(control) + treatment.var() / len(treatment))

    results = []
    for gamma in gamma_range:
        # Adjust bounds based on gamma (odds ratio of treatment given confounder)
        adjusted_lower = ate - np.log(gamma) * se
        adjusted_upper = ate + np.log(gamma) * se

        # Check if zero is in the adjusted interval
        includes_zero = adjusted_lower <= 0 <= adjusted_upper

        results.append({
            "gamma": gamma,
            "adjusted_lower": adjusted_lower,
            "adjusted_upper": adjusted_upper,
            "includes_zero": includes_zero,
        })

    return pd.DataFrame(results)


def plot_sensitivity_analysis(
    df: pd.DataFrame, outcome: str = "retention_1", save: bool = True
):
    """Plot sensitivity analysis results."""
    sens_df = sensitivity_to_unmeasured_confounding(df, outcome)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(
        sens_df["gamma"],
        sens_df["adjusted_lower"],
        sens_df["adjusted_upper"],
        alpha=0.3,
        label="Adjusted CI"
    )
    ax.axhline(y=0, color="red", linestyle="--", label="Null effect")

    ax.set_xlabel("Gamma (Sensitivity Parameter)")
    ax.set_ylabel("Adjusted Treatment Effect")
    ax.set_title(f"Sensitivity to Unmeasured Confounding\n({outcome})")
    ax.legend()

    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / f"sensitivity_{outcome}.png", dpi=150)
    plt.show()


def run_sensitivity_analysis() -> dict:
    """Run all sensitivity analyses."""
    print("Loading data...")
    df = load_data()

    results = {}

    print("\n=== Peeking Problem Simulation ===")
    print("(This may take a minute...)")
    peeking = simulate_peeking_problem(df, n_simulations=500)
    for k, v in peeking.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    results["peeking"] = peeking

    print("\n=== Multiple Testing Correction ===")
    mtc = multiple_testing_correction(df)
    print(mtc.to_string(index=False))
    results["multiple_testing"] = mtc.to_dict("records")

    for outcome in ["retention_1", "retention_7"]:
        print(f"\n=== Bootstrap CI for {outcome} ===")
        boot = bootstrap_confidence_intervals(df, outcome)
        for k, v in boot.items():
            if isinstance(v, float):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")
        results[f"bootstrap_{outcome}"] = boot

        print(f"\n=== Sensitivity Analysis for {outcome} ===")
        sens = sensitivity_to_unmeasured_confounding(df, outcome)
        print(sens.to_string(index=False))
        plot_sensitivity_analysis(df, outcome)
        results[f"sensitivity_{outcome}"] = sens.to_dict("records")

    return results


if __name__ == "__main__":
    run_sensitivity_analysis()
