"""Classical A/B testing methods."""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.power import zt_ind_solve_power

from src.utils import load_data, create_binary_treatment


def difference_in_means(
    df: pd.DataFrame, outcome: str = "retention_1"
) -> dict:
    """Calculate difference in means with confidence interval."""
    df = create_binary_treatment(df)

    control = df[df["treatment"] == 0][outcome]
    treatment = df[df["treatment"] == 1][outcome]

    mean_control = control.mean()
    mean_treatment = treatment.mean()
    ate = mean_treatment - mean_control

    # Pooled standard error
    se = np.sqrt(control.var() / len(control) + treatment.var() / len(treatment))

    # 95% CI
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se

    return {
        "outcome": outcome,
        "mean_control": mean_control,
        "mean_treatment": mean_treatment,
        "ate": ate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "relative_effect": ate / mean_control if mean_control > 0 else None,
    }


def proportions_test(
    df: pd.DataFrame, outcome: str = "retention_1"
) -> dict:
    """Two-proportion z-test for binary outcomes."""
    df = create_binary_treatment(df)

    control = df[df["treatment"] == 0]
    treatment = df[df["treatment"] == 1]

    count = np.array([treatment[outcome].sum(), control[outcome].sum()])
    nobs = np.array([len(treatment), len(control)])

    z_stat, p_value = proportions_ztest(count, nobs)

    # Confidence intervals for each proportion
    ci_treatment = proportion_confint(count[0], nobs[0], alpha=0.05, method="wilson")
    ci_control = proportion_confint(count[1], nobs[1], alpha=0.05, method="wilson")

    return {
        "outcome": outcome,
        "z_statistic": z_stat,
        "p_value": p_value,
        "significant_at_05": p_value < 0.05,
        "ci_treatment": ci_treatment,
        "ci_control": ci_control,
    }


def check_sample_ratio_mismatch(df: pd.DataFrame, expected_ratio: float = 0.5) -> dict:
    """Check for sample ratio mismatch (SRM)."""
    n_control = (df["version"] == "gate_30").sum()
    n_treatment = (df["version"] == "gate_40").sum()
    n_total = n_control + n_treatment

    observed_ratio = n_treatment / n_total
    expected_treatment = n_total * expected_ratio

    # Chi-squared test
    chi2_stat = ((n_treatment - expected_treatment) ** 2 / expected_treatment +
                 (n_control - (n_total - expected_treatment)) ** 2 / (n_total - expected_treatment))
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)

    return {
        "n_control": n_control,
        "n_treatment": n_treatment,
        "observed_ratio": observed_ratio,
        "expected_ratio": expected_ratio,
        "chi2_statistic": chi2_stat,
        "p_value": p_value,
        "srm_detected": p_value < 0.01,  # Use stricter threshold for SRM
    }


def power_analysis(
    df: pd.DataFrame, outcome: str = "retention_1", mde: float = 0.01
) -> dict:
    """Post-hoc power analysis."""
    df = create_binary_treatment(df)

    control = df[df["treatment"] == 0][outcome]
    p_control = control.mean()

    # Effect size (Cohen's h for proportions)
    p_treatment = p_control + mde
    effect_size = 2 * (np.arcsin(np.sqrt(p_treatment)) - np.arcsin(np.sqrt(p_control)))

    n_per_group = len(df) // 2

    # Calculate achieved power
    power = zt_ind_solve_power(
        effect_size=effect_size,
        nobs1=n_per_group,
        alpha=0.05,
        ratio=1.0,
        alternative="two-sided",
    )

    # Calculate required sample size for 80% power
    required_n = zt_ind_solve_power(
        effect_size=effect_size,
        power=0.8,
        alpha=0.05,
        ratio=1.0,
        alternative="two-sided",
    )

    return {
        "outcome": outcome,
        "baseline_rate": p_control,
        "minimum_detectable_effect": mde,
        "effect_size_cohens_h": effect_size,
        "current_n_per_group": n_per_group,
        "achieved_power": power,
        "required_n_for_80_power": int(np.ceil(required_n)),
        "adequately_powered": power >= 0.8,
    }


def run_classical_analysis() -> dict:
    """Run all classical A/B test analyses."""
    print("Loading data...")
    df = load_data()

    results = {}

    for outcome in ["retention_1", "retention_7"]:
        print(f"\n=== Analysis for {outcome} ===")

        print("\n--- Difference in Means ---")
        dim = difference_in_means(df, outcome)
        for k, v in dim.items():
            if isinstance(v, float):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

        print("\n--- Proportions Z-Test ---")
        prop_test = proportions_test(df, outcome)
        for k, v in prop_test.items():
            if isinstance(v, float):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

        results[outcome] = {"difference_in_means": dim, "proportions_test": prop_test}

    print("\n=== Sample Ratio Mismatch Check ===")
    srm = check_sample_ratio_mismatch(df)
    for k, v in srm.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    results["srm"] = srm

    print("\n=== Power Analysis (MDE = 1%) ===")
    power = power_analysis(df, "retention_1", mde=0.01)
    for k, v in power.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    results["power"] = power

    return results


if __name__ == "__main__":
    run_classical_analysis()
