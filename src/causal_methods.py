"""Causal inference methods for A/B test analysis."""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from src.utils import load_data, create_binary_treatment


def regression_adjustment(
    df: pd.DataFrame, outcome: str = "retention_1", covariates: list = None
) -> dict:
    """OLS regression adjustment for ATE estimation."""
    df = create_binary_treatment(df)

    if covariates is None:
        covariates = ["sum_gamerounds"]

    # Cap extreme values
    df = df.copy()
    cap = df["sum_gamerounds"].quantile(0.99)
    df["sum_gamerounds_capped"] = df["sum_gamerounds"].clip(upper=cap)

    X = df[["treatment", "sum_gamerounds_capped"]]
    X = sm.add_constant(X)
    y = df[outcome]

    model = sm.OLS(y, X).fit()

    ate = model.params["treatment"]
    se = model.bse["treatment"]
    ci_lower, ci_upper = model.conf_int().loc["treatment"]

    return {
        "outcome": outcome,
        "ate": ate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": model.pvalues["treatment"],
        "r_squared": model.rsquared,
        "covariates_used": ["sum_gamerounds_capped"],
    }


def propensity_score_weighting(
    df: pd.DataFrame, outcome: str = "retention_1"
) -> dict:
    """Inverse propensity score weighting (IPW) for ATE estimation."""
    df = create_binary_treatment(df).copy()

    # Cap extreme values for propensity model
    cap = df["sum_gamerounds"].quantile(0.99)
    df["sum_gamerounds_capped"] = df["sum_gamerounds"].clip(upper=cap)

    # Fit propensity score model
    X = df[["sum_gamerounds_capped"]]
    T = df["treatment"]

    ps_model = LogisticRegression(random_state=42)
    ps_model.fit(X, T)
    ps = ps_model.predict_proba(X)[:, 1]

    # Clip propensity scores to avoid extreme weights
    ps = np.clip(ps, 0.01, 0.99)
    df["ps"] = ps

    # IPW weights
    df["ipw"] = np.where(df["treatment"] == 1, 1 / ps, 1 / (1 - ps))

    # Weighted means
    treated = df[df["treatment"] == 1]
    control = df[df["treatment"] == 0]

    weighted_mean_treated = np.average(treated[outcome], weights=treated["ipw"])
    weighted_mean_control = np.average(control[outcome], weights=control["ipw"])

    ate = weighted_mean_treated - weighted_mean_control

    # Bootstrap for standard errors
    n_bootstrap = 1000
    bootstrap_ates = []
    for _ in range(n_bootstrap):
        sample = df.sample(n=len(df), replace=True)
        t = sample[sample["treatment"] == 1]
        c = sample[sample["treatment"] == 0]
        wm_t = np.average(t[outcome], weights=t["ipw"])
        wm_c = np.average(c[outcome], weights=c["ipw"])
        bootstrap_ates.append(wm_t - wm_c)

    se = np.std(bootstrap_ates)
    ci_lower = np.percentile(bootstrap_ates, 2.5)
    ci_upper = np.percentile(bootstrap_ates, 97.5)

    return {
        "outcome": outcome,
        "ate": ate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "mean_ps_treated": df[df["treatment"] == 1]["ps"].mean(),
        "mean_ps_control": df[df["treatment"] == 0]["ps"].mean(),
    }


def cuped_estimator(
    df: pd.DataFrame, outcome: str = "retention_1", covariate: str = "sum_gamerounds"
) -> dict:
    """
    CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.
    Uses game rounds as a proxy for pre-experiment behavior.
    """
    df = create_binary_treatment(df).copy()

    # Cap extreme values
    cap = df[covariate].quantile(0.99)
    df["cov_capped"] = df[covariate].clip(upper=cap)

    # Calculate theta (optimal coefficient)
    cov_outcome_cov = np.cov(df[outcome], df["cov_capped"])[0, 1]
    var_cov = df["cov_capped"].var()
    theta = cov_outcome_cov / var_cov if var_cov > 0 else 0

    # CUPED-adjusted outcome
    mean_cov = df["cov_capped"].mean()
    df["y_cuped"] = df[outcome] - theta * (df["cov_capped"] - mean_cov)

    # Calculate ATE on adjusted outcome
    treated = df[df["treatment"] == 1]["y_cuped"]
    control = df[df["treatment"] == 0]["y_cuped"]

    ate = treated.mean() - control.mean()
    se = np.sqrt(treated.var() / len(treated) + control.var() / len(control))

    # Variance reduction
    var_original = df[outcome].var()
    var_cuped = df["y_cuped"].var()
    variance_reduction = 1 - var_cuped / var_original

    return {
        "outcome": outcome,
        "ate": ate,
        "se": se,
        "ci_lower": ate - 1.96 * se,
        "ci_upper": ate + 1.96 * se,
        "theta": theta,
        "variance_reduction": variance_reduction,
        "original_variance": var_original,
        "cuped_variance": var_cuped,
    }


def doubly_robust_estimator(
    df: pd.DataFrame, outcome: str = "retention_1"
) -> dict:
    """Doubly robust estimation combining outcome model and propensity scores."""
    df = create_binary_treatment(df).copy()

    # Cap extreme values
    cap = df["sum_gamerounds"].quantile(0.99)
    df["sum_gamerounds_capped"] = df["sum_gamerounds"].clip(upper=cap)

    X = df[["sum_gamerounds_capped"]]
    T = df["treatment"].values
    Y = df[outcome].values

    # Propensity score model
    ps_model = LogisticRegression(random_state=42)
    ps_model.fit(X, T)
    ps = np.clip(ps_model.predict_proba(X)[:, 1], 0.01, 0.99)

    # Outcome models (separate for treatment and control)
    X_with_const = sm.add_constant(X)

    treated_idx = T == 1
    control_idx = T == 0

    # Outcome model for treated
    model_1 = sm.OLS(Y[treated_idx], X_with_const[treated_idx]).fit()
    mu_1 = model_1.predict(X_with_const)

    # Outcome model for control
    model_0 = sm.OLS(Y[control_idx], X_with_const[control_idx]).fit()
    mu_0 = model_0.predict(X_with_const)

    # Doubly robust estimator
    dr_treated = T * (Y - mu_1) / ps + mu_1
    dr_control = (1 - T) * (Y - mu_0) / (1 - ps) + mu_0

    ate = dr_treated.mean() - dr_control.mean()

    # Bootstrap for standard errors
    n_bootstrap = 1000
    bootstrap_ates = []
    n = len(df)

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        dr_t_boot = dr_treated[idx]
        dr_c_boot = dr_control[idx]
        bootstrap_ates.append(dr_t_boot.mean() - dr_c_boot.mean())

    se = np.std(bootstrap_ates)
    ci_lower = np.percentile(bootstrap_ates, 2.5)
    ci_upper = np.percentile(bootstrap_ates, 97.5)

    return {
        "outcome": outcome,
        "ate": ate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ps_model_accuracy": ps_model.score(X, T),
    }


def run_causal_analysis() -> dict:
    """Run all causal inference methods."""
    print("Loading data...")
    df = load_data()

    results = {}

    for outcome in ["retention_1", "retention_7"]:
        print(f"\n{'='*50}")
        print(f"=== Causal Analysis for {outcome} ===")
        print("=" * 50)

        print("\n--- Regression Adjustment ---")
        reg_adj = regression_adjustment(df, outcome)
        for k, v in reg_adj.items():
            if isinstance(v, float):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

        print("\n--- Propensity Score Weighting (IPW) ---")
        ipw = propensity_score_weighting(df, outcome)
        for k, v in ipw.items():
            if isinstance(v, float):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

        print("\n--- CUPED Variance Reduction ---")
        cuped = cuped_estimator(df, outcome)
        for k, v in cuped.items():
            if isinstance(v, float):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

        print("\n--- Doubly Robust Estimation ---")
        dr = doubly_robust_estimator(df, outcome)
        for k, v in dr.items():
            if isinstance(v, float):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

        results[outcome] = {
            "regression_adjustment": reg_adj,
            "ipw": ipw,
            "cuped": cuped,
            "doubly_robust": dr,
        }

    return results


if __name__ == "__main__":
    run_causal_analysis()
