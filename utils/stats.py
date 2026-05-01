"""Statistical analysis engine: model fitting, ANOVA, optimization."""
import numpy as np
import pandas as pd
from itertools import combinations

import statsmodels.api as sm
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from scipy.optimize import differential_evolution


# ─── Model Matrix ────────────────────────────────────────────────────────────

def build_model_matrix(X: np.ndarray, degree: str = "quadratic",
                       include_interactions: bool = True) -> tuple[np.ndarray, list]:
    """
    Expand coded design matrix X into a model matrix.

    Returns (X_model, term_names) where X_model has intercept prepended.
    degree: 'linear' | '2fi' (linear + 2FI) | 'quadratic' (full RSM)
    """
    n, k = X.shape
    terms = []
    cols = []

    # intercept
    cols.append(np.ones(n))
    terms.append("Intercept")

    # main effects
    for i in range(k):
        cols.append(X[:, i])
        terms.append(f"x{i+1}")

    if degree in ("2fi", "quadratic") and include_interactions:
        for i, j in combinations(range(k), 2):
            cols.append(X[:, i] * X[:, j])
            terms.append(f"x{i+1}*x{j+1}")

    if degree == "quadratic":
        for i in range(k):
            cols.append(X[:, i] ** 2)
            terms.append(f"x{i+1}²")

    return np.column_stack(cols), terms


def rename_terms(terms: list, factor_names: list) -> list:
    """Replace x1, x2, ... with actual factor names."""
    result = []
    for t in terms:
        new_t = t
        for i, name in enumerate(factor_names):
            new_t = new_t.replace(f"x{i+1}²", f"{name}²")
            new_t = new_t.replace(f"x{i+1}", name)
        result.append(new_t)
    return result


# ─── MLR ─────────────────────────────────────────────────────────────────────

def fit_mlr(X_coded: np.ndarray, y: np.ndarray, degree: str = "quadratic",
            factor_names: list = None) -> dict:
    """Fit MLR model and return full diagnostics."""
    X_model, terms = build_model_matrix(X_coded, degree=degree)
    n, p = X_model.shape

    if n < p:
        degree = "2fi" if degree == "quadratic" else "linear"
        X_model, terms = build_model_matrix(X_coded, degree=degree)
        n, p = X_model.shape

    ols = sm.OLS(y, X_model)
    result = ols.fit()

    # Q² via LOO cross-validation
    loo = LeaveOneOut()
    y_pred_loo = np.empty(n)
    for train_idx, test_idx in loo.split(X_model):
        try:
            m = sm.OLS(y[train_idx], X_model[train_idx]).fit()
            y_pred_loo[test_idx] = m.predict(X_model[test_idx])
        except Exception:
            y_pred_loo[test_idx] = np.mean(y[train_idx])

    ss_tot = np.sum((y - np.mean(y)) ** 2)
    press = np.sum((y - y_pred_loo) ** 2)
    q2 = max(0.0, 1.0 - press / ss_tot) if ss_tot > 0 else 0.0

    if factor_names:
        display_terms = rename_terms(terms, factor_names)
    else:
        display_terms = terms

    coef_df = pd.DataFrame({
        "Term": display_terms,
        "Coefficient": result.params,
        "Std Error": result.bse,
        "t-value": result.tvalues,
        "p-value": result.pvalues,
        "Significant": result.pvalues < 0.05,
    })

    # ANOVA table
    ss_model = np.sum((result.fittedvalues - np.mean(y)) ** 2)
    ss_res = np.sum(result.resid ** 2)
    df_model = p - 1
    df_res = n - p
    ms_model = ss_model / df_model if df_model > 0 else np.nan
    ms_res = ss_res / df_res if df_res > 0 else np.nan
    f_stat = ms_model / ms_res if ms_res and ms_res > 0 else np.nan
    anova_df = pd.DataFrame([
        {"Source": "Model", "SS": ss_model, "df": df_model, "MS": ms_model,
         "F": f_stat, "p-value": result.f_pvalue},
        {"Source": "Residual", "SS": ss_res, "df": df_res, "MS": ms_res,
         "F": np.nan, "p-value": np.nan},
        {"Source": "Total", "SS": ss_tot, "df": n - 1, "MS": np.nan,
         "F": np.nan, "p-value": np.nan},
    ])

    return {
        "type": "MLR",
        "degree": degree,
        "terms": terms,
        "display_terms": display_terms,
        "coefficients": coef_df,
        "r2": result.rsquared,
        "r2_adj": result.rsquared_adj,
        "q2": q2,
        "rmse": np.sqrt(ms_res) if ms_res else np.nan,
        "anova": anova_df,
        "y_pred": result.fittedvalues,
        "residuals": result.resid,
        "y_obs": y,
        "_params": np.asarray(result.params),
        "_X_model": X_model,
        "_X_coded": X_coded,
        "_n_factors": X_coded.shape[1],
        "n_obs": n,
    }


def predict_mlr(model: dict, X_coded_new: np.ndarray) -> np.ndarray:
    X_new, _ = build_model_matrix(X_coded_new, degree=model["degree"])
    return X_new @ model["_params"]


# ─── PLS ─────────────────────────────────────────────────────────────────────

def fit_pls(X_coded: np.ndarray, y: np.ndarray, n_components: int = None,
            factor_names: list = None) -> dict:
    """Fit PLS1 model with auto-component selection via CV."""
    n, k = X_coded.shape

    if n_components is None:
        n_components = max(1, min(k, n // 3))

    # Select best n_components via LOO
    best_nc, best_q2 = 1, -np.inf
    for nc in range(1, n_components + 1):
        pls = PLSRegression(n_components=nc, scale=True)
        loo = LeaveOneOut()
        y_loo = np.empty(n)
        for tr, te in loo.split(X_coded):
            pls.fit(X_coded[tr], y[tr])
            y_loo[te] = pls.predict(X_coded[te]).ravel()
        q2 = r2_score(y, y_loo)
        if q2 > best_q2:
            best_q2, best_nc = q2, nc

    pls = PLSRegression(n_components=best_nc, scale=True)
    pls.fit(X_coded, y)
    y_pred = pls.predict(X_coded).ravel()
    residuals = y - y_pred
    r2 = r2_score(y, y_pred)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    rmse = np.sqrt(np.mean(residuals ** 2))

    # VIP scores
    T = pls.x_scores_
    W = pls.x_weights_
    P = pls.x_loadings_
    n_comp = pls.n_components
    vip = np.zeros(k)
    s = np.diag(T.T @ T @ np.diag(pls.y_loadings_.ravel() ** 2))
    total_s = np.sum(s)
    for i in range(k):
        weight = np.array([W[i, j] / np.linalg.norm(W[:, j]) for j in range(n_comp)])
        vip[i] = np.sqrt(k * np.sum(s * weight ** 2) / total_s)

    fnames = factor_names if factor_names else [f"x{i+1}" for i in range(k)]
    vip_df = pd.DataFrame({"Factor": fnames, "VIP": vip}).sort_values("VIP", ascending=False)

    ss_model = np.sum((y_pred - np.mean(y)) ** 2)
    ss_res = np.sum(residuals ** 2)
    df_model = best_nc
    df_res = n - best_nc - 1
    ms_model = ss_model / df_model if df_model > 0 else np.nan
    ms_res = ss_res / df_res if df_res > 0 else np.nan
    f_stat = ms_model / ms_res if ms_res and ms_res > 0 else np.nan

    anova_df = pd.DataFrame([
        {"Source": "Model", "SS": ss_model, "df": df_model, "MS": ms_model,
         "F": f_stat, "p-value": np.nan},
        {"Source": "Residual", "SS": ss_res, "df": df_res, "MS": ms_res,
         "F": np.nan, "p-value": np.nan},
        {"Source": "Total", "SS": ss_tot, "df": n - 1, "MS": np.nan,
         "F": np.nan, "p-value": np.nan},
    ])

    return {
        "type": "PLS",
        "degree": "linear",
        "n_components": best_nc,
        "vip": vip_df,
        "r2": r2,
        "r2_adj": 1 - (1 - r2) * (n - 1) / max(1, n - best_nc - 1),
        "q2": max(0.0, best_q2),
        "rmse": rmse,
        "anova": anova_df,
        "y_pred": y_pred,
        "residuals": residuals,
        "y_obs": y,
        "_model": pls,
        "_X_coded": X_coded,
        "_n_factors": k,
        "n_obs": n,
    }


def predict_pls(model: dict, X_coded_new: np.ndarray) -> np.ndarray:
    return model["_model"].predict(X_coded_new).ravel()


def predict_response(model: dict, X_coded_new: np.ndarray) -> np.ndarray:
    if model["type"] == "MLR":
        return predict_mlr(model, X_coded_new)
    return predict_pls(model, X_coded_new)


# ─── Desirability ────────────────────────────────────────────────────────────

def single_desirability(y: float, config: dict) -> float:
    """Compute individual desirability score (0..1)."""
    goal = config.get("goal", "maximize")
    lo = config.get("lower_limit", None)
    hi = config.get("upper_limit", None)
    target = config.get("target", None)
    weight = config.get("weight", 1.0)

    if goal == "maximize":
        if lo is None or hi is None:
            return 0.5
        if y <= lo:
            return 0.0
        if y >= hi:
            return 1.0
        return ((y - lo) / (hi - lo)) ** weight

    elif goal == "minimize":
        if lo is None or hi is None:
            return 0.5
        if y <= lo:
            return 1.0
        if y >= hi:
            return 0.0
        return ((hi - y) / (hi - lo)) ** weight

    elif goal == "target":
        if target is None:
            return 0.5
        lo_eff = lo if lo is not None else target * 0.8
        hi_eff = hi if hi is not None else target * 1.2
        if y <= lo_eff or y >= hi_eff:
            return 0.0
        if y <= target:
            return ((y - lo_eff) / (target - lo_eff)) ** weight
        return ((hi_eff - y) / (hi_eff - target)) ** weight

    return 0.5


def overall_desirability(y_values: dict, response_configs: dict,
                         importances: dict = None) -> float:
    """Geometric mean of individual desirabilities, weighted by importance."""
    d_values = []
    weights = []
    for name, y_val in y_values.items():
        if name in response_configs:
            d = single_desirability(y_val, response_configs[name])
            imp = (importances or {}).get(name, 3)
            d_values.append(d ** imp)
            weights.append(imp)

    if not d_values:
        return 0.0
    total_w = sum(weights)
    return np.prod(d_values) ** (1.0 / total_w) if total_w > 0 else 0.0


# ─── Optimization ────────────────────────────────────────────────────────────

def run_optimization(models: dict, factor_names: list, factor_ranges: list,
                     response_configs: dict, importances: dict = None,
                     n_restarts: int = 5) -> dict:
    """Multi-response desirability optimization using differential evolution."""
    k = len(factor_names)
    bounds = [(-1.0, 1.0)] * k  # coded space

    def _scalar(arr):
        return np.asarray(arr).ravel()[0]

    def neg_desirability(x_coded):
        x = np.array(x_coded).reshape(1, -1)
        y_vals = {}
        for rname, m in models.items():
            try:
                y_vals[rname] = float(_scalar(predict_response(m, x)))
            except Exception:
                y_vals[rname] = np.nan
        return -overall_desirability(y_vals, response_configs, importances)

    best_result = None
    best_d = -np.inf
    for seed in range(n_restarts):
        try:
            res = differential_evolution(neg_desirability, bounds, seed=seed,
                                         maxiter=500, tol=1e-8)
            if -res.fun > best_d:
                best_d = -res.fun
                best_result = res
        except Exception:
            continue

    if best_result is None:
        # Fallback: return center of design space
        x_center = np.zeros(k)
        fun_val = neg_desirability(x_center)
        best_result = type("R", (), {"x": x_center, "fun": fun_val, "success": False})()
        best_d = -fun_val

    x_opt_coded = best_result.x
    from .doe_engine import coded_to_natural
    x_opt_natural = coded_to_natural(x_opt_coded.reshape(1, -1), factor_ranges)[0]

    y_opt = {}
    for rname, m in models.items():
        pred = np.asarray(predict_response(m, x_opt_coded.reshape(1, -1))).ravel()
        y_opt[rname] = float(pred[0]) if len(pred) > 0 else float("nan")

    return {
        "optimal_coded": {n: float(v) for n, v in zip(factor_names, x_opt_coded)},
        "optimal_natural": {n: float(v) for n, v in zip(factor_names, x_opt_natural)},
        "predicted_responses": y_opt,
        "desirability": float(best_d),
        "success": best_result.success,
    }


def perturbation_analysis(models: dict, x_opt_coded: dict, factor_names: list,
                           delta: float = 0.1, n_points: int = 20) -> dict:
    """Compute how each response changes as factors deviate from optimum."""
    x0 = np.array([x_opt_coded[n] for n in factor_names])
    result = {}
    for fi, fname in enumerate(factor_names):
        steps = np.linspace(-delta, delta, n_points)
        rows = {}
        for rname, m in models.items():
            preds = []
            for s in steps:
                x = x0.copy()
                x[fi] = np.clip(x[fi] + s, -1, 1)
                preds.append(float(np.asarray(predict_response(m, x.reshape(1, -1))).ravel()[0]))
            rows[rname] = preds
        result[fname] = {"steps": steps.tolist(), "predictions": rows}
    return result
