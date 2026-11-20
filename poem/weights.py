"""Steps 2 and 3 of the PoEM framework: the composition weights, the coverage and the composition strength γ.

Φ̃ is the (N, K) matrix of policy log-ratios φ_k(x, y) = log π_k(y|x) − log π_ref(y|x) (Eq. 5), one
row per calibration (prompt, response) and one column per expert; `pids` holds each row's prompt id and t the
target reward on the same rows. Every fit centers rewards and log-ratios within each prompt, which removes the
prompt-only term log Z_r(x).

    composition_weights  α̂ and the experts PoEM decodes with, by fit_ridge or fit_nnls (step 2)
    fit_ridge        α̂ of Eq. 10 for a combined reward: ridge with a penalty on negative weights,
                     λ by 5-fold cross-validation over prompts, Σ_k |α̂_k| = 1
    fit_nnls         α̂ for a held-out reward: non-negative least squares, the eight largest weights, Σ_k α̂_k = 1
    calibration_mask the rows of a fit: no basis reward beyond 20 standard deviations, no row of the held-out expert
    decode_set       the experts PoEM decodes with: |α̂_k| ≥ 0.01
    coverage         Cov_B(r_tar), Eq. 11
    gamma_geom       γ_geom, Eq. 20, on the Gram matrix of centered_gram
    gamma_star       the drift-matched γ*, Eq. 21; drift_target is its right-hand side
    poem_members     the weighted members of one PoEM decode, Eq. 6
    effective_rank   exp(−Σ_i p_i log p_i) of R̃ or Φ̃; gram_effective_rank of ΔΘ (App. B.4)

fit_ridge, fit_nnls and coverage take raw rows and center them inside, so a caller that drops rows re-centers on the
rows it keeps. The fits run in float32 and γ in float64.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)   # ridge strengths λ tried by cross-validation
LAMBDA_NEG = 1e7                                 # penalty on negative weights
DECODE_THRESHOLD = 0.01                          # smallest |α̂_k| PoEM decodes with
GAMMA_GRID = (0.5, 1.0, 1.5, 2.0, 3.0)           # the γ grid the recipe decodes (App. B.3); gamma_star
                                                 # interpolates on it


def center_per_prompt(a, pids, dtype=np.float32):
    """a (N,) or (N, K) minus the mean of its prompt's rows, as a copy in `dtype`."""
    out = np.array(a, dtype=dtype)
    for p in np.unique(pids):
        m = pids == p
        out[m] -= out[m].mean(0) if out.ndim == 2 else out[m].mean()
    return out


def prompt_level_split(pids, train_frac=0.8, seed=0):
    """(train, held) row masks: the sorted prompt ids permuted with default_rng(seed), the first
    int(n_prompts · train_frac) of them train."""
    unique = np.unique(pids)
    perm = np.random.default_rng(seed).permutation(unique)
    m = np.isin(pids, perm[:int(len(unique) * train_frac)])
    return m, ~m


def _l1_normalize(w, *, signed=False):
    """w / Σ_k w_k, or w / Σ_k |w_k| when signed; w unchanged when that sum is at most 1e-9 in magnitude."""
    s = float(w.sum() if not signed else np.abs(w).sum())
    return w / s if abs(s) > 1e-9 else w


def _r2(y, yhat):
    """Coefficient of determination; NaN when y is empty (a cross-validation fold with no prompts) or constant."""
    if len(y) == 0:
        return float("nan")
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


def _solve_ridge(Phi, t, lam, lam_neg):
    """Ridge with a penalty on negative weights (Table 4): from the ridge solution of ‖Φ̃w − t̃‖² + λ‖w‖², at most 500
    gradient steps on ‖Φ̃w − t̃‖² + λ‖w‖² + λ_neg‖max(0, −w)‖² of size η = 1/(λ_max(Φ̃ᵀΦ̃) + λ + λ_neg), stopping
    when a step moves w by less than 1e-6. A step multiplies a negative w_k by about 1 − 2ηλ_neg ≈ −1, so a negative
    weight changes sign rather than shrinking, while the data term moves w by about ηλ_max per step. On log-ratio
    scales (λ_max ≈ 1e5-1e7) negative weights are pushed toward 0 over the 500 steps; on unit-scale features they can
    end near −w_k. The fixed budget, not a minimum, defines the estimator. Returns float32."""
    K = Phi.shape[1]
    PtP, Ptt = Phi.T @ Phi, Phi.T @ t
    eta = 1.0 / max(float(np.linalg.eigvalsh(PtP).max()) + lam + lam_neg, 1e-8)
    w = np.linalg.solve(PtP + lam * np.eye(K, dtype=Phi.dtype), Ptt)
    for _ in range(500):
        grad = (2 * (PtP @ w - Ptt) + 2 * lam * w
                - 2 * lam_neg * np.maximum(0.0, -w))
        w_new = w - eta * grad
        if np.linalg.norm(w_new - w) < 1e-6:
            w = w_new
            break
        w = w_new
    return w.astype(np.float32)


def solve_nnls(Phi, t):
    """Non-negative least squares, w ≥ 0. Returns float32."""
    w, _ = nnls(Phi, t)
    return w.astype(np.float32)


def _cv_lambda(Phi, t, pids):
    """The λ of LAMBDAS with the best mean R² over 5 prompt folds (the sorted prompt ids permuted with
    default_rng(0)), each fold held out once from _solve_ridge. Returns (λ, mean R²); ties keep the first λ."""
    folds = np.array_split(np.random.default_rng(0).permutation(np.unique(pids)), 5)
    masks = [np.isin(pids, f) for f in folds]
    best, best_cv = None, -np.inf
    for lam in LAMBDAS:
        rs = [_r2(t[m], Phi[m] @ _solve_ridge(Phi[~m], t[~m], lam, LAMBDA_NEG)) for m in masks]
        score = float(np.nanmean(rs))
        if score > best_cv:
            best, best_cv = lam, score
    if best is None:
        raise ValueError("ridge: every fold's R² is undefined (the centered target is 0 on every held-out fold)")
    return best, best_cv


def fit_ridge(Phi, t, pids):
    """α̂ of Eq. 10 for a combined reward. Φ̃ and t̃ are centered per prompt (float32) and the prompts
    split 80/20 (seed 0); on the 80 %, λ is picked from LAMBDAS by cross-validation (_cv_lambda) and _solve_ridge fits
    w; α̂ = w / Σ_k |w_k| in float64. Returns {alpha_hat, alpha_hat_raw (w), lambda, r2_cv, r2_heldout (R² on the
    other 20 %)}."""
    Pc, tc = center_per_prompt(Phi, pids), center_per_prompt(t, pids)
    held = prompt_level_split(pids, 0.8, seed=0)[1]
    lam, r2_cv = _cv_lambda(Pc[~held], tc[~held], pids[~held])
    w = _solve_ridge(Pc[~held], tc[~held], lam, LAMBDA_NEG)
    return {"alpha_hat": _l1_normalize(np.asarray(w, np.float64), signed=True), "alpha_hat_raw": w,
            "lambda": lam, "r2_cv": r2_cv, "r2_heldout": _r2(tc[held], Pc[held] @ w)}


def fit_nnls(X, t, pids, *, topk=8, drop_zeros=False):
    """α̂ for a held-out reward (and every RM-basis target): non-negative least squares of the centered t̃ on the
    centered columns X̃ (float32, solved in float64), then the `topk` largest |w_k| rescaled to Σ_k w_k = 1. When fewer
    than `topk` weights are non-zero the support keeps zero weights, in numpy's sort order; with drop_zeros (the RM
    bases) weights with |w_k| ≤ 1e-6 leave the support and the rest are rescaled again to Σ_k w_k = 1 (float32).
    Returns (support, weights): column indices of X by decreasing |w| and float32 weights."""
    Xc, tc = center_per_prompt(X, pids), center_per_prompt(t, pids)
    w = solve_nnls(Xc.astype(np.float64), tc.astype(np.float64))
    support = np.argsort(-np.abs(w))[:topk].tolist()
    w = np.asarray(_l1_normalize(w[support]), dtype=np.float32)
    if drop_zeros:
        keep = np.abs(w) > 1e-6
        support, w = [s for s, k in zip(support, keep) if k], w[keep] / w[keep].sum()
    return support, w


def composition_weights(X, t, pids, estimator="ridge", *, topk=8, drop_zeros=False):
    """Step 2: α̂ of the target t on the columns of X, by fit_ridge ("ridge") or fit_nnls ("nnls", with `topk` and
    `drop_zeros`), and the experts PoEM decodes with. Returns (α̂ (K,) float64, decode indices into the columns,
    their float32 weights, the fit's report): ridge decodes decode_set(α̂) and reports alpha_hat_raw, lambda,
    lambda_neg, r2_cv and r2_heldout; NNLS decodes its support and reports topk."""
    if estimator == "ridge":
        f = fit_ridge(X, t, pids)
        keep, w = decode_set(f["alpha_hat"])
        report = {"alpha_hat_raw": [float(x) for x in f["alpha_hat_raw"]], "lambda": float(f["lambda"]),
                  "lambda_neg": LAMBDA_NEG, "r2_cv": float(f["r2_cv"]), "r2_heldout": float(f["r2_heldout"])}
        return f["alpha_hat"], keep, w, report
    keep, w = fit_nnls(X, t, pids, topk=topk, drop_zeros=drop_zeros)
    alpha = np.zeros(X.shape[1])
    alpha[keep] = w
    return alpha, keep, w, {"topk": topk}


def calibration_mask(z, sources, heldout=None, max_abs_z=20.0):
    """The calibration rows a fit uses: those whose z-scored basis rewards z (N, R) are all within max_abs_z of the
    mean, and, for held-out reward `heldout`, whose source (the expert that wrote the row) is not the held-out
    expert."""
    mask = np.abs(z).max(1) <= max_abs_z
    return mask & (np.asarray(sources) != heldout) if heldout else mask


def decode_set(alpha, threshold=DECODE_THRESHOLD):
    """The experts PoEM decodes with: |α_k| ≥ threshold on float32(α), or all of them when none passes.
    Returns (indices, float32 weights)."""
    w = np.asarray(alpha, dtype=np.float32)
    keep = [k for k, wk in enumerate(w) if abs(float(wk)) >= threshold]
    if not keep:
        keep = list(range(len(w)))
    return keep, w[keep]


def coverage(X, t, pids, *, n_splits=20):
    """Cov_B(r_tar) of Eq. 11: X and t centered per prompt (float32); for s = 0 … n_splits − 1 the
    prompts split 80/20 with seed s, w = solve_nnls on the 80 %, and 1 − ‖t̃ − X̃w‖² / ‖t̃‖² on the other 20 %.
    Returns the mean over the splits."""
    Xc, tc = center_per_prompt(X, pids), center_per_prompt(t, pids)
    out = []
    for s in range(n_splits):
        tr, te = prompt_level_split(pids, 0.8, seed=s)
        e = tc[te] - Xc[te] @ solve_nnls(Xc[tr], tc[tr])
        out.append(1 - e @ e / (tc[te] @ tc[te]))
    return float(np.mean(out))


def centered_gram(Phi, pids):
    """G = Φ̃ᵀΦ̃ / N, Φ̃ centered per prompt in float64."""
    Phi = center_per_prompt(Phi, pids, np.float64)
    return Phi.T @ Phi / len(Phi)


def gamma_geom(G, w, cols=None):
    """γ_geom of Eq. 20 for weights w on the experts `cols` of G (all of G when None), in decode units:
    computed on w / s with s = Σ_k |w_k|, then divided by s, so a decode multiplies w as given. Signed weights enter
    wᵀGw signed. NaN when s = 0 or wᵀGw ≤ 0."""
    if cols is not None:
        G = G[np.ix_(cols, cols)]
    w = np.asarray(w, dtype=float)
    s = float(np.abs(w).sum())
    if s <= 0:
        return float("nan")
    w = w / s
    denom = float(w @ G @ w)
    return float(np.sqrt(float(np.abs(w) @ np.diag(G)) / denom)) / s if denom > 0 else float("nan")


def drift_target(weights, expert_drift):
    """The right-hand side of Eq. 21: Σ_k |α_k| D(π_k) with α rescaled to Σ_k |α_k| = 1; expert_drift[k] is
    D(π_k) of the expert with weight weights[k]."""
    aw = np.abs(np.asarray(weights, float))
    return float(sum(wi / aw.sum() * d for wi, d in zip(aw, expert_drift)))


def gamma_star(drift, target, gammas=GAMMA_GRID):
    """γ* of Eq. 21: drift[i] = D(PoEM at gammas[i]); the curve gets D = 0 at γ = 0, is made non-decreasing
    with a running maximum and interpolated linearly at `target`. target ≤ 0 gives 0; a target at or above the
    whole curve gives the last γ."""
    x = np.maximum.accumulate([0.0] + list(drift))
    g = np.array([0.0] + list(gammas))
    if target <= 0:
        return 0.0
    if target >= x[-1]:
        return float(g[-1])
    return float(np.interp(target, x, g))


def poem_members(adapters, weights, gamma=1.0, *, tol=1e-4):
    """[(adapter_k, γw_k)] + [(None, 1 − γΣ_k w_k)], the logit weights of Eq. 6; None is π_ref. γw_k are
    float64 products and Σ is numpy's float64 sum. The π_ref member is dropped when |1 − γΣ_k w_k| ≤ tol, and its
    weight then goes to the last member, so the weights sum to 1."""
    wg = gamma * np.asarray(weights, dtype=np.float64)
    members = [(a, float(x)) for a, x in zip(adapters, wg)]
    w_ref = 1.0 - float(wg.sum())
    if abs(w_ref) > tol:
        members.append((None, w_ref))
    else:
        members[-1] = (members[-1][0], members[-1][1] + w_ref)
    return members


def _entropy_rank(p):
    """exp(−Σ_i p_i log p_i) of the variance shares p."""
    return float(np.exp(-(p * np.log(p)).sum()))


def effective_rank(X, groups=None):
    """Effective rank of a reward matrix R̃ or log-ratio matrix Φ̃ (App. B.4): X centered within each group
    (float64; e.g. the prompt ids) when groups is given, columns divided by their standard deviation (when above
    1e-12), and p_i = s_i² / Σ_j s_j² over the singular values s_i > 1e-10."""
    if groups is not None:
        X = center_per_prompt(X, groups, np.float64)
    sd = X.std(0)
    X = X / np.where(sd > 1e-12, sd, 1.0)
    s = np.linalg.svd(X, compute_uv=False)
    s = s[s > 1e-10]
    return _entropy_rank(s ** 2 / (s ** 2).sum())


def gram_effective_rank(G):
    """Effective rank of the experts' weight updates ΔΘ from their uncentered Gram matrix G (App. B.4):
    p_i = λ_i / Σ_j λ_j over the eigenvalues of G, clipped at 0, largest first, zero shares dropped."""
    ev = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None)
    p = ev / ev.sum()
    return _entropy_rank(p[p > 0])
