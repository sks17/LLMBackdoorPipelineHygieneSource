"""Decision-free statistics: uncertainty for the delivered-rate estimates.

Per ANALYSIS_PLAN.md §5.1, outcomes are deterministic per fully-specified trial, so the informative
quantity is a rate with *generalization* uncertainty over the base population -- estimated by a
cluster bootstrap over ``base_id`` (the primary interval), with a Wilson interval offered as a
labeled per-trial approximation. Between-condition effects (H1/H3) are risk differences with a
paired cluster-bootstrap CI. The counterfactual control is summarized with the exact McNemar test.

Everything here uses numpy (a core dependency) and stdlib ``math`` only -- no scipy, so this layer
adds no install and is not gated on the pending decisions (the TOST equivalence margin and the
multiplicity scheme, which the H2/H4 tables need, are the gated part and live elsewhere).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

# Two-sided z for a 95% interval; kept explicit so the confidence level is visible at the call site.
Z_95 = 1.959963984540054


def wilson_ci(k: int, n: int, *, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (well-behaved at 0 and 1).

    A per-trial approximation: it treats the n trials in a cell as independent Bernoulli draws,
    which understates uncertainty when trials share a base. Reported alongside (never instead of)
    the cluster bootstrap, which is the primary interval.
    """
    if n <= 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def _cluster_sums(
    values: np.ndarray, clusters: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the unique clusters and, per cluster, the summed value and the row count."""
    uniq, inverse = np.unique(clusters, return_inverse=True)
    sums: np.ndarray = np.zeros(len(uniq), dtype=float)
    counts: np.ndarray = np.zeros(len(uniq), dtype=float)
    np.add.at(sums, inverse, values.astype(float))
    np.add.at(counts, inverse, 1.0)
    return uniq, sums, counts


def bootstrap_rate_ci(
    values: Sequence[float] | np.ndarray,
    clusters: Sequence[object] | np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Cluster (base) bootstrap CI for a mean rate: resample whole clusters with replacement.

    Returns ``(point, lo, hi)``. When every cluster carries the same value the interval collapses to
    the point -- an honest statement that, given these bases, there is no base-level variation to
    generalize from (that boundary case is exactly why Wilson is reported too).
    """
    vals: np.ndarray = np.asarray(values, dtype=float)
    clus: np.ndarray = np.asarray(clusters)
    if vals.size == 0:
        return (math.nan, math.nan, math.nan)
    point = float(vals.mean())
    _, sums, counts = _cluster_sums(vals, clus)
    m = len(sums)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n_boot, m))
    num = sums[idx].sum(axis=1)
    den = counts[idx].sum(axis=1)
    boots = num / np.where(den == 0, 1.0, den)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (point, float(lo), float(hi))


def bootstrap_paired_diff_ci(
    df: pd.DataFrame,
    *,
    cond_col: str,
    cond_a: str,
    cond_b: str,
    value_col: str = "delivered",
    cluster_col: str = "base_id",
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Paired cluster-bootstrap CI for the risk difference rate(cond_a) - rate(cond_b).

    The two conditions are measured on the *same* bases, so the bases are resampled once per
    iteration and both conditions are recomputed on the resampled set -- a paired design that
    respects the shared-base correlation (H1/H3 effect sizes). Returns ``(point, lo, hi)``.
    """
    sub = df[df[cond_col].isin([cond_a, cond_b])]
    uniq = np.array(sorted(set(sub[cluster_col])))
    if uniq.size == 0:
        return (math.nan, math.nan, math.nan)

    def aligned(cond: str) -> tuple[np.ndarray, np.ndarray]:
        part = sub[sub[cond_col] == cond]
        summed = part.groupby(cluster_col)[value_col].sum()
        sized = part.groupby(cluster_col)[value_col].size()
        s = np.array([float(summed.get(u, 0.0)) for u in uniq])
        c = np.array([float(sized.get(u, 0.0)) for u in uniq])
        return s, c

    a_sum, a_cnt = aligned(cond_a)
    b_sum, b_cnt = aligned(cond_b)

    def rate(s: np.ndarray, c: np.ndarray) -> float:
        total = float(c.sum())
        return float(s.sum() / total) if total else math.nan

    point = rate(a_sum, a_cnt) - rate(b_sum, b_cnt)
    m = len(uniq)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n_boot, m))
    ra = a_sum[idx].sum(axis=1) / np.where(a_cnt[idx].sum(axis=1) == 0, 1.0, a_cnt[idx].sum(axis=1))
    rb = b_sum[idx].sum(axis=1) / np.where(b_cnt[idx].sum(axis=1) == 0, 1.0, b_cnt[idx].sum(axis=1))
    boots = ra - rb
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (point, float(lo), float(hi))


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial, p0=0.5) on the discordant pair counts.

    With the counterfactual's degenerate control arm (trigger-absent twins never deliver, so c==0)
    this reduces to a sign test on the present-delivered pairs: it confirms that inserting the
    trigger changes delivery. It is a control/sanity statistic, not evidence for H1-H4.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5**n)
    return min(1.0, 2.0 * tail)


def mcnemar_from_pairs(
    df: pd.DataFrame,
    *,
    value_col: str = "delivered",
    pair_col: str = "pair_id",
    present_col: str = "trigger_present",
) -> dict[str, object]:
    """Recover matched present/absent pairs by ``pair_id`` and compute the McNemar statistic.

    ``pair_id`` includes ``trigger_id`` (ANALYSIS_PLAN.md correction 2), so each key groups exactly
    one present row and its trigger-absent twin. ``b`` = present-delivered & absent-not; ``c`` =
    present-not & absent-delivered (expected 0 under a clean control). Pairs missing either side are
    skipped and counted.
    """
    b = c = pairs = skipped = 0
    for _, group in df.groupby(pair_col):
        present = group[group[present_col]]
        absent = group[~group[present_col]]
        if present.empty or absent.empty:
            skipped += 1
            continue
        p_delivered = bool(present[value_col].iloc[0])
        a_delivered = bool(absent[value_col].iloc[0])
        pairs += 1
        if p_delivered and not a_delivered:
            b += 1
        elif (not p_delivered) and a_delivered:
            c += 1
    return {
        "n_pairs": pairs,
        "skipped": skipped,
        "b": b,
        "c": c,
        "p_value": exact_mcnemar_p(b, c),
    }


# --- equivalence testing (TOST) + multiplicity ---------------------------------------------------
# H2 (model-invariance) and H4 (synthetic-vs-real parity) are *parity* claims: "no significant
# difference" is not evidence of sameness. TOST inverts the test -- it asks whether the difference
# is provably *within* a pre-registered margin. The margin is a decision (proposed +-5 pp); it is a
# parameter here, defaulted but never hard-coded into a verdict. Both multiplicity corrections are
# produced side by side (Holm and Benjamini-Hochberg) so the choice is made from the evidence.


def _cluster_aligned(
    part: pd.DataFrame, uniq: np.ndarray, cluster_col: str, value_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """Per cluster in ``uniq``: the summed value and the row count (0 where a cluster is absent)."""
    summed = part.groupby(cluster_col)[value_col].sum()
    sized = part.groupby(cluster_col)[value_col].size()
    s = np.array([float(summed.get(u, 0.0)) for u in uniq])
    c = np.array([float(sized.get(u, 0.0)) for u in uniq])
    return s, c


def _resampled_rate(sums: np.ndarray, counts: np.ndarray, idx: np.ndarray) -> np.ndarray:
    den = counts[idx].sum(axis=1)
    return sums[idx].sum(axis=1) / np.where(den == 0, 1.0, den)


def bootstrap_diff_samples(
    df: pd.DataFrame,
    *,
    cond_col: str,
    cond_a: str,
    cond_b: str,
    value_col: str = "delivered",
    cluster_col: str = "base_id",
    paired: bool = True,
    n_boot: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Bootstrap distribution of rate(a) - rate(b), clustering on ``cluster_col``.

    ``paired=True`` (H2: the same bases run through every model) resamples the shared clusters once
    and recomputes both conditions on them. ``paired=False`` (H4: synthetic and real are disjoint
    base sets) resamples each condition's clusters independently.
    """
    a = df[df[cond_col] == cond_a]
    b = df[df[cond_col] == cond_b]
    rng = np.random.default_rng(seed)
    if paired:
        uniq = np.array(sorted(set(a[cluster_col]) | set(b[cluster_col])))
        if uniq.size == 0:
            return np.empty(0)
        a_sum, a_cnt = _cluster_aligned(a, uniq, cluster_col, value_col)
        b_sum, b_cnt = _cluster_aligned(b, uniq, cluster_col, value_col)
        idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
        return _resampled_rate(a_sum, a_cnt, idx) - _resampled_rate(b_sum, b_cnt, idx)
    ua = np.array(sorted(set(a[cluster_col])))
    ub = np.array(sorted(set(b[cluster_col])))
    if ua.size == 0 or ub.size == 0:
        return np.empty(0)
    a_sum, a_cnt = _cluster_aligned(a, ua, cluster_col, value_col)
    b_sum, b_cnt = _cluster_aligned(b, ub, cluster_col, value_col)
    ia = rng.integers(0, len(ua), size=(n_boot, len(ua)))
    ib = rng.integers(0, len(ub), size=(n_boot, len(ub)))
    return _resampled_rate(a_sum, a_cnt, ia) - _resampled_rate(b_sum, b_cnt, ib)


def tost_equivalence(
    df: pd.DataFrame,
    *,
    cond_col: str,
    cond_a: str,
    cond_b: str,
    margin: float,
    paired: bool = True,
    value_col: str = "delivered",
    cluster_col: str = "base_id",
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Bootstrap TOST for equivalence of two conditions' delivery rates within +-``margin``.

    Equivalence is declared when the (1 - 2*alpha) cluster-bootstrap CI of the difference falls in
    (-margin, +margin). ``p_tost`` is the larger of the two one-sided bootstrap tail masses (what
    the multiplicity correction is applied to). Returns ``equivalent=None`` when a side is empty.
    """
    a = df[df[cond_col] == cond_a][value_col]
    b = df[df[cond_col] == cond_b][value_col]
    samples = bootstrap_diff_samples(
        df,
        cond_col=cond_col,
        cond_a=cond_a,
        cond_b=cond_b,
        value_col=value_col,
        cluster_col=cluster_col,
        paired=paired,
        n_boot=n_boot,
        seed=seed,
    )
    if samples.size == 0 or len(a) == 0 or len(b) == 0:
        return {
            "diff": math.nan,
            "ci_lo": math.nan,
            "ci_hi": math.nan,
            "p_tost": math.nan,
            "equivalent": None,
        }
    diff = float(a.mean() - b.mean())
    ci_lo, ci_hi = (float(x) for x in np.percentile(samples, [100 * alpha, 100 * (1 - alpha)]))
    p_lower = float(np.mean(samples <= -margin))
    p_upper = float(np.mean(samples >= margin))
    p_tost = max(p_lower, p_upper)
    equivalent = bool(ci_lo > -margin and ci_hi < margin)
    return {
        "diff": diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_tost": p_tost,
        "equivalent": equivalent,
    }


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (family-wise error). NaNs pass through."""
    idx = [i for i, p in enumerate(pvalues) if not math.isnan(p)]
    m = len(idx)
    adjusted = [math.nan] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda k: pvalues[k])):
        running = max(running, (m - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg step-up adjusted p-values (false-discovery-rate). NaNs pass through."""
    idx = [i for i, p in enumerate(pvalues) if not math.isnan(p)]
    m = len(idx)
    adjusted = [math.nan] * len(pvalues)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = sorted(idx, key=lambda k: pvalues[k])[rank]
        prev = min(prev, pvalues[i] * m / (rank + 1))
        adjusted[i] = min(1.0, prev)
    return adjusted


# --- Firth-penalized logistic (pooled H1/H3 sensitivity under complete separation) --------------
# The per-cell delivered rates are near-total 0/1 cells (ANALYSIS_PLAN.md §5.1), so the pooled
# sensitivity view for H1 (policy effect) and H3 (policy x position) is a *Firth-penalized* logistic
# regression -- never a vanilla GLM, whose MLE diverges to +-inf under the guaranteed complete
# separation (§5.2 rows H1/H3, §11 "Vanilla logistic on separated cells -- Firth/exact only").
#
# Firth (1993) adds the Jeffreys-prior penalty ``0.5*log det(X'WX)`` to the log-likelihood. Its
# gradient is the penalized score ``U*(b) = X'(y - p) + X' diag(h)(0.5 - p)``, where ``h`` are the
# hat-matrix diagonals ``H = W^{1/2} X (X'WX)^{-1} X' W^{1/2}`` and ``W = diag(p(1-p))``. The
# penalty guarantees a finite maximum-penalized-likelihood estimate even under separation, removing
# the first-order (small-sample) bias of the MLE as a bonus. This is a *secondary* sensitivity view
# -- the primary path stays the per-cell exact/CI machinery in ``tables.py`` -- and it is
# self-contained pure numpy (no scipy/statsmodels/firthlogist dependency, per §5.4's
# in-repo-implementation allowance).
#
# p-values are Wald (``z = b / se``, ``se`` from the diagonal of ``(X'WX)^{-1}``). Wald is
# documented as acceptable in the brief; the cheap standard-error-based test, adequate for a
# sensitivity read. (Penalized-likelihood-ratio tests are more accurate near the boundary but need a
# per-coefficient refit; deliberately omitted to keep this layer light.)


@dataclass
class FirthResult:
    """Result of a Firth-penalized logistic fit, keyed by design-matrix column name.

    ``params``/``std_err``/``p_values`` map each design column (including the intercept, when
    present) to its penalized coefficient, Wald standard error, and two-sided Wald p-value.
    """

    params: dict[str, float]
    std_err: dict[str, float]
    p_values: dict[str, float]
    n: int
    n_iter: int
    converged: bool


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function, elementwise."""
    out: np.ndarray = np.where(
        eta >= 0.0,
        1.0 / (1.0 + np.exp(-np.clip(eta, -700.0, 700.0))),
        np.exp(np.clip(eta, -700.0, 700.0)) / (1.0 + np.exp(np.clip(eta, -700.0, 700.0))),
    )
    return out


def _penalized_loglik(x: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    """Firth penalized log-likelihood ``sum(y log p + (1-y) log(1-p)) + 0.5 log det(X'WX)``.

    Used only as the monotone objective for step-halving, so the ``log det`` is computed with
    ``slogdet`` (sign-safe) and probabilities are clipped away from 0/1 to keep the log finite.
    """
    eps = 1e-12
    p = np.clip(_sigmoid(x @ beta), eps, 1.0 - eps)
    ll = float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    w = p * (1.0 - p)
    xtwx = x.T @ (x * w[:, None])
    _, logdet = np.linalg.slogdet(xtwx)
    return ll + 0.5 * float(logdet)


def firth_logit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: list[str],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> FirthResult:
    """Fit Firth-penalized logistic regression by penalized-likelihood Newton-Raphson.

    ``x`` is the ``n x p`` design matrix (include an intercept column yourself if wanted) and ``y``
    the 0/1 outcome. Each Newton step is ``b <- b + (X'WX)^{-1} U*(b)`` with the penalized score
    ``U*`` above; the step is halved until the penalized log-likelihood does not decrease, which
    makes the iteration monotone and convergent even when the data are perfectly separable (the
    whole point -- a vanilla GLM would send coefficients to +-inf). Convergence is declared when the
    max absolute coefficient step falls below ``tol``.

    Standard errors are the square-root diagonal of ``(X'WX)^{-1}`` at the solution and p-values are
    two-sided Wald. ``(X'WX)^{-1}`` is formed with the pseudo-inverse so a numerically singular
    information matrix degrades gracefully rather than raising.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.ndim != 2:
        raise ValueError("x must be a 2-D design matrix")
    n, p = xa.shape
    if len(feature_names) != p:
        raise ValueError(f"feature_names has {len(feature_names)} entries but x has {p} columns")
    if ya.shape != (n,):
        raise ValueError("y must be a 1-D array with one entry per design row")

    beta = np.zeros(p, dtype=float)
    loglik = _penalized_loglik(xa, ya, beta)
    converged = False
    n_iter = 0
    for it in range(1, max_iter + 1):
        n_iter = it
        pr = _sigmoid(xa @ beta)
        w = pr * (1.0 - pr)
        xtwx = xa.T @ (xa * w[:, None])
        cov = np.linalg.pinv(xtwx)
        # Hat diagonals H_ii = w_i * x_i' (X'WX)^{-1} x_i (never form the full n x n hat matrix).
        h = w * np.einsum("ij,ij->i", xa @ cov, xa)
        score = xa.T @ (ya - pr + h * (0.5 - pr))
        step = cov @ score
        # Step-halving on the penalized log-likelihood keeps Newton monotone under separation.
        new_beta = beta + step
        new_ll = _penalized_loglik(xa, ya, new_beta)
        halvings = 0
        while new_ll < loglik - 1e-12 and halvings < 40:
            step = 0.5 * step
            new_beta = beta + step
            new_ll = _penalized_loglik(xa, ya, new_beta)
            halvings += 1
        beta = new_beta
        loglik = new_ll
        if float(np.max(np.abs(step))) < tol:
            converged = True
            break

    pr = _sigmoid(xa @ beta)
    w = pr * (1.0 - pr)
    cov = np.linalg.pinv(xa.T @ (xa * w[:, None]))
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    params: dict[str, float] = {}
    std_err: dict[str, float] = {}
    p_values: dict[str, float] = {}
    for j, name in enumerate(feature_names):
        b = float(beta[j])
        s = float(se[j])
        params[name] = b
        std_err[name] = s
        # Two-sided Wald: erfc(|z|/sqrt2) == 2*(1 - Phi(|z|)).
        p_values[name] = math.erfc(abs(b / s) / math.sqrt(2.0)) if s > 0.0 else math.nan
    return FirthResult(
        params=params,
        std_err=std_err,
        p_values=p_values,
        n=n,
        n_iter=n_iter,
        converged=converged,
    )


def firth_logit_from_frame(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    *,
    add_intercept: bool = True,
    interactions: bool = False,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> FirthResult:
    """Build a treatment-coded design matrix from categorical predictors and Firth-fit it.

    Each predictor is one-hot / treatment ("dummy") coded with the **first level in sorted order
    dropped as the reference** (deterministic, documented), giving one coefficient per non-reference
    level named ``"{predictor}={level}"``. With ``interactions=True`` every cross-product of dummies
    from *different* predictors is added (named ``"a:b"``), so ``delivered ~ C(policy)*C(position)``
    for H3 is ``predictors=["pipeline_policy", "trigger_position"], interactions=True`` and the H1
    ``delivered ~ C(policy)`` is ``predictors=["pipeline_policy"]``. The outcome is coerced to float
    0/1; an all-1 or all-0 predictor cell (guaranteed here) is exactly the separation Firth handles.
    """
    ya = df[outcome].astype(float).to_numpy()
    n = len(df)
    named_cols: list[tuple[str, np.ndarray]] = []
    if add_intercept:
        named_cols.append(("Intercept", np.ones(n, dtype=float)))

    dummy_groups: list[list[tuple[str, np.ndarray]]] = []
    for pred in predictors:
        series = df[pred].astype(str)
        levels = sorted(series.unique())  # first sorted level is the dropped reference
        group: list[tuple[str, np.ndarray]] = []
        for level in levels[1:]:
            col = (series == level).to_numpy(dtype=float)
            group.append((f"{pred}={level}", col))
        named_cols.extend(group)
        dummy_groups.append(group)

    if interactions:
        for group_a, group_b in combinations(dummy_groups, 2):
            for name_a, col_a in group_a:
                for name_b, col_b in group_b:
                    named_cols.append((f"{name_a}:{name_b}", col_a * col_b))

    if not named_cols:
        raise ValueError("design matrix is empty: pass predictors or add_intercept=True")
    names = [name for name, _ in named_cols]
    design = np.column_stack([col for _, col in named_cols])
    return firth_logit(design, ya, feature_names=names, max_iter=max_iter, tol=tol)
