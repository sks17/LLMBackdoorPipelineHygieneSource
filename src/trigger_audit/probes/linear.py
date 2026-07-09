"""A numpy-only L2-regularized logistic-regression probe.

Deliberately dependency-free: sklearn would be the obvious choice, but the base package
must stay importable (and this probe trainable) with numpy alone. The implementation is
simple and correct -- standardization, full-batch gradient descent with a fixed iteration
budget and tolerance, deterministic zero init -- and can later be swapped for an sklearn
solver behind the same ``fit``/``predict_proba``/``decision_scores`` interface without
touching any caller.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PathLike = str | Path

# Below this, a feature column is treated as constant and its scale left at 1.0, so
# zero-variance columns (e.g. a dead activation dimension) never produce NaNs.
_STD_FLOOR = 1e-12


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    clipped = np.clip(logits, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class LinearProbe:
    """L2-regularized logistic regression trained by full-batch gradient descent.

    Inputs are standardized with the training mean/std (stored on the probe, so scoring new
    data applies the same transform). Initialization is deterministic (zeros), making fits
    bit-reproducible across runs -- a requirement for the pre-registered experiment grid.
    """

    def __init__(
        self,
        *,
        l2: float = 1e-3,
        lr: float = 0.5,
        max_iter: int = 500,
        tol: float = 1e-8,
    ) -> None:
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        if lr <= 0 or max_iter <= 0:
            raise ValueError("lr and max_iter must be positive")
        self._l2 = float(l2)
        self._lr = float(lr)
        self._max_iter = int(max_iter)
        self._tol = float(tol)
        self._weights: np.ndarray | None = None
        self._bias: float = 0.0
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._converged: bool = False

    @property
    def is_fitted(self) -> bool:
        """True once ``fit`` (or ``load``) has populated the parameters."""
        return self._weights is not None

    @property
    def converged(self) -> bool:
        """True if the fit met ``tol`` before exhausting ``max_iter``.

        A pre-registered grid wants this on record: a probe that ran out its iteration budget
        without the objective stabilising is a different artifact from one that converged, and
        silently treating them alike hides an under-trained probe behind a plausible score.
        """
        return self._converged

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        assert self._mean is not None and self._std is not None
        return (features - self._mean) / self._std

    def fit(self, features: np.ndarray, labels: np.ndarray) -> LinearProbe:
        """Fit on ``(n, d)`` features and boolean/0-1 labels; returns self for chaining."""
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64).ravel()
        if x.ndim != 2:
            raise ValueError(f"expected a 2-D feature matrix, got shape {x.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"features ({x.shape[0]}) and labels ({y.shape[0]}) disagree")
        if x.shape[0] == 0:
            raise ValueError("cannot fit on an empty dataset")

        self._mean = x.mean(axis=0)
        std = x.std(axis=0)
        self._std = np.where(std < _STD_FLOOR, 1.0, std)
        z = self._standardize(x)

        n, d = z.shape
        weights = np.zeros(d, dtype=np.float64)
        bias = 0.0
        previous_loss = np.inf
        converged = False
        for _ in range(self._max_iter):
            logits = z @ weights + bias
            probs = _sigmoid(logits)

            # Full penalized objective evaluated at the CURRENT parameters: cross-entropy via
            # logaddexp (stable for large |logits|) plus the L2 term on the same ``weights``
            # the logits were computed from. Comparing this against the previous iterate's
            # objective makes ``tol`` a like-for-like stopping test, rather than mixing a
            # pre-step cross-entropy with a post-step penalty.
            loss = float(
                np.mean(np.logaddexp(0.0, logits) - y * logits)
                + 0.5 * self._l2 * float(weights @ weights)
            )
            if abs(previous_loss - loss) < self._tol:
                converged = True
                break
            previous_loss = loss

            error = probs - y
            grad_w = z.T @ error / n + self._l2 * weights
            grad_b = float(error.mean())
            weights -= self._lr * grad_w
            bias -= self._lr * grad_b

        self._weights = weights
        self._bias = bias
        self._converged = converged
        return self

    def decision_scores(self, features: np.ndarray) -> np.ndarray:
        """Return logits (monotone in probability; the scores thresholds are calibrated on)."""
        if self._weights is None:
            raise RuntimeError("LinearProbe is not fitted; call fit() or load() first")
        z = self._standardize(np.asarray(features, dtype=np.float64))
        return z @ self._weights + self._bias

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return P(label=1) for each row."""
        return _sigmoid(self.decision_scores(features))

    def save(self, path: PathLike) -> None:
        """Persist parameters and hyperparameters to an ``.npz`` file."""
        if self._weights is None or self._mean is None or self._std is None:
            raise RuntimeError("cannot save an unfitted LinearProbe")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            np.savez(
                handle,
                weights=self._weights,
                bias=np.float64(self._bias),
                mean=self._mean,
                std=self._std,
                hyperparams=np.array([self._l2, self._lr, self._max_iter, self._tol]),
                converged=np.array([self._converged], dtype=bool),
            )

    @classmethod
    def load(cls, path: PathLike) -> LinearProbe:
        """Reconstruct a fitted probe from ``save`` output (exact score round-trip)."""
        with np.load(Path(path)) as archive:
            l2, lr, max_iter, tol = (float(v) for v in archive["hyperparams"])
            probe = cls(l2=l2, lr=lr, max_iter=int(max_iter), tol=tol)
            probe._weights = np.asarray(archive["weights"], dtype=np.float64)
            probe._bias = float(archive["bias"])
            probe._mean = np.asarray(archive["mean"], dtype=np.float64)
            probe._std = np.asarray(archive["std"], dtype=np.float64)
            probe._converged = bool(archive["converged"][0]) if "converged" in archive else False
        return probe
