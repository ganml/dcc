"""
August 2, 2026

Distribution-Centroid Clustering (DCC) Algorithm
==================================================

Implementation of the DCC algorithm described in "Clustering Multiset-Valued
Data via Parametric Distribution Centroids" (G. Gan, 2026).

Problem setting
----------------
Each observation x_i (i = 1,...,n) consists of d components. The j-th
component x_ij of observation i is itself a MULTISET of N_ij numerical
values:

    x_ij = {v_ij^(1), v_ij^(2), ..., v_ij^(N_ij)}

Instead of representing a cluster centroid by a numerical vector, DCC
represents each cluster centroid by a collection of PARAMETRIC PROBABILITY
DISTRIBUTIONS, one per component:

    Theta_h = (theta_h1, theta_h2, ..., theta_hd)

The dissimilarity between an observation and a cluster is the negative
log-likelihood of the observation's multisets under the cluster's
component-wise distributions (Eq. (eq:distmea) in the paper):

    D(x_i, C_h) = - sum_{j=1}^{d} sum_{m=1}^{N_ij} ln f_j(x_ij^(m); theta_hj)

The algorithm alternates, exactly as in Algorithm 1 of the paper:

    1. Initialize cluster memberships.
    2. Repeat until the stopping criterion is satisfied:
         a. UPDATE STEP:  for every cluster h and component j, estimate
            theta_hj by maximum likelihood using all values (pooled across
            observations currently assigned to C_h) of component j.
         b. ASSIGNMENT STEP: for every observation, recompute D(x_i, C_h)
            for all h and reassign x_i to the cluster with smallest D.

When N_ij = 1 for every i, j and every component distribution is chosen to
be Normal(mu_hj, sigma^2) with a common, fixed variance, DCC reduces exactly
to classical k-means (Section "Relationship to k-means" of the paper).

"""

from __future__ import annotations

import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Optional


# ---------------------------------------------------------------------------
# 1. Parametric distribution families
# ---------------------------------------------------------------------------
#
# Each distribution family D_j implements:
#   - fit(data)      -> MLE parameter tuple theta_hj estimated from pooled
#                        values of component j across all observations in C_h
#   - logpdf(data,theta) -> array of ln f_j(v; theta) for each value v
#
# New families can be added by subclassing DistributionFamily.

class DistributionFamily:
    """Abstract base class for a parametric distribution family D_j."""

    name = "abstract"

    def fit(self, data: np.ndarray) -> Tuple[float, ...]:
        """Maximum-likelihood estimate of theta from pooled 1-D data."""
        raise NotImplementedError

    def logpdf(self, data: np.ndarray, theta: Tuple[float, ...]) -> np.ndarray:
        """ln f_j(v; theta) evaluated elementwise on data."""
        raise NotImplementedError

    def param_names(self) -> Tuple[str, ...]:
        raise NotImplementedError

    # small helper so the update step never crashes on tiny/degenerate
    # clusters (e.g. right after a random initialization)
    def _safe(self, data: np.ndarray, min_n: int) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        if data.size < min_n:
            raise ValueError(
                f"{self.name} requires at least {min_n} observation(s) to fit; "
                f"got {data.size}."
            )
        return data


class Gamma(DistributionFamily):
    """Gamma(shape=k, scale=theta), support x > 0."""

    name = "Gamma"

    def fit(self, data):
        data = self._safe(data, 1)
        data = data[data > 0]
        if data.size < 2 or np.allclose(data, data[0]):
            # degenerate cluster fallback: point-mass-like gamma
            mean = max(float(np.mean(data)) if data.size else 1.0, 1e-6)
            return (10.0, mean / 10.0)  # tight peak around the mean
        shape, _loc, scale = stats.gamma.fit(data, floc=0)
        shape = max(shape, 1e-6)
        scale = max(scale, 1e-6)
        return (shape, scale)

    def logpdf(self, data, theta):
        shape, scale = theta
        data = np.asarray(data, dtype=float)
        return stats.gamma.logpdf(data, a=shape, loc=0, scale=scale)

    def param_names(self):
        return ("shape", "scale")


class Normal(DistributionFamily):
    """Normal(mu, sigma^2)."""

    name = "Normal"

    def __init__(self, min_sigma: float = 1e-6):
        self.min_sigma = min_sigma

    def fit(self, data):
        data = self._safe(data, 1)
        mu = float(np.mean(data))
        sigma = float(np.std(data, ddof=0))
        sigma = max(sigma, self.min_sigma)
        return (mu, sigma)

    def logpdf(self, data, theta):
        mu, sigma = theta
        data = np.asarray(data, dtype=float)
        return stats.norm.logpdf(data, loc=mu, scale=sigma)

    def param_names(self):
        return ("mu", "sigma")


class Lognormal(DistributionFamily):
    """Lognormal: log(X) ~ Normal(mu, sigma^2), support x > 0."""

    name = "Lognormal"

    def __init__(self, min_sigma: float = 1e-6):
        self.min_sigma = min_sigma

    def fit(self, data):
        data = self._safe(data, 1)
        data = data[data > 0]
        logs = np.log(data) if data.size else np.array([0.0])
        mu = float(np.mean(logs))
        sigma = float(np.std(logs, ddof=0))
        sigma = max(sigma, self.min_sigma)
        return (mu, sigma)

    def logpdf(self, data, theta):
        mu, sigma = theta
        data = np.asarray(data, dtype=float)
        return stats.lognorm.logpdf(data, s=sigma, loc=0, scale=np.exp(mu))

    def param_names(self):
        return ("mu", "sigma")


class Weibull(DistributionFamily):
    """Weibull_min(shape=k, scale=lambda), support x > 0."""

    name = "Weibull"

    def fit(self, data):
        data = self._safe(data, 1)
        data = data[data > 0]
        if data.size < 2 or np.allclose(data, data[0]):
            mean = max(float(np.mean(data)) if data.size else 1.0, 1e-6)
            return (5.0, mean)
        shape, _loc, scale = stats.weibull_min.fit(data, floc=0)
        return (max(shape, 1e-6), max(scale, 1e-6))

    def logpdf(self, data, theta):
        shape, scale = theta
        data = np.asarray(data, dtype=float)
        return stats.weibull_min.logpdf(data, c=shape, loc=0, scale=scale)

    def param_names(self):
        return ("shape", "scale")


class Exponential(DistributionFamily):
    """Exponential(rate), support x > 0. MLE: rate = 1 / mean."""

    name = "Exponential"

    def fit(self, data):
        data = self._safe(data, 1)
        data = data[data > 0]
        mean = max(float(np.mean(data)) if data.size else 1.0, 1e-8)
        return (1.0 / mean,)

    def logpdf(self, data, theta):
        (rate,) = theta
        data = np.asarray(data, dtype=float)
        return stats.expon.logpdf(data, loc=0, scale=1.0 / rate)

    def param_names(self):
        return ("rate",)


class Poisson(DistributionFamily):
    """Poisson(lambda), support x in {0,1,2,...}. MLE: lambda = mean."""

    name = "Poisson"

    def fit(self, data):
        data = self._safe(data, 1)
        lam = max(float(np.mean(data)), 1e-8)
        return (lam,)

    def logpdf(self, data, theta):
        (lam,) = theta
        data = np.asarray(data, dtype=float)
        return stats.poisson.logpmf(data, mu=lam)

    def param_names(self):
        return ("lambda",)

class Bernoulli(DistributionFamily):
    """Bernoulli(p), support x in {0,1}. MLE: p = mean."""

    name = "Bernoulli"

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def fit(self, data):
        data = self._safe(data, 1)
        p = float(np.mean(data))
        # keep p strictly inside (0,1) so logpmf never hits -inf on the
        # opposite outcome (e.g. p=1 but a 0 shows up in another observation)
        p = min(max(p, self.eps), 1.0 - self.eps)
        return (p,)

    def logpdf(self, data, theta):
        (p,) = theta
        data = np.asarray(data, dtype=float)
        return stats.bernoulli.logpmf(data, p=p)

    def param_names(self):
        return ("p",)

# ---------------------------------------------------------------------------
# 2. The DCC algorithm
# ---------------------------------------------------------------------------

@dataclass
class DCCResult:
    labels: np.ndarray                       # shape (n,), cluster index per observation
    centroids: List[List[Tuple[float, ...]]] # centroids[h][j] = theta_hj
    objective: float                         # final value of J (Eq. eq:obj)
    n_iter: int
    converged: bool
    history: List[float] = field(default_factory=list)  # J at each iteration


class DCC:
    """
    Distribution-Centroid Clustering.

    Parameters
    ----------
    dist_families : sequence of DistributionFamily
        dist_families[j] is D_j, the parametric family used for component j
        (same family for every cluster, as in the paper).
    k : int
        Number of clusters.
    max_iter : int
        Maximum number of (update, assignment) iterations.
    tol : float
        Convergence tolerance on the relative change of the objective J.
    n_init : int
        Number of random initializations; the best (lowest J) run is kept,
        analogous to how k-means is typically run in practice.
    random_state : int or None
        Seed for reproducibility.
    verbose : bool
    """

    def __init__(
        self,
        dist_families: Sequence[DistributionFamily],
        k: int,
        max_iter: int = 100,
        tol: float = 1e-6,
        n_init: int = 10,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.dist_families = list(dist_families)
        self.d = len(self.dist_families)
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose

        self.labels_: Optional[np.ndarray] = None
        self.centroids_: Optional[List[List[Tuple[float, ...]]]] = None
        self.objective_: Optional[float] = None
        self.n_iter_: Optional[int] = None
        self.history_: List[float] = []

    # -- core dissimilarity measure, Eq. (eq:distmea) -----------------------
    def _dissimilarity(self, x_i: List[np.ndarray], theta_h: List[Tuple[float, ...]]) -> float:
        """D(x_i, C_h) = -sum_j sum_m ln f_j(x_ij^(m); theta_hj)."""
        total = 0.0
        for j in range(self.d):
            values = x_i[j]
            if len(values) == 0:
                continue
            ll = self.dist_families[j].logpdf(np.asarray(values, dtype=float), theta_h[j])
            total -= float(np.sum(ll))
        return total

    def _all_dissimilarities(self, X: List[List[np.ndarray]], centroids) -> np.ndarray:
        """Return an (n, k) matrix of D(x_i, C_h)."""
        n = len(X)
        D = np.empty((n, self.k))
        for i in range(n):
            for h in range(self.k):
                D[i, h] = self._dissimilarity(X[i], centroids[h])
        return D

    # -- update step: MLE per cluster/component, Eq. (eq:updateC) ----------
    def _update_centroids(self, X: List[List[np.ndarray]], labels: np.ndarray):
        centroids = []
        for h in range(self.k):
            members = np.where(labels == h)[0]
            theta_h = []
            for j in range(self.d):
                if len(members) == 0:
                    # empty cluster: keep a weak/uninformative placeholder;
                    # will be repopulated by the reseeding logic below
                    pooled = np.array([1.0])
                else:
                    pooled = np.concatenate([np.asarray(X[i][j], dtype=float) for i in members]) \
                        if any(len(X[i][j]) for i in members) else np.array([1.0])
                theta_h.append(self.dist_families[j].fit(pooled))
            centroids.append(theta_h)
        return centroids

    # -- assignment step, Eq. (eq:updateU) ----------------------------------
    def _assign(self, X, centroids):
        D = self._all_dissimilarities(X, centroids)
        labels = np.argmin(D, axis=1)
        obj = float(np.sum(D[np.arange(len(X)), labels]))
        return labels, obj, D

    def _reseed_empty_clusters(self, labels, D, rng):
        """
        If the assignment step produces an empty cluster, reseed it with the
        observation currently contributing the most to the objective
        (i.e. worst-fit point), a standard safeguard used in k-means-style
        algorithms so that every cluster keeps enough data to estimate
        theta_hj at the next update step.
        """
        counts = np.bincount(labels, minlength=self.k)
        empty = np.where(counts == 0)[0]
        if len(empty) == 0:
            return labels
        labels = labels.copy()
        cur_cost = D[np.arange(len(labels)), labels]
        for h in empty:
            worst = int(np.argmax(cur_cost))
            labels[worst] = h
            cur_cost[worst] = -np.inf  # don't reuse the same point twice
        return labels

    # -- one full run from a given initial membership -----------------------
    def _run_once(self, X: List[List[np.ndarray]], init_labels: np.ndarray, rng) -> DCCResult:
        n = len(X)
        labels = init_labels.copy()
        history = []
        prev_obj = None
        centroids = None
        converged = False
        it = 0

        for it in range(1, self.max_iter + 1):
            # ---- Update step ----
            centroids = self._update_centroids(X, labels)

            # ---- Assignment step ----
            new_labels, obj, D = self._assign(X, centroids)
            new_labels = self._reseed_empty_clusters(new_labels, D, rng)
            history.append(obj)

            if self.verbose:
                print(f"  iter {it}: J = {obj:.6f}")

            no_membership_change = np.array_equal(new_labels, labels)
            rel_change = (
                abs(prev_obj - obj) / (abs(prev_obj) + 1e-12)
                if prev_obj is not None
                else np.inf
            )
            labels = new_labels
            prev_obj = obj

            if no_membership_change or rel_change < self.tol:
                converged = True
                break

        # final centroid re-estimation on the converged partition
        centroids = self._update_centroids(X, labels)
        _, obj, _ = self._assign(X, centroids)

        return DCCResult(
            labels=labels,
            centroids=centroids,
            objective=obj,
            n_iter=it,
            converged=converged,
            history=history,
        )

    # -- random-partition initialization (guarantees nonempty clusters) -----
    def _init_labels(self, n: int, rng: np.random.Generator) -> np.ndarray:
        labels = rng.integers(0, self.k, size=n)
        # guarantee every cluster gets at least one observation
        for h in range(self.k):
            if not np.any(labels == h):
                idx = rng.integers(0, n)
                labels[idx] = h
        return labels

    # -- public API -----------------------------------------------------------
    def fit(self, X: List[List[Sequence[float]]]) -> "DCC":
        """
        Fit the DCC model.

        Parameters
        ----------
        X : list of n observations. X[i] is a list of d components.
            X[i][j] is a 1-D array-like: the multiset x_ij (its N_ij values).

        Returns
        -------
        self
        """
        if len(X) == 0:
            raise ValueError("X must contain at least one observation.")
        if len(X[0]) != self.d:
            raise ValueError(
                f"Each observation must have d={self.d} components "
                f"(matching the number of dist_families given)."
            )

        # normalize multisets to numpy arrays once
        X_arr = [[np.asarray(comp, dtype=float) for comp in x] for x in X]
        n = len(X_arr)

        rng = np.random.default_rng(self.random_state)

        best_result: Optional[DCCResult] = None
        for run in range(self.n_init):
            init_labels = self._init_labels(n, rng)
            if self.verbose:
                print(f"Run {run + 1}/{self.n_init}")
            result = self._run_once(X_arr, init_labels, rng)
            if best_result is None or result.objective < best_result.objective:
                best_result = result

        self.labels_ = best_result.labels
        self.centroids_ = best_result.centroids
        self.objective_ = best_result.objective
        self.n_iter_ = best_result.n_iter
        self.history_ = best_result.history
        self.converged_ = best_result.converged
        return self

    def predict(self, X: List[List[Sequence[float]]]) -> np.ndarray:
        """Assign new multiset-valued observations to the fitted centroids."""
        if self.centroids_ is None:
            raise RuntimeError("Call fit(...) before predict(...).")
        X_arr = [[np.asarray(comp, dtype=float) for comp in x] for x in X]
        D = self._all_dissimilarities(X_arr, self.centroids_)
        return np.argmin(D, axis=1)

    def summary(self) -> str:
        """Human-readable summary of the fitted distribution centroids."""
        if self.centroids_ is None:
            return "DCC model has not been fit yet."
        lines = [f"DCC fit: k={self.k}, J={self.objective_:.4f}, "
                 f"iterations={self.n_iter_}, converged={self.converged_}"]
        for h, theta_h in enumerate(self.centroids_):
            lines.append(f"Cluster {h} (n={int(np.sum(self.labels_ == h))}):")
            for j, theta_hj in enumerate(theta_h):
                fam = self.dist_families[j]
                pretty = ", ".join(
                    f"{name}={val:.4f}" for name, val in zip(fam.param_names(), theta_hj)
                )
                lines.append(f"    component {j} ~ {fam.name}({pretty})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Demo / sanity check against synthetic data (and the k-means special case)
# ---------------------------------------------------------------------------

def _demo():
    rng = np.random.default_rng(42)

    # ----- Demo A: single-component Gamma-distributed multisets -----------
    # 3 true clusters with distinct Gamma(shape, scale) parameters.
    true_params = [(2.0, 3.0), (8.0, 1.5), (4.0, 6.0)]
    n_per_cluster = 40
    min_card, max_card = 15, 30

    X = []
    true_labels = []
    for h, (shape, scale) in enumerate(true_params):
        for _ in range(n_per_cluster):
            card = rng.integers(min_card, max_card + 1)
            multiset = rng.gamma(shape, scale, size=card)
            X.append([multiset])  # single component (d = 1)
            true_labels.append(h)
    true_labels = np.array(true_labels)

    model = DCC(dist_families=[Gamma()], k=3, n_init=10, random_state=0)
    model.fit(X)

    print("=" * 70)
    print("Demo A: single-component Gamma-distributed multisets")
    print("=" * 70)
    print(model.summary())

    try:
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(true_labels, model.labels_)
        print(f"\nAdjusted Rand Index vs. true clusters: {ari:.4f}")
    except ImportError:
        pass

    # ----- Demo B: two components with different distribution families ----
    # component 0 ~ Gamma  (e.g. claim amounts)
    # component 1 ~ Normal (e.g. a second, real-valued measurement)
    true_params_2 = [
        ((2.0, 2.0), (0.0, 1.0)),
        ((9.0, 1.0), (5.0, 1.0)),
    ]
    X2, true_labels2 = [], []
    for h, (g_params, n_params) in enumerate(true_params_2):
        for _ in range(60):
            card = rng.integers(10, 25)
            comp0 = rng.gamma(g_params[0], g_params[1], size=card)
            comp1 = rng.normal(n_params[0], n_params[1], size=card)
            X2.append([comp0, comp1])
            true_labels2.append(h)
    true_labels2 = np.array(true_labels2)

    model2 = DCC(dist_families=[Gamma(), Normal()], k=2, n_init=10, random_state=1)
    model2.fit(X2)

    print("\n" + "=" * 70)
    print("Demo B: two-component observations, Gamma x Normal")
    print("=" * 70)
    print(model2.summary())

    try:
        from sklearn.metrics import adjusted_rand_score
        ari2 = adjusted_rand_score(true_labels2, model2.labels_)
        print(f"\nAdjusted Rand Index vs. true clusters: {ari2:.4f}")
    except ImportError:
        pass

    # ----- Demo C: k-means special case (N_ij = 1, Normal, common sigma) --
    print("\n" + "=" * 70)
    print("Demo C: DCC reduces to k-means when N_ij=1 and D_j = Normal(mu, sigma^2)")
    print("=" * 70)
    from sklearn.cluster import KMeans

    centers = np.array([[0.0, 0.0], [8.0, 8.0], [0.0, 8.0]])
    pts = np.vstack([c + rng.normal(0, 1.0, size=(50, 2)) for c in centers])
    X3 = [[[pts[i, 0]], [pts[i, 1]]] for i in range(pts.shape[0])]

    model3 = DCC(dist_families=[Normal(), Normal()], k=3, n_init=10, random_state=2)
    model3.fit(X3)

    km = KMeans(n_clusters=3, n_init=10, random_state=2).fit(pts)

    try:
        from sklearn.metrics import adjusted_rand_score
        ari3 = adjusted_rand_score(model3.labels_, km.labels_)
        print(f"Adjusted Rand Index between DCC labels and k-means labels: {ari3:.4f}")
    except ImportError:
        pass
    print(model3.summary())


if __name__ == "__main__":
    _demo()
