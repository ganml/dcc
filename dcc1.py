# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 18:35:40 2026

@author: gjgan
"""
import numpy as np
from dcc_algorithm import DCC, Gamma, Normal
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import adjusted_rand_score


def generate_demo_b(rng):
    """Recreate the exact Demo B synthetic dataset from dcc_algorithm.py."""
    true_params_2 = [
        ((2.0, 2.0), (0.0, 4.0)),   # cluster 0: Gamma(2,2), Normal(0,4)
        ((6.0, 1.0), (3.0, 4.0)),   # cluster 1: Gamma(6,1), Normal(3,4)
    ]
    X, true_labels = [], []
    for h, (g_params, n_params) in enumerate(true_params_2):
        for _ in range(60):
            card = rng.integers(5, 25)
            comp0 = rng.gamma(g_params[0], g_params[1], size=card)
            comp1 = rng.normal(n_params[0], n_params[1], size=card)
            X.append([comp0, comp1])
            true_labels.append(h)
    return X, np.array(true_labels)



rng = np.random.default_rng(2)
X, true_labels = generate_demo_b(rng)

model = DCC(dist_families=[Gamma(), Normal()], k=2, n_init=10, random_state=1)
model.fit(X)
labels = model.labels_
centroids = model.centroids_

n = len(X)
cluster_colors = ["#1f77b4", "#d62728"]  # blue / red, by DCC-assigned cluster
true_markers = ["o", "^"]                # circle / triangle, by TRUE cluster


# ---- Panel A: scatter of per-observation means -------------------------
plt.figure(figsize=(6, 4))
mean0 = np.array([np.mean(x[0]) for x in X])
mean1 = np.array([np.mean(x[1]) for x in X])
for h in range(model.k):
    for t in range(2):
        mask = (labels == h) & (true_labels == t)
        if not np.any(mask):
            continue
        plt.scatter(
            mean0[mask], mean1[mask],
            c=cluster_colors[h], marker=true_markers[t],
            s=60, alpha=0.8, edgecolor="k", linewidth=0.4,
            label=f"cluster {h}, true cluster {t}",
        )
plt.xlabel("mean of component 0 (Gamma-valued multiset)")
plt.ylabel("mean of component 1 (Normal-valued multiset)")
#plt.title("(A) Per-observation multiset means\ncolor = DCC assignment, marker = true cluster")
plt.legend(fontsize=10, loc="best")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("dat1.eps", bbox_inches='tight', pad_inches=0)
plt.close()

# ---- Panel B: pooled component-0 values + fitted Gamma centroids ------
plt.figure(figsize=(6, 4))
for h in range(model.k):
    members = np.where(labels == h)[0]
    pooled = np.concatenate([X[i][0] for i in members])
    plt.hist(pooled, bins=25, density=True, alpha=0.4,
            color=cluster_colors[h], label=f"cluster {h} data (n={len(pooled)})")
    shape, scale = centroids[h][0]
    xs = np.linspace(0, pooled.max() * 1.05, 300)
    plt.plot(xs, stats.gamma.pdf(xs, a=shape, scale=scale),
            color=cluster_colors[h], linewidth=2.5,
            label=f"cluster {h} fit: Gamma(shape={shape:.2f}, scale={scale:.2f})")
plt.xlabel("value")
plt.ylabel("density")
#plt.title("(B) Component 0 (Gamma) \u2014 pooled multiset values\nvs. fitted centroid densities")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("dat1c0.pdf", bbox_inches='tight', pad_inches=0)

# ---- Panel C: pooled component-1 values + fitted Normal centroids -----
plt.figure(figsize=(6, 4))
for h in range(model.k):
    members = np.where(labels == h)[0]
    pooled = np.concatenate([X[i][1] for i in members])
    plt.hist(pooled, bins=25, density=True, alpha=0.4,
            color=cluster_colors[h], label=f"cluster {h} data (n={len(pooled)})")
    mu, sigma = centroids[h][1]
    xs = np.linspace(pooled.min() - 1, pooled.max() + 1, 300)
    plt.plot(xs, stats.norm.pdf(xs, loc=mu, scale=sigma),
            color=cluster_colors[h], linewidth=2.5,
            label=f"cluster {h} fit: Normal(mu={mu:.2f}, sigma={sigma:.2f})")
plt.xlabel("value")
plt.ylabel("density")
#plt.title("(C) Component 1 (Normal) \u2014 pooled multiset values\nvs. fitted centroid densities")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("dat1c1.pdf", bbox_inches='tight', pad_inches=0)

# ---- Panel D: multiset cardinalities by cluster -----------------------
plt.figure(figsize=(6, 4))
card = np.array([len(x[0]) for x in X])  # N_i (same for both components here)
data_by_cluster = [card[labels == h] for h in range(model.k)]
bp = plt.boxplot(data_by_cluster, patch_artist=True,
                 tick_labels=[f"cluster {h}\n(n_obs={len(data_by_cluster[h])})" for h in range(model.k)])
for patch, color in zip(bp["boxes"], cluster_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
plt.ylabel("multiset cardinality $N_{ij}$")
#plt.set_title("(D) Observation multiset sizes by cluster\n(DCC handles unequal cardinalities natively)")
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("dat1b.pdf", bbox_inches='tight', pad_inches=0)


ari = adjusted_rand_score(true_labels, labels)
print(f"Adjusted Rand Index (DCC labels vs. true clusters): {ari:.4f}")
print(model.summary())

plt.figure(figsize=(6, 4))
plt.plot(model.history_)
plt.xlabel("Iteration")
plt.ylabel("Objective function value")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"dat1k{model.k}hist.pdf", bbox_inches='tight', pad_inches=0)


# summarize each component by its mean, then apply k-means
pts = np.zeros((len(X), 2))
for i in range(len(X)):
    pts[i, 0] = X[i][0].mean()
    pts[i, 1] = X[i][1].mean()
    
from sklearn.cluster import KMeans

km = KMeans(n_clusters=2, n_init=10, random_state=2).fit(pts)

ari3 = adjusted_rand_score(true_labels, km.labels_)

plt.figure(figsize=(6, 4))
mean0 = np.array([np.mean(x[0]) for x in X])
mean1 = np.array([np.mean(x[1]) for x in X])
for h in range(model.k):
    for t in range(2):
        mask = (km.labels_ == h) & (true_labels == t)
        if not np.any(mask):
            continue
        plt.scatter(
            mean0[mask], mean1[mask],
            c=cluster_colors[h], marker=true_markers[t],
            s=60, alpha=0.8, edgecolor="k", linewidth=0.4,
            label=f"cluster {h}, true cluster {t}",
        )
plt.xlabel("mean of component 0 (Gamma-valued multiset)")
plt.ylabel("mean of component 1 (Normal-valued multiset)")
#plt.title("(A) Per-observation multiset means\ncolor = DCC assignment, marker = true cluster")
plt.legend(fontsize=10, loc="best")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("dat1km.pdf", bbox_inches='tight', pad_inches=0)