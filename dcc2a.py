# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 19:23:11 2026

@author: gjgan
"""

import pandas as pd
import numpy as np
from dcc_algorithm import DCC, Gamma
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from datetime import datetime

tr = pd.read_csv('peril.training.csv')
te = pd.read_csv('peril.validation.csv')

# strip whitespace on the actual columns used for matching (both frames)
tr['Description'] = tr['Description'].astype(str).str.strip()
te['Description'] = te['Description'].astype(str).str.strip()

dat2 = pd.concat([tr, te], axis=0)

# deterministic, de-duplicated, NaN-free list of descriptions
descriptions = sorted(dat2['Description'].dropna().unique())
print(len(descriptions))

# group once (fast) instead of re-scanning tr/te for every description
tr_groups = tr.groupby('Description')['Loss'].apply(list)
te_groups = te.groupby('Description')['Loss'].apply(list)

X = []
for desc in descriptions:
    tmp = tr_groups.get(desc, []) + te_groups.get(desc, [])
    X.append([np.log1p(tmp)])


model = DCC(dist_families=[Gamma()], k=5, n_init=10, random_state=0)
checkpoint0 = datetime.now()
model.fit(X)
checkpoint1 = datetime.now()

print(checkpoint1-checkpoint0)

labels = model.labels_
centroids = model.centroids_
n = len(X)
cluster_colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
# blue / red / green / purple / orange, by DCC-assigned cluster
true_markers = ["o", "^", "s", "D", "P"]
# circle / triangle / square / diamond / plus (filled), by TRUE cluster

print("=" * 70)
print(model.summary())

X2 = np.zeros((dat2.shape[0],1))
labels2 = []
for i in range(dat2.shape[0]):
    X2[i][0] = np.log(dat2['Loss'].iloc[i])
    labels2.append(labels[descriptions.index(dat2['Description'].iloc[i])].item())
    
print(silhouette_score(X2, labels2))

plt.figure(figsize=(6, 4))
plt.hist(X2, bins=100, density=True)
plt.xlabel("log(1+Loss)")
plt.ylabel("Density")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("dat2.pdf", bbox_inches='tight', pad_inches=0)

plt.figure(figsize=(6, 4))
for h in range(model.k):
    members = np.where(labels == h)[0]
    pooled = np.concatenate([X[i][0] for i in members])
    plt.hist(pooled, bins=30, density=True, alpha=0.4,
            color=cluster_colors[h], label=f"cluster {h} data (n={len(pooled)})")
    shape, scale = centroids[h][0]
    xs = np.linspace(0, pooled.max() * 1.05, 300)
    plt.plot(xs, stats.gamma.pdf(xs, a=shape, scale=scale),
            color=cluster_colors[h], linewidth=2.5,
            label=f"cluster {h} fit: Gamma(shape={shape:.2f}, scale={scale:.2f})")
plt.xlabel("Log-loss")
plt.ylabel("Density")
#plt.title("(B) Component 0 (Gamma) \u2014 pooled multiset values\nvs. fitted centroid densities")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"dat2k{model.k}.pdf", bbox_inches='tight', pad_inches=0)

plt.figure(figsize=(6, 4))
plt.plot(model.history_)
plt.xlabel("Iteration")
plt.ylabel("Objective function value")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"dat2k{model.k}hist.pdf", bbox_inches='tight', pad_inches=0)



pts = np.zeros((len(X), 1))
for i in range(len(X)):
    pts[i, 0] = X[i][0].mean()

km = KMeans(n_clusters=model.k, n_init=10, random_state=2).fit(pts)

print(silhouette_score(pts, km.labels_))



def getLabel(labelsN, X):
    ind = np.unique(labelsN)
    k = len(ind)

    name_sets = {
        2: ['low', 'high'],
        3: ['low', 'medium', 'high'],
        5: ['negligible', 'low', 'medium', 'high', 'catastrophic'],
    }
    if k not in name_sets:
        raise ValueError(
            f"getLabel only has severity names defined for k in {sorted(name_sets)}, got k={k}."
        )
    names = name_sets[k]

    # average value of component 0 for each cluster label in `ind`
    avgs = np.array([
        np.concatenate([X[i][0] for i in np.where(labelsN == h)[0]]).mean()
        for h in ind
    ])

    # order cluster labels by increasing average severity
    order = np.argsort(avgs)          # positions into `ind`/`avgs`, low -> high
    sorted_cluster_labels = ind[order]  # actual cluster ids, low -> high

    label_map = {cluster_id: names[rank] for rank, cluster_id in enumerate(sorted_cluster_labels)}

    labelsS = np.array([label_map[h] for h in labelsN])
    return pd.Categorical(labelsS, categories=names, ordered=True)
        
dccLabels = getLabel(model.labels_, X)
kmLabels = getLabel(km.labels_, X)
pd.crosstab(dccLabels, kmLabels, margins=True)
dfLabel = pd.DataFrame({'Description': descriptions, 'DCC': dccLabels, 'Kmeans': kmLabels})
dfLabel.to_csv(f"label{model.k}.csv", index=False)
