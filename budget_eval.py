#!/usr/bin/env python3
"""Budgeted transpiler search. Each policy may run K transpiles per circuit (K=1 is the Qiskit default itself);
returns the best (2Q + 0.5*depth) candidate found. Prior for the learned policy is trained on TRAIN circuits only."""
import json, collections
import numpy as np
from bandit_transpile import ARMS, bucket
T = json.load(open("seed_table.json")); A = len(ARMS); S = len(T[0]["tab"][0]); N = len(T)
tab = np.array([r["tab"] for r in T], float)                    # (N, A, S, 2)
cost = tab[..., 0] + 0.5 * tab[..., 1]                           # (N, A, S)
dflt = np.array([r["default"] for r in T], float); dcost = dflt[:, 0] + 0.5 * dflt[:, 1]
fam = np.array([r["family"] for r in T]); bk = [bucket(tuple(r["feat"])) for r in T]
KS = (1, 4, 8, 16)

def split(rng):
    tr, te = [], []
    for f in set(fam):
        ii = np.where(fam == f)[0]; rng.shuffle(ii); k = max(1, int(round(.6 * len(ii)))); tr += list(ii[:k]); te += list(ii[k:])
    return np.array(tr), np.array(te)

def prior(tr):
    P = collections.defaultdict(lambda: np.zeros(A)); c = collections.defaultdict(float); G = np.zeros(A)
    for i in tr:
        r = 1 - cost[i].mean(1) / max(dcost[i], 1)                 # config-level mean reward over seeds
        P[bk[i]] += r; c[bk[i]] += 1; G += r
    return {b: P[b] / c[b] for b in P}, G / len(tr)

def run_policy(kind, i, K, pri, glob, rng):
    best = dcost[i]; bestm = (dflt[i, 0], dflt[i, 1]); used = 1
    if K == 1: return best, bestm
    p = pri.get(bk[i], glob).copy(); n = np.full(A, 2.0)           # prior pseudo-count 2
    tried = [set() for _ in range(A)]
    while used < K:
        if kind == "random": a = rng.integers(A)
        elif kind == "greedy_prior":
            pr = pri.get(bk[i], glob).copy()
            for q in range(A):
                if len(tried[q]) == S: pr[q] = -9
            a = int(np.argmax(pr))
        else:                                                      # learned: eps-greedy + online updates from prior
            a = rng.integers(A) if rng.random() < 0.2 else int(np.argmax(p))
        free = [s for s in range(S) if s not in tried[a]]
        if not free: p[a] = -9; 
        if not free: 
            if all(len(t) == S for t in tried): break
            continue
        s = int(rng.choice(free)); tried[a].add(s); c = cost[i, a, s]; used += 1
        r = 1 - c / max(dcost[i], 1); p[a] += (r - p[a]) / (n[a] + 1); n[a] += 1
        if c < best: best = c; bestm = (tab[i, a, s, 0], tab[i, a, s, 1])
    return best, bestm

res = {k: {K: [] for K in KS} for k in ("random", "greedy_prior", "learned_bandit")}
for sp in range(10):
    rng = np.random.default_rng(sp); tr, te = split(rng); pri, glob = prior(tr)
    print("split", sp, flush=True)
    for kind in res:
        for K in KS:
            g = []; d = []; cc = []
            for i in te:
                for rep in range(3 if K > 1 else 1):
                    b, (g2, d2) = run_policy(kind, i, K, pri, glob, rng)
                    cc.append(1 - b / max(dcost[i], 1)); g.append(1 - g2 / max(dflt[i, 0], 1)); d.append(1 - d2 / max(dflt[i, 1], 1))
            res[kind][K].append((np.mean(cc) * 100, np.mean(g) * 100, np.mean(d) * 100))
out = {}
print(f"{N} circuits; 12 configs x {S} seeds; K = transpiles per circuit (K=1 is Qiskit default); 10 splits, test circuits unseen")
print(f"{'policy':16s} " + " ".join(f"K={K:<2d}(cost/2Q/depth %)  " for K in KS))
for kind in res:
    row = []
    for K in KS:
        a = np.array(res[kind][K]); m = a.mean(0); out[f"{kind}|{K}"] = m.round(2).tolist(); row.append(f"{m[0]:5.1f}/{m[1]:5.1f}/{m[2]:5.1f}      ")
    print(f"{kind:16s} " + " ".join(row))
# oracle over the whole 12x8 grid
ob = 1 - cost.reshape(N, -1).min(1) / np.maximum(dcost, 1); print("oracle best of all 96 candidates: cost red %.1f%%" % (ob.mean() * 100)); out["oracle96"] = float(ob.mean() * 100)
json.dump(out, open("budget_results.json", "w"), indent=1)
