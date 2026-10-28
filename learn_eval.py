#!/usr/bin/env python3
import json, collections
import numpy as np
from bandit_transpile import ARMS, bucket

T = json.load(open("arm_table.json"))
A = len(ARMS)
cx = np.array([[s[0] for s in r["arms"]] for r in T], float)       # (circuits, arms)
dp = np.array([[s[1] for s in r["arms"]] for r in T], float)
dcx = np.array([r["default"][0] for r in T], float); ddp = np.array([r["default"][1] for r in T], float)
cost = cx + 0.5 * dp; dcost = dcx + 0.5 * ddp
reward = 1 - cost / np.maximum(dcost, 1)[:, None]                          # >0 : better than Qiskit default
fam = np.array([r["family"] for r in T]); buckets = [bucket(tuple(r["feat"])) for r in T]
N = len(T)


def train_bandit(idx, rng, episodes=4000, eps0=0.5):
    """Contextual epsilon-greedy: Q[bucket][arm] incremental mean; epsilon decays. Observes ONE arm per pull."""
    Q = collections.defaultdict(lambda: np.zeros(A)); n = collections.defaultdict(lambda: np.zeros(A))
    for ep in range(episodes):
        i = rng.choice(idx); b = buckets[i]; eps = eps0 * (1 - ep / episodes) + 0.02
        a = rng.integers(A) if rng.random() < eps else int(np.argmax(Q[b]))
        r = reward[i, a]; n[b][a] += 1; Q[b][a] += (r - Q[b][a]) / n[b][a]
    glob = np.zeros(A); cnt = np.zeros(A)       # fallback arm for unseen buckets
    for b in Q: glob += Q[b] * n[b]; cnt += n[b]
    return Q, n, int(np.argmax(glob / np.maximum(cnt, 1)))


def evaluate(split_idx_test, Q, n, fallback, rng):
    pol = []
    for i in split_idx_test:
        b = buckets[i]; pol.append(int(np.argmax(Q[b])) if b in Q and n[b].sum() > 0 else fallback)
    return np.array(pol)


def split(rng, frac=0.6):
    tr, te = [], []
    for f in set(fam):
        ii = np.where(fam == f)[0]; rng.shuffle(ii); k = max(1, int(round(frac * len(ii))))
        tr += list(ii[:k]); te += list(ii[k:])
    return np.array(tr), np.array(te)


def summarize(idx, arm_choice):
    c = cx[idx, arm_choice]; d = dp[idx, arm_choice]
    return {"gate_red_%": float(np.mean(1 - c / np.maximum(dcx[idx], 1)) * 100),
            "depth_red_%": float(np.mean(1 - d / np.maximum(ddp[idx], 1)) * 100),
            "cost_red_%": float(np.mean(1 - (c + .5 * d) / np.maximum(dcost[idx], 1)) * 100),
            "matched_or_better": float(np.mean((c + .5 * d) <= dcost[idx] + 1e-9))}


rows = collections.defaultdict(list)
for seed in range(30):
    rng = np.random.default_rng(seed); tr, te = split(rng)
    Q, n, fb = train_bandit(tr, rng)
    pol = evaluate(te, Q, n, fb, rng)
    fixed = int(np.argmax(reward[tr].mean(0)))                      # best single arm on train
    oracle = np.argmin(cost[te], axis=1)                            # best arm per circuit (needs all 12 transpiles)
    ridx = rng.integers(A, size=len(te))
    default_arm = ARMS.index((2, "sabre", "sabre"))
    for name, ch in (("bandit(1 transpile)", pol), ("best_fixed_arm(train)", np.full(len(te), fixed)),
                     ("random_arm", ridx), ("oracle(12 transpiles)", oracle),
                     ("default_arm_table(2,sabre,sabre)", np.full(len(te), default_arm))):
        rows[name].append(summarize(te, ch))
out = {}
print(f"{N} circuits, {len(set(fam))} families, {A} arms; 30 random 60/40 stratified splits; test = unseen circuits")
print(f"{'policy':36s} {'2Q-gate red%':>13s} {'depth red%':>11s} {'cost red%':>10s} {'match/better':>13s}")
for name, L in rows.items():
    m = {k: (float(np.mean([r[k] for r in L])), float(np.std([r[k] for r in L]))) for k in L[0]}
    out[name] = m
    print(f"{name:36s} {m['gate_red_%'][0]:8.1f}±{m['gate_red_%'][1]:3.1f} {m['depth_red_%'][0]:8.1f}±{m['depth_red_%'][1]:3.1f} "
          f"{m['cost_red_%'][0]:6.1f}±{m['cost_red_%'][1]:3.1f} {m['matched_or_better'][0]:10.2f}")
# per-family look at the full-data policy
rng = np.random.default_rng(0); Q, n, fb = train_bandit(np.arange(N), rng)
print("learned arms per bucket (trained on all):")
for b in sorted(Q):
    print("  bucket", b, "->", ARMS[int(np.argmax(Q[b]))], f"(n={int(n[b].sum())}, mean reward {Q[b].max():+.3f})")
# leave-one-family-out for the structured families
print("leave-one-family-out (bandit trained on other families):")
lofo = {}
for f in sorted(set(fam)):
    te = np.where(fam == f)[0]; tr = np.where(fam != f)[0]
    Q, n, fb = train_bandit(tr, np.random.default_rng(1)); pol = evaluate(te, Q, n, fb, None)
    s = summarize(te, pol); lofo[f] = s
    print(f"  {f:7s} n={len(te):3d} gate {s['gate_red_%']:+6.1f}%  depth {s['depth_red_%']:+6.1f}%  cost {s['cost_red_%']:+6.1f}%")
out["lofo"] = lofo
json.dump(out, open("results.json", "w"), indent=1)
