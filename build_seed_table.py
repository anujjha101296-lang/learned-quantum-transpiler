import json, time, warnings; warnings.filterwarnings("ignore")
from bandit_transpile import *
SEEDS = range(8)
C = build_circuits(); t0 = time.time(); out = []
for idx, (fam, n, qc) in enumerate(C):
    base = stats(transpile(qc, coupling_map=CM, basis_gates=BASIS, seed_transpiler=0))
    tab = [[run(qc, a, seed=s) for s in SEEDS] for a in ARMS]      # [config][seed] -> (2q, depth)
    out.append({"i": idx, "family": fam, "n": n, "feat": features(qc), "default": base, "tab": tab})
    if idx % 10 == 0: print(idx, f"{time.time()-t0:.0f}s", flush=True)
json.dump(out, open("seed_table.json", "w")); print("done", f"{time.time()-t0:.0f}s")
