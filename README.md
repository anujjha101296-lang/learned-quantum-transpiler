# Learned transpiler-configuration search (Qiskit)
`bandit_transpile.py` builds 114 circuits (GHZ, QFT, EfficientSU2, QAOA ring, quantum volume, CDKM adders, 50 random) and measures 12 configs (opt level 1-3 x layout sabre/dense x routing sabre/basic) on a 4x4 grid coupling map, basis {cx, rz, sx, x}. `build_seed_table.py` repeats each config for 8 seeds. `learn_eval.py` = single-shot config choice; `budget_eval.py` = budgeted search (K transpiles per circuit, K=1 is Qiskit default). Cost = 2Q gates + 0.5 depth. Prior trained on 60% of circuits, evaluated on unseen 40% (stratified by family), 10 splits.

Single-shot config choice vs Qiskit default (level 2, sabre), cost reduction over 30 stratified 60/40 splits (`learn_eval.py`):
| policy | cost red. | 2Q red. | depth red. | matched or better |
|---|---|---|---|---|
| bandit (1 transpile) | 0.0 +- 0.6% | -0.1% | -0.2% | 89% |
| best fixed config (learned on train) | +0.4% | 0.0% | +0.5% | 77% |
| random config | -41.4% | -49.1% | -33.3% | 26% |
| oracle (best of 12 configs per circuit) | +2.2% | +0.7% | +3.8% | 100% |
Choosing a config alone gives ~0%; a careless choice is far worse. Gains come from search over seeds:
| K transpiles | random search (cost/2Q/depth %) | learned prior (cost/2Q/depth %) |
|---|---|---|
| 4 | 1.4 / 0.4 / 2.4 | 3.3 / 2.1 / 4.9 |
| 8 | 2.3 / 0.9 / 4.0 | 4.5 / 3.1 / 6.3 |
| 16 | 3.3 / 1.7 / 5.1 | 5.3 / 3.4 / 7.8 |
(greedy use of the prior is on par or slightly better than eps-greedy: 5.6 / 3.7 / 8.1 at K=16)
Limits: 'lookahead' routing dropped (>15 s per call). Synthetic benchmark, one coupling map. Earlier 14.8% / 7.0% figures were not reproducible against Qiskit's real default.
