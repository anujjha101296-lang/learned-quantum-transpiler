#!/usr/bin/env python3
"""RL-guided transpiler configuration selection (contextual epsilon-greedy bandit).

Target: 4x4 grid coupling map, basis {cx, rz, sx, x}. Arms = (opt_level, layout, routing) configs, routing in {sabre, basic}
(lookahead routing was dropped: >15 s per call on 8-qubit circuits).
Reward = relative improvement of cost = 2Q-gates + 0.5*depth vs Qiskit's default transpile.
Protocol: circuits split 60/40 train/test (stratified by family). The bandit learns ONLY on train
circuits; policy is then frozen and evaluated on unseen test circuits (single transpile call each).
Controls: default, random-arm, best-fixed-arm (learned on train), oracle (best of all arms on test)."""
import json, sys, time, itertools, warnings
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT, efficient_su2, quantum_volume, CDKMRippleCarryAdder
from qiskit.circuit.random import random_circuit
from qiskit.transpiler import CouplingMap

warnings.filterwarnings("ignore")
CM = CouplingMap.from_grid(4, 4)
BASIS = ["cx", "rz", "sx", "x"]
ARMS = [(lv, lay, rt) for lv in (1, 2, 3) for lay in ("sabre", "dense") for rt in ("sabre", "basic")]


def ghz(n):
    qc = QuantumCircuit(n); qc.h(0)
    for i in range(n - 1): qc.cx(i, i + 1)
    return qc

def qaoa_ring(n, p, rng):
    qc = QuantumCircuit(n); qc.h(range(n))
    for _ in range(p):
        g, b = rng.random() * 3, rng.random() * 3
        for i in range(n): qc.rzz(g, i, (i + 1) % n)
        for i in range(n): qc.rx(b, i)
    return qc

def qft_c(n): return QFT(n).decompose(reps=2)

def su2(n, reps, rng):
    qc = efficient_su2(n, entanglement="full", reps=reps)
    return qc.assign_parameters(rng.random(qc.num_parameters) * 6.28).decompose()

def qvol(n, seed): return quantum_volume(n, seed=seed).decompose(reps=3)

def adder(k): return CDKMRippleCarryAdder(k).decompose(reps=2)

def build_circuits():
    rng = np.random.default_rng(7); C = []
    for n in range(4, 13): C.append(("ghz", n, ghz(n)))
    for n in range(4, 13): C.append(("qft", n, qft_c(n)))
    for n in range(4, 11):
        for r in (1, 2): C.append(("su2", n, su2(n, r, rng)))
    for n in range(4, 13):
        for p in (1, 2): C.append(("qaoa", n, qaoa_ring(n, p, rng)))
    for n in range(4, 9):
        for s in range(2): C.append(("qv", n, qvol(n, 100 + s)))
    for k in (2, 3, 4, 5): C.append(("adder", 2 * k + 2, adder(k)))
    for i in range(50):
        n = int(rng.integers(5, 13)); d = int(rng.integers(6, 30))
        C.append(("random", n, random_circuit(n, d, max_operands=2, seed=500 + i)))
    return C


def stats(qc):
    ops = qc.count_ops()
    return int(ops.get("cx", 0)), qc.depth()


def run(qc, arm, seed=0):
    lv, lay, rt = arm
    out = transpile(qc, coupling_map=CM, basis_gates=BASIS, optimization_level=lv,
                    layout_method=lay, routing_method=rt, seed_transpiler=seed)
    return stats(out)


def cost(s): return s[0] + 0.5 * s[1]


def features(qc):
    two = sum(1 for i in qc.data if len(i.qubits) == 2); n = qc.num_qubits
    d = max(qc.depth(), 1)
    return (n, two, two / max(len(qc.data), 1), d)


def bucket(f):
    n, two, dens, d = f
    return (0 if n <= 6 else 1 if n <= 9 else 2, 0 if two < 25 else 1 if two < 100 else 2)


if __name__ == "__main__":
    t0 = time.time()
    C = build_circuits(); print(f"{len(C)} circuits", flush=True)
    # ---- measure every arm on every circuit once (cached table), plus the default baseline
    table = []
    for idx, (fam, n, qc) in enumerate(C):
        base = transpile(qc, coupling_map=CM, basis_gates=BASIS, seed_transpiler=0)
        b = stats(base)
        arms = [run(qc, a) for a in ARMS]
        table.append({"i": idx, "family": fam, "n": n, "feat": features(qc), "default": b, "arms": arms})
        if idx % 5 == 0: print(idx, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(table, open("arm_table.json", "w"))
    print("table built", f"{time.time()-t0:.0f}s")
