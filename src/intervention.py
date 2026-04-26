"""
Stage 5: Intervention (local prep)

Trains the probe on the balanced-only data, extracts the probe direction,
chooses threshold τ on a validation split (not test), and saves everything
needed for the Colab intervention notebook.

Outputs:
  - data/probe_direction.npy     (4096-dim normalized vector)
  - data/probe_bias.npy          (scalar)
  - data/intervention_config.json (τ, best layer, α sweep values, split IDs)
"""

import json
import random
import re
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

random.seed(CFG["seeds"]["random"])
np.random.seed(CFG["seeds"]["numpy"])

BEST_LAYER = 16  # from Stage 4 analysis


def heuristic_label(resp):
    number_pat = re.compile(r'\b\d+\.?\d*\s*(?:eV|g/cm|meV|J|kJ|GPa)')
    estimate_pat = re.compile(
        r'(?:approximately|around|roughly|about|estimated?|~)\s*\d+\.?\d*',
        re.IGNORECASE,
    )
    range_pat = re.compile(
        r'\d+\.?\d*\s*[-–to]+\s*\d+\.?\d*\s*(?:eV|g/cm|meV)',
        re.IGNORECASE,
    )
    if number_pat.search(resp) or estimate_pat.search(resp) or range_pat.search(resp):
        return 1
    return 0


def load_act(tid, layer):
    return np.load(ROOT / "data" / "trajectories" / "activations" / f"{tid}.npy")[layer]


def main():
    print("Stage 5: Intervention Prep")
    print("=" * 60)

    with open(ROOT / "data" / "trajectories" / "all_trajectories.json") as f:
        all_trajs = json.load(f)

    # Balanced empty-side with heuristic labels
    bal_empty = []
    for t in all_trajs:
        if t["side"] != "empty" or t["system_prompt_variant"] != "balanced":
            continue
        bal_empty.append({
            "trajectory_id": t["trajectory_id"],
            "formula": t["formula"],
            "property": t["property"],
            "template_index": t["template_index"],
            "label_binary": heuristic_label(t["assistant_response"]),
        })

    # Balanced data-present (for accuracy preservation check)
    bal_present = []
    for t in all_trajs:
        if t["side"] != "data_present" or t["system_prompt_variant"] != "balanced":
            continue
        bal_present.append({
            "trajectory_id": t["trajectory_id"],
            "formula": t["formula"],
        })

    print(f"Balanced empty: {len(bal_empty)}")
    print(f"Balanced present: {len(bal_present)}")

    # Three-way split by material: train (50%), val (20%), test (30%)
    formulas = sorted(set(d["formula"] for d in bal_empty))
    random.shuffle(formulas)
    n_train = int(0.50 * len(formulas))
    n_val = int(0.20 * len(formulas))
    train_f = set(formulas[:n_train])
    val_f = set(formulas[n_train:n_train + n_val])
    test_f = set(formulas[n_train + n_val:])

    assert len(train_f & val_f) == 0
    assert len(train_f & test_f) == 0
    assert len(val_f & test_f) == 0

    train = [d for d in bal_empty if d["formula"] in train_f]
    val = [d for d in bal_empty if d["formula"] in val_f]
    test = [d for d in bal_empty if d["formula"] in test_f]

    from collections import Counter
    print(f"\nTrain: {len(train)} ({Counter(d['label_binary'] for d in train)})")
    print(f"Val:   {len(val)} ({Counter(d['label_binary'] for d in val)})")
    print(f"Test:  {len(test)} ({Counter(d['label_binary'] for d in test)})")

    # Train probe on train split
    train_X = np.stack([load_act(d["trajectory_id"], BEST_LAYER) for d in train])
    train_y = np.array([d["label_binary"] for d in train])
    val_X = np.stack([load_act(d["trajectory_id"], BEST_LAYER) for d in val])
    val_y = np.array([d["label_binary"] for d in val])
    test_X = np.stack([load_act(d["trajectory_id"], BEST_LAYER) for d in test])
    test_y = np.array([d["label_binary"] for d in test])

    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    clf.fit(train_X, train_y)

    # Validation AUROC
    val_proba = clf.predict_proba(val_X)[:, 1]
    val_auroc = roc_auc_score(val_y, val_proba) if len(set(val_y)) >= 2 else float("nan")
    print(f"\nVal AUROC: {val_auroc:.4f}")

    # Test AUROC (held out, reported only)
    test_proba = clf.predict_proba(test_X)[:, 1]
    test_auroc = roc_auc_score(test_y, test_proba) if len(set(test_y)) >= 2 else float("nan")
    print(f"Test AUROC: {test_auroc:.4f}")

    # Extract probe direction (normalized weight vector)
    direction = clf.coef_[0].copy()
    direction_norm = direction / np.linalg.norm(direction)
    bias = clf.intercept_[0]

    print(f"\nProbe direction: shape={direction_norm.shape}, norm={np.linalg.norm(direction_norm):.4f}")
    print(f"Probe bias: {bias:.4f}")

    # Choose threshold τ on validation split
    # Sweep thresholds, pick the one that maximizes F1 on val
    best_tau = 0.5
    best_f1 = 0
    for tau in np.arange(0.1, 0.9, 0.01):
        preds = (val_proba >= tau).astype(int)
        tp = ((preds == 1) & (val_y == 1)).sum()
        fp = ((preds == 1) & (val_y == 0)).sum()
        fn = ((preds == 0) & (val_y == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau

    print(f"\nThreshold τ (chosen on val): {best_tau:.2f}")
    print(f"Val F1 at τ: {best_f1:.4f}")

    # Apply τ on test to see what intervention would catch
    test_fires = (test_proba >= best_tau).astype(int)
    test_fab = test_y == 1
    true_pos = (test_fires & test_fab).sum()
    false_pos = (test_fires & ~test_fab).sum()
    false_neg = (~test_fires & test_fab).sum()
    print(f"\nTest set intervention preview:")
    print(f"  Probe fires: {test_fires.sum()}/{len(test)}")
    print(f"  True positives (catches fabrication): {true_pos}")
    print(f"  False positives (flags admit): {false_pos}")
    print(f"  Missed fabrications: {false_neg}")

    # False positive rate on data-present
    present_X = np.stack([
        load_act(d["trajectory_id"], BEST_LAYER) for d in bal_present
        if (ROOT / "data" / "trajectories" / "activations" / f"{d['trajectory_id']}.npy").exists()
    ])
    present_proba = clf.predict_proba(present_X)[:, 1]
    present_fires = (present_proba >= best_tau).sum()
    print(f"\n  Data-present false-positive rate: {present_fires}/{len(present_X)} "
          f"({present_fires/len(present_X):.1%})")

    # Save artifacts
    np.save(ROOT / "data" / "probe_direction.npy", direction_norm)
    np.save(ROOT / "data" / "probe_bias.npy", np.array([bias]))
    np.save(ROOT / "data" / "probe_weights_raw.npy", direction)

    config = {
        "best_layer": BEST_LAYER,
        "threshold_tau": float(best_tau),
        "val_auroc": float(val_auroc),
        "test_auroc": float(test_auroc),
        "val_f1_at_tau": float(best_f1),
        "alpha_sweep": [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0],
        "train_ids": [d["trajectory_id"] for d in train],
        "val_ids": [d["trajectory_id"] for d in val],
        "test_ids": [d["trajectory_id"] for d in test],
        "present_ids": [d["trajectory_id"] for d in bal_present],
        "probe_C": 1.0,
        "probe_max_iter": 1000,
    }
    with open(ROOT / "data" / "intervention_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved:")
    print(f"  data/probe_direction.npy")
    print(f"  data/probe_bias.npy")
    print(f"  data/probe_weights_raw.npy")
    print(f"  data/intervention_config.json")


if __name__ == "__main__":
    main()
