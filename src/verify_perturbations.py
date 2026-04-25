"""
Stage 1 Verification

Check 1: Re-verify every perturbation returns empty from the MP API using
         the same formula-based search that trajectory collection will use.
         Drop any that return data; regenerate replacements via the tightened
         generator; re-verify until all 200 are confirmed empty.

Check 2: Element frequency distribution across real and perturbed sets.
         Assert no element appears in more than 25 % of the perturbed set.
         If violated, drop excess perturbations for the over-represented
         element, regenerate replacements that avoid it, and recheck.

Outputs:
  - data/perturbations_verified.json   (final 200 with raw API confirmation)
  - reports/stage_1.md                 (human-readable gate report)
"""

import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from mp_api.client import MPRester

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

random.seed(CFG["seeds"]["random"])
np.random.seed(CFG["seeds"]["numpy"])

MP_API_KEY = os.environ.get("MP_API_KEY")
N_TARGET = CFG["data"]["n_perturbations"]  # 200
ELEMENT_CAP = 0.25  # no element in > 25 % of perturbed formulas

# Import perturbation helpers from data_construction
import sys
sys.path.insert(0, str(ROOT / "src"))
from data_construction import (
    PERTURBATION_FNS,
    SUBSTITUTION_MAP,
    ADDABLE_ELEMENTS,
    _parse_formula,
    _formula_str,
    _elements_in,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def query_mp_formula(mpr: MPRester, formula: str) -> list[dict]:
    """Query MP API by formula — same method trajectory collection will use.

    Returns a list of dicts with material_id and formula for every match,
    or an empty list if the formula is not in the database.
    """
    docs = mpr.materials.summary.search(
        formula=formula,
        fields=["material_id", "formula_pretty"],
    )
    return [
        {"material_id": str(d.material_id), "formula": d.formula_pretty}
        for d in docs
    ]


def element_freq(formulas: list[str]) -> dict[str, int]:
    """For each element, count how many formulas contain it."""
    c: Counter[str] = Counter()
    for f in formulas:
        els = _elements_in(f)
        for el in els:
            c[el] += 1
    return dict(c)


def freq_table_lines(freq: dict[str, int], total: int, top_n: int = 0) -> list[str]:
    """Return lines for a frequency table sorted by count descending."""
    items = sorted(freq.items(), key=lambda x: -x[1])
    if top_n:
        items = items[:top_n]
    lines = []
    lines.append(f"| {'El':>3s} | {'Count':>5s} | {'Frac':>6s} |")
    lines.append(f"|{'-'*5}|{'-'*7}|{'-'*8}|")
    for el, cnt in items:
        lines.append(f"| {el:>3s} | {cnt:5d} | {cnt/total:6.1%} |")
    return lines


def generate_one_perturbation(
    mpr: MPRester,
    real_materials: list[dict],
    used_formulas: set[str],
    real_formulas: set[str],
    excluded_elements: set[str],
    max_tries: int = 200,
) -> dict | None:
    """Generate a single verified-empty perturbation avoiding excluded elements."""
    strategies = list(PERTURBATION_FNS.keys())
    indices = list(range(len(real_materials)))
    random.shuffle(indices)

    for attempt in range(max_tries):
        src_idx = indices[attempt % len(indices)]
        source = real_materials[src_idx]
        strategy = random.choice(strategies)
        fn = PERTURBATION_FNS[strategy]

        perturbed = fn(source["formula"])
        if perturbed is None:
            continue
        if perturbed in used_formulas or perturbed in real_formulas:
            continue

        # Check excluded elements
        if excluded_elements and _elements_in(perturbed) & excluded_elements:
            continue

        # Verify empty via API + assertion
        results = query_mp_formula(mpr, perturbed)
        if len(results) != 0:
            continue

        return {
            "perturbed_formula": perturbed,
            "source_material_id": source["material_id"],
            "source_formula": source["formula"],
            "perturbation_type": strategy,
            "mp_api_response": [],  # confirmed empty
        }

        if attempt % 10 == 0:
            time.sleep(0.2)

    return None


# ---------------------------------------------------------------------------
# Check 1: Verify all perturbations return empty
# ---------------------------------------------------------------------------

def verify_all(mpr, perturbations, real_materials):
    """Verify every perturbation; drop non-empty; regenerate replacements."""
    print("=" * 60)
    print("CHECK 1: Verify all perturbations return empty from MP API")
    print("=" * 60)

    real_formulas = {m["formula"] for m in real_materials}
    used_formulas = {p["perturbed_formula"] for p in perturbations}
    initial_count = len(perturbations)

    verified = []
    dropped = []

    for i, p in enumerate(perturbations):
        formula = p["perturbed_formula"]
        results = query_mp_formula(mpr, formula)

        if len(results) == 0:
            entry = dict(p)
            entry["mp_api_response"] = []
            verified.append(entry)
        else:
            dropped.append({
                "formula": formula,
                "source_formula": p["source_formula"],
                "strategy": p["perturbation_type"],
                "mp_results": results,
            })
            used_formulas.discard(formula)

        if (i + 1) % 20 == 0:
            print(f"  Verified {i+1}/{len(perturbations)}... "
                  f"({len(verified)} pass, {len(dropped)} dropped)")

        if (i + 1) % 10 == 0:
            time.sleep(0.3)

    print(f"\n  Initial: {initial_count}")
    print(f"  Verified empty: {len(verified)}")
    print(f"  Dropped (non-empty): {len(dropped)}")

    # Regenerate replacements for dropped perturbations
    n_needed = N_TARGET - len(verified)
    n_regenerated = 0

    if n_needed > 0:
        print(f"\n  Regenerating {n_needed} replacements...")
        for _ in range(n_needed):
            new_p = generate_one_perturbation(
                mpr, real_materials, used_formulas, real_formulas,
                excluded_elements=set(),
            )
            assert new_p is not None, "Could not generate replacement perturbation"
            verified.append(new_p)
            used_formulas.add(new_p["perturbed_formula"])
            n_regenerated += 1
            if n_regenerated % 10 == 0:
                print(f"    Regenerated {n_regenerated}/{n_needed}...")

    # Final assertion
    assert len(verified) == N_TARGET, (
        f"Expected {N_TARGET} verified perturbations, got {len(verified)}"
    )

    print(f"\n  Final count: {len(verified)} verified-empty perturbations")

    return verified, dropped, n_regenerated


# ---------------------------------------------------------------------------
# Check 2: Element frequency cap
# ---------------------------------------------------------------------------

def enforce_element_cap(mpr, perturbations, real_materials):
    """Ensure no element exceeds ELEMENT_CAP fraction of the perturbed set.

    Iteratively: find the most over-represented element, drop excess
    perturbations containing it, regenerate replacements excluding it,
    and recheck.  Converges because each round strictly reduces the
    count of the most over-represented element.
    """
    print("\n" + "=" * 60)
    print("CHECK 2: Element frequency cap (max 25 % per element)")
    print("=" * 60)

    max_count = int(ELEMENT_CAP * N_TARGET)  # 50
    real_formulas = {m["formula"] for m in real_materials}
    used_formulas = {p["perturbed_formula"] for p in perturbations}
    total_dropped_for_cap: list[dict] = []
    total_regenerated_for_cap = 0
    rebalance_rounds = 0

    while True:
        pert_formulas = [p["perturbed_formula"] for p in perturbations]
        freq = element_freq(pert_formulas)
        violations = {el: cnt for el, cnt in freq.items() if cnt > max_count}

        if not violations:
            print(f"  All elements within {ELEMENT_CAP:.0%} cap.")
            break

        rebalance_rounds += 1
        worst_el = max(violations, key=violations.get)
        excess = violations[worst_el] - max_count
        print(f"\n  Round {rebalance_rounds}: {worst_el} appears in "
              f"{violations[worst_el]}/{N_TARGET} ({violations[worst_el]/N_TARGET:.1%}) "
              f"formulas — need to drop {excess}")

        # Identify which perturbations contain the worst element
        indices_with_el = [
            i for i, p in enumerate(perturbations)
            if worst_el in _elements_in(p["perturbed_formula"])
        ]
        random.shuffle(indices_with_el)
        to_drop = indices_with_el[:excess]

        dropped_this_round = []
        for idx in sorted(to_drop, reverse=True):
            p = perturbations.pop(idx)
            used_formulas.discard(p["perturbed_formula"])
            dropped_this_round.append(p)
            total_dropped_for_cap.append({
                "formula": p["perturbed_formula"],
                "reason": f"element {worst_el} over-represented",
            })

        print(f"    Dropped {len(dropped_this_round)} perturbations containing {worst_el}")

        # Regenerate replacements excluding all currently-violated elements
        # (re-read violations since we just changed the list)
        current_violations = set(violations.keys())
        n_needed = N_TARGET - len(perturbations)
        print(f"    Regenerating {n_needed} replacements (excluding {current_violations})...")

        for k in range(n_needed):
            new_p = generate_one_perturbation(
                mpr, real_materials, used_formulas, real_formulas,
                excluded_elements=current_violations,
            )
            if new_p is None:
                # Relax: only exclude the worst element
                new_p = generate_one_perturbation(
                    mpr, real_materials, used_formulas, real_formulas,
                    excluded_elements={worst_el},
                )
            assert new_p is not None, (
                f"Could not generate replacement avoiding {worst_el}"
            )
            perturbations.append(new_p)
            used_formulas.add(new_p["perturbed_formula"])
            total_regenerated_for_cap += 1

        print(f"    Now have {len(perturbations)} perturbations")

        if rebalance_rounds > 20:
            raise RuntimeError("Element cap rebalancing did not converge in 20 rounds")

    assert len(perturbations) == N_TARGET
    return perturbations, total_dropped_for_cap, total_regenerated_for_cap


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_report(
    real_materials,
    initial_count,
    dropped_empty_check,
    n_regen_empty,
    dropped_cap_check,
    n_regen_cap,
    final_perturbations,
):
    """Write reports/stage_1.md."""
    report_path = ROOT / "reports" / "stage_1.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    real_formulas_list = [m["formula"] for m in real_materials]
    pert_formulas_list = [p["perturbed_formula"] for p in final_perturbations]
    real_freq = element_freq(real_formulas_list)
    pert_freq = element_freq(pert_formulas_list)

    # 10 random samples
    random.seed(77)
    samples = random.sample(final_perturbations, 10)

    lines = []
    lines.append("# Stage 1 Report: Data Construction Verification")
    lines.append("")
    lines.append("## Check 1: Perturbation Emptiness Verification")
    lines.append("")
    lines.append(f"- **Perturbations initially generated:** {initial_count}")
    lines.append(f"- **Dropped (MP returned data):** {len(dropped_empty_check)}")
    lines.append(f"- **Replacements regenerated:** {n_regen_empty}")
    lines.append(f"- **Final verified-empty count:** {len(final_perturbations)}")
    lines.append("")

    if dropped_empty_check:
        lines.append("### Examples of dropped perturbations (MP returned data)")
        lines.append("")
        lines.append("| Perturbed Formula | Source Formula | Strategy | MP Returned |")
        lines.append("|---|---|---|---|")
        for d in dropped_empty_check[:5]:
            mp_ids = ", ".join(r["material_id"] for r in d["mp_results"][:3])
            lines.append(
                f"| {d['formula']} | {d['source_formula']} | "
                f"{d['strategy']} | {mp_ids} |"
            )
        lines.append("")
    else:
        lines.append("No perturbations were dropped in the emptiness check; "
                      "all 200 returned empty on re-verification.")
        lines.append("")

    lines.append(f"**Runtime assertion passed:** all {len(final_perturbations)} "
                 f"perturbations confirmed empty via `mpr.materials.summary.search("
                 f"formula=...)`.")
    lines.append("")

    lines.append("## Check 2: Element Frequency Distribution")
    lines.append("")
    lines.append(f"- **Element cap threshold:** {ELEMENT_CAP:.0%} "
                 f"(max {int(ELEMENT_CAP * N_TARGET)} of {N_TARGET})")
    lines.append(f"- **Dropped for rebalancing:** {len(dropped_cap_check)}")
    lines.append(f"- **Replacements regenerated for rebalancing:** {n_regen_cap}")
    lines.append("")

    # Real materials frequency table
    lines.append("### Real materials element frequency (top 25)")
    lines.append("")
    lines.extend(freq_table_lines(real_freq, len(real_formulas_list), top_n=25))
    lines.append("")

    # Perturbed set frequency table
    lines.append("### Perturbed set element frequency (top 25)")
    lines.append("")
    lines.extend(freq_table_lines(pert_freq, len(pert_formulas_list), top_n=25))
    lines.append("")

    # Confirm cap
    max_pert_el = max(pert_freq, key=pert_freq.get)
    max_pert_cnt = pert_freq[max_pert_el]
    passed = max_pert_cnt <= int(ELEMENT_CAP * N_TARGET)
    lines.append(f"**25% threshold assertion: {'PASSED' if passed else 'FAILED'}** "
                 f"(most frequent element in perturbed set: {max_pert_el} at "
                 f"{max_pert_cnt}/{N_TARGET} = {max_pert_cnt/N_TARGET:.1%})")
    lines.append("")

    # Strategy distribution
    strat_counts = Counter(p["perturbation_type"] for p in final_perturbations)
    lines.append("### Perturbation strategy distribution")
    lines.append("")
    lines.append("| Strategy | Count |")
    lines.append("|---|---|")
    for s in sorted(strat_counts):
        lines.append(f"| {s} | {strat_counts[s]} |")
    lines.append("")

    # 10 samples
    lines.append("## 10 Random Samples for Spot-Check")
    lines.append("")
    lines.append("| # | Perturbed | Source | Strategy |")
    lines.append("|---|---|---|---|")
    for i, s in enumerate(samples, 1):
        lines.append(
            f"| {i} | {s['perturbed_formula']} | {s['source_formula']} | "
            f"{s['perturbation_type']} |"
        )
    lines.append("")

    report_text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"\nReport written to {report_path}")
    return report_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Stage 1 Verification")
    print("=" * 60)

    # Load existing data
    with open(ROOT / CFG["paths"]["materials"]) as f:
        data = json.load(f)

    real_materials = data["real_materials"]
    perturbations = data["perturbations"]
    initial_count = len(perturbations)

    print(f"Loaded {len(real_materials)} real materials, "
          f"{len(perturbations)} perturbations")

    with MPRester(MP_API_KEY) as mpr:
        # Check 1: verify emptiness
        verified, dropped_empty, n_regen_empty = verify_all(
            mpr, perturbations, real_materials
        )

        # Check 2: element frequency cap
        verified, dropped_cap, n_regen_cap = enforce_element_cap(
            mpr, verified, real_materials
        )

    # Save verified perturbations
    out_path = ROOT / "data" / "perturbations_verified.json"
    with open(out_path, "w") as f:
        json.dump(verified, f, indent=2)
    print(f"\nSaved {len(verified)} verified perturbations to {out_path}")

    # Write report
    report = write_report(
        real_materials,
        initial_count,
        dropped_empty,
        n_regen_empty,
        dropped_cap,
        n_regen_cap,
        verified,
    )

    print("\n" + report)


if __name__ == "__main__":
    main()
