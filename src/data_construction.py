"""
Stage 1: Data Construction

Fetches 200 real materials from Materials Project (all with band_gap,
formation_energy_per_atom, and density populated), generates 200 plausible
perturbations verified to return empty from the API, and saves everything
to data/materials.json.

Perturbation strategies (tight):
  1. Stoichiometry: ±1 or ±2 on exactly one coefficient.
  2. Substitution: swap exactly one element symbol, stoichiometry unchanged.
  3. Fictitious: add or remove one element with count ≤ 3.

Each perturbation is verified empty via a runtime assertion.
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

MP_API_KEY = os.environ.get("MP_API_KEY")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

random.seed(CFG["seeds"]["random"])
np.random.seed(CFG["seeds"]["numpy"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = ["material_id", "formula_pretty", "band_gap",
                   "formation_energy_per_atom", "density"]
N_REAL = CFG["data"]["n_real_materials"]
N_PERT = CFG["data"]["n_perturbations"]

# Substitution map: each element maps to plausible chemical neighbors
SUBSTITUTION_MAP = {
    "O": ["S", "Se", "Te"],
    "S": ["O", "Se", "Te"],
    "N": ["P", "As"],
    "Si": ["Ge", "Sn"],
    "Al": ["Ga", "In"],
    "Fe": ["Co", "Ni", "Mn"],
    "Ti": ["Zr", "Hf"],
    "Ca": ["Sr", "Ba"],
    "Li": ["Na", "K"],
    "Mg": ["Be", "Zn"],
    "Cu": ["Ag", "Au"],
    "Zn": ["Cd", "Hg"],
    "C": ["Si", "Ge"],
    "P": ["As", "Sb"],
    "Se": ["Te", "S"],
    "Cl": ["Br", "I"],
    "F": ["Cl", "Br"],
    "Na": ["K", "Rb"],
    "K": ["Rb", "Cs"],
    "Ba": ["Sr", "Ra"],
    "Mn": ["Cr", "Re"],
    "Co": ["Rh", "Ir"],
    "Ni": ["Pd", "Pt"],
    "Cr": ["Mo", "W"],
    "V": ["Nb", "Ta"],
    "Sr": ["Ca", "Ba"],
    "Ga": ["Al", "In"],
    "Ge": ["Si", "Sn"],
    "Bi": ["Sb", "As"],
    "Sn": ["Ge", "Pb"],
    "Pb": ["Sn", "Bi"],
    "Y": ["Sc", "La"],
    "La": ["Y", "Ce"],
    "Ce": ["La", "Pr"],
    "Eu": ["Sm", "Gd"],
    "W": ["Mo", "Cr"],
    "Mo": ["W", "Cr"],
    "Te": ["Se", "S"],
    "Sb": ["Bi", "As"],
    "Br": ["Cl", "I"],
    "I": ["Br", "Cl"],
    "Ag": ["Cu", "Au"],
    "Au": ["Ag", "Cu"],
    "Rb": ["K", "Cs"],
    "Cs": ["Rb", "K"],
    "Hg": ["Cd", "Zn"],
    "Cd": ["Zn", "Hg"],
    "In": ["Ga", "Tl"],
    "Tl": ["In", "Ga"],
    "Ru": ["Os", "Fe"],
    "Rh": ["Ir", "Co"],
    "Pd": ["Pt", "Ni"],
    "Pt": ["Pd", "Ni"],
    "Ir": ["Rh", "Co"],
    "Os": ["Ru", "Fe"],
    "Hf": ["Zr", "Ti"],
    "Zr": ["Hf", "Ti"],
    "Nb": ["Ta", "V"],
    "Ta": ["Nb", "V"],
    "Sc": ["Y", "La"],
    "B": ["Al", "Ga"],
    "Be": ["Mg", "Zn"],
    "Sm": ["Eu", "Nd"],
}

# Elements that can be added in the "fictitious" strategy — drawn from the
# same pool the real materials use so the distribution stays similar
ADDABLE_ELEMENTS = [
    "Li", "Be", "B", "C", "N", "F", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Cs", "Ba",
    "La", "Ce", "Hf", "Ta", "W", "Pt", "Au", "Pb", "Bi",
]


def fetch_real_materials(mpr: MPRester) -> list[dict]:
    """Fetch real materials with all three properties populated."""
    print(f"Fetching real materials from Materials Project (target: {N_REAL})...")

    docs = mpr.materials.summary.search(
        band_gap=(0.01, None),
        density=(0.1, None),
        fields=REQUIRED_FIELDS,
        num_chunks=10,
    )

    valid = []
    for doc in docs:
        rec = {
            "material_id": str(doc.material_id),
            "formula": doc.formula_pretty,
            "band_gap": float(doc.band_gap),
            "formation_energy_per_atom": float(doc.formation_energy_per_atom),
            "density": float(doc.density),
        }
        if all(rec[p] is not None for p in ["band_gap", "formation_energy_per_atom", "density"]):
            valid.append(rec)

    print(f"  Found {len(valid)} valid materials with all properties populated.")

    if len(valid) < N_REAL:
        raise RuntimeError(
            f"Only found {len(valid)} materials, need {N_REAL}. "
            "Relax filters or check API connectivity."
        )

    random.shuffle(valid)
    selected = valid[:N_REAL]
    print(f"  Selected {len(selected)} materials.")
    return selected


# ---------------------------------------------------------------------------
# Formula helpers
# ---------------------------------------------------------------------------

def _parse_formula(formula: str) -> list[tuple[str, int]]:
    """Parse 'Li2Fe3O4' -> [('Li', 2), ('Fe', 3), ('O', 4)]."""
    tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    return [(el, int(n) if n else 1) for el, n in tokens if el]


def _formula_str(parsed: list[tuple[str, int]]) -> str:
    return "".join(f"{el}{n if n > 1 else ''}" for el, n in parsed)


def _elements_in(formula: str) -> set[str]:
    return {el for el, _ in _parse_formula(formula)}


# ---------------------------------------------------------------------------
# Tight perturbation generators
# ---------------------------------------------------------------------------

def perturb_stoichiometry(formula: str) -> str | None:
    """±1 or ±2 on exactly one coefficient."""
    parsed = _parse_formula(formula)
    if len(parsed) < 2:
        return None
    idx = random.randrange(len(parsed))
    el, n = parsed[idx]
    delta = random.choice([-2, -1, 1, 2])
    new_n = n + delta
    if new_n < 1:
        return None
    parsed[idx] = (el, new_n)
    return _formula_str(parsed)


def perturb_substitution(formula: str) -> str | None:
    """Swap exactly one element symbol; stoichiometry unchanged."""
    parsed = _parse_formula(formula)
    candidates = [(i, el) for i, (el, _) in enumerate(parsed) if el in SUBSTITUTION_MAP]
    if not candidates:
        return None
    idx, el = random.choice(candidates)
    replacement = random.choice(SUBSTITUTION_MAP[el])
    # Don't substitute to an element already in the formula
    existing_els = {e for e, _ in parsed}
    if replacement in existing_els:
        return None
    _, n = parsed[idx]
    parsed[idx] = (replacement, n)
    return _formula_str(parsed)


def perturb_fictitious(formula: str) -> str | None:
    """Add or remove one element with count ≤ 3."""
    parsed = _parse_formula(formula)
    existing_els = {e for e, _ in parsed}

    if len(parsed) >= 2 and random.random() < 0.4:
        # Remove one element
        idx = random.randrange(len(parsed))
        new_parsed = [p for i, p in enumerate(parsed) if i != idx]
        if not new_parsed:
            return None
        return _formula_str(new_parsed)
    else:
        # Add one element not already present
        pool = [e for e in ADDABLE_ELEMENTS if e not in existing_els]
        if not pool:
            return None
        new_el = random.choice(pool)
        count = random.randint(1, 3)
        parsed.append((new_el, count))
        return _formula_str(parsed)


PERTURBATION_FNS = {
    "stoichiometry": perturb_stoichiometry,
    "substitution": perturb_substitution,
    "fictitious": perturb_fictitious,
}


def assert_empty(mpr: MPRester, formula: str) -> None:
    """Runtime assertion: querying this formula must return no MP results."""
    docs = mpr.materials.summary.search(formula=formula, fields=["material_id"])
    assert len(docs) == 0, (
        f"Perturbation '{formula}' returned {len(docs)} results from MP — "
        f"not empty! IDs: {[str(d.material_id) for d in docs[:5]]}"
    )


def _element_freq(formulas: list[str]) -> Counter:
    """Count element occurrences across a list of formulas."""
    c = Counter()
    for f in formulas:
        for el, _ in _parse_formula(f):
            c[el] += 1
    return c


def generate_perturbations(mpr: MPRester, real_materials: list[dict]) -> list[dict]:
    """Generate N_PERT perturbations, each verified empty via assertion."""
    print(f"Generating {N_PERT} perturbations (verified empty)...")

    perturbations = []
    attempts = 0
    max_attempts = N_PERT * 20

    strategies = list(PERTURBATION_FNS.keys())
    used_formulas = set()
    real_formulas = {m["formula"] for m in real_materials}

    # Round-robin through a shuffled copy of real materials so we don't
    # over-draw from the first few
    source_order = list(range(len(real_materials)))
    random.shuffle(source_order)

    while len(perturbations) < N_PERT and attempts < max_attempts:
        src_idx = source_order[attempts % len(source_order)]
        source = real_materials[src_idx]
        strategy = random.choice(strategies)
        fn = PERTURBATION_FNS[strategy]

        perturbed_formula = fn(source["formula"])
        attempts += 1

        if perturbed_formula is None:
            continue
        if perturbed_formula in used_formulas or perturbed_formula in real_formulas:
            continue

        # Verify empty — runtime assertion
        try:
            assert_empty(mpr, perturbed_formula)
        except AssertionError:
            continue

        perturbations.append({
            "perturbed_formula": perturbed_formula,
            "source_material_id": source["material_id"],
            "source_formula": source["formula"],
            "perturbation_type": strategy,
        })
        used_formulas.add(perturbed_formula)
        if len(perturbations) % 20 == 0:
            print(f"  {len(perturbations)}/{N_PERT} verified perturbations...")

        # Polite rate limiting
        if attempts % 10 == 0:
            time.sleep(0.3)

    if len(perturbations) < N_PERT:
        raise RuntimeError(
            f"Only generated {len(perturbations)} verified-empty perturbations "
            f"after {attempts} attempts. Need {N_PERT}."
        )

    print(f"  Generated {len(perturbations)} perturbations in {attempts} attempts.")
    return perturbations


def print_element_distribution(real_materials: list[dict], perturbations: list[dict]):
    """Print and compare element frequency across real and perturbed sets."""
    real_freq = _element_freq([m["formula"] for m in real_materials])
    pert_freq = _element_freq([p["perturbed_formula"] for p in perturbations])

    all_els = sorted(set(real_freq) | set(pert_freq),
                     key=lambda e: real_freq.get(e, 0), reverse=True)

    print("\n=== Element Frequency Distribution ===")
    print(f"  {'El':>3s}  {'Real':>5s}  {'Pert':>5s}  {'Diff':>5s}")
    print(f"  {'---':>3s}  {'-----':>5s}  {'-----':>5s}  {'-----':>5s}")
    for el in all_els[:30]:
        r = real_freq.get(el, 0)
        p = pert_freq.get(el, 0)
        diff = p - r
        flag = " **" if abs(diff) > r * 0.5 and r > 5 else ""
        print(f"  {el:>3s}  {r:5d}  {p:5d}  {diff:+5d}{flag}")

    if len(all_els) > 30:
        print(f"  ... and {len(all_els) - 30} more elements")


def summarize(real_materials: list[dict], perturbations: list[dict]):
    """Print summary statistics."""
    print("\n=== Data Construction Summary ===")
    print(f"Real materials: {len(real_materials)}")

    formulas = [m["formula"] for m in real_materials]
    n_oxides = sum(1 for f in formulas if any(el == "O" for el, _ in _parse_formula(f)))
    n_sulfides = sum(1 for f in formulas if any(el == "S" for el, _ in _parse_formula(f)))
    print(f"  Oxides: {n_oxides}, Sulfides: {n_sulfides}")

    for prop in ["band_gap", "formation_energy_per_atom", "density"]:
        vals = [m[prop] for m in real_materials]
        print(f"  {prop}: min={min(vals):.3f}, max={max(vals):.3f}, "
              f"mean={np.mean(vals):.3f}, std={np.std(vals):.3f}")

    print(f"\nPerturbations: {len(perturbations)}")
    for strategy in PERTURBATION_FNS:
        n = sum(1 for p in perturbations if p["perturbation_type"] == strategy)
        print(f"  {strategy}: {n}")

    print_element_distribution(real_materials, perturbations)


def main():
    print("Stage 1: Data Construction")
    print("=" * 50)

    out_path = ROOT / CFG["paths"]["materials"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with MPRester(MP_API_KEY) as mpr:
        real_materials = fetch_real_materials(mpr)
        perturbations = generate_perturbations(mpr, real_materials)

    summarize(real_materials, perturbations)

    output = {
        "real_materials": real_materials,
        "perturbations": perturbations,
        "metadata": {
            "n_real": len(real_materials),
            "n_perturbations": len(perturbations),
            "seeds": CFG["seeds"],
            "perturbation_strategies": list(PERTURBATION_FNS.keys()),
        },
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {out_path}")
    print("Stage 1 complete. Ready for human review (gate 1).")


if __name__ == "__main__":
    main()
