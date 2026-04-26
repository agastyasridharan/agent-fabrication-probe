"""
Stage 1 (Exoplanet domain): Data Construction

Fetches confirmed exoplanets from the NASA Exoplanet Archive with the four
properties of interest populated, plus generates plausible perturbations:
existing multi-planet hosts paired with the next unused planet letter.

Each perturbation is verified against the full set of known planet names —
we keep only ones that the archive has zero results for, so the agent's tool
will return empty at trajectory time.

Usage:
  python src/exoplanet_data_construction.py
  python src/exoplanet_data_construction.py --smoke-test
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

random.seed(CFG["seeds"]["random"])
np.random.seed(CFG["seeds"]["numpy"])

# pl_orbper: orbital period (days)
# pl_rade:   planet radius (Earth radii)
# pl_bmasse: best mass estimate (Earth masses)
# pl_eqt:    equilibrium temperature (K)
PROPERTIES = ["pl_orbper", "pl_rade", "pl_bmasse", "pl_eqt"]


def query_archive(where: str, select: str) -> list[dict]:
    tab = NasaExoplanetArchive.query_criteria(
        table="pscomppars", select=select, where=where,
    )
    return [{c: row[c] for c in tab.colnames} for row in tab]


def fetch_real_planets(n_target: int) -> list[dict]:
    print(f"Fetching real planets (target: {n_target})...")
    where = " AND ".join(f"{p} IS NOT NULL" for p in PROPERTIES)
    select = ",".join(["pl_name", "hostname", "sy_pnum"] + PROPERTIES)
    rows = query_archive(where, select)
    print(f"  Got {len(rows)} candidates with all four properties.")
    if len(rows) < n_target:
        raise RuntimeError(f"Only {len(rows)} candidates, need {n_target}")

    parsed = [
        {
            "pl_name": str(r["pl_name"]),
            "hostname": str(r["hostname"]),
            "sy_pnum": int(r["sy_pnum"]),
            **{p: float(r[p]) for p in PROPERTIES},
        }
        for r in rows
    ]
    random.shuffle(parsed)
    return parsed[:n_target]


def parse_letter(pl_name: str, hostname: str) -> str | None:
    suffix = pl_name[len(hostname):].strip()
    if len(suffix) == 1 and "b" <= suffix <= "z":
        return suffix
    return None


def next_letter(letters: list[str]) -> str | None:
    used = set(letters)
    for c in "bcdefghijklmnopqrstuvwxyz":
        if c not in used:
            return c
    return None


def fetch_multi_planet_systems() -> dict[str, list[str]]:
    print("Fetching multi-planet systems for perturbation source pool...")
    rows = query_archive(where="sy_pnum >= 2", select="pl_name,hostname")
    systems: dict[str, list[str]] = {}
    skipped = 0
    for r in rows:
        letter = parse_letter(str(r["pl_name"]), str(r["hostname"]))
        if letter is None:
            skipped += 1
            continue
        systems.setdefault(str(r["hostname"]), []).append(letter)
    print(f"  Got {len(systems)} hosts ({skipped} planets had unparseable suffixes)")
    return systems


def fetch_all_known_names() -> set[str]:
    print("Fetching the full known-name set for empty-verification...")
    rows = query_archive(where="pl_name IS NOT NULL", select="pl_name")
    print(f"  Loaded {len(rows)} known planet names.")
    return {str(r["pl_name"]) for r in rows}


def generate_perturbations(
    systems: dict[str, list[str]], known_names: set[str], n_target: int
) -> list[dict]:
    print(f"Generating {n_target} next-letter perturbations...")
    hosts = list(systems.keys())
    random.shuffle(hosts)

    perturbations = []
    for host in hosts:
        if len(perturbations) >= n_target:
            break
        nxt = next_letter(systems[host])
        if nxt is None:
            continue
        new_name = f"{host} {nxt}"
        if new_name in known_names:
            continue
        perturbations.append({
            "perturbed_name": new_name,
            "source_hostname": host,
            "existing_letters": sorted(systems[host]),
            "next_letter": nxt,
            "system_size": len(systems[host]),
        })

    if len(perturbations) < n_target:
        raise RuntimeError(f"Only generated {len(perturbations)}, need {n_target}")
    return perturbations


def assert_disjoint(perturbations: list[dict], known_names: set[str]) -> None:
    overlap = {p["perturbed_name"] for p in perturbations} & known_names
    assert not overlap, (
        f"{len(overlap)} perturbations match known names: {sorted(overlap)[:5]}"
    )


def summarize(real_planets: list[dict], perturbations: list[dict]) -> None:
    print("\n=== Summary ===")
    print(f"Real planets: {len(real_planets)}")
    for prop in PROPERTIES:
        vals = [r[prop] for r in real_planets]
        print(f"  {prop}: min={min(vals):.3g}, max={max(vals):.3g}, "
              f"median={np.median(vals):.3g}")
    n_multi = sum(1 for r in real_planets if r["sy_pnum"] >= 2)
    print(f"  Multi-planet systems: {n_multi}/{len(real_planets)}")

    print(f"\nPerturbations: {len(perturbations)}")
    print(f"  Next-letter distribution: "
          f"{dict(sorted(Counter(p['next_letter'] for p in perturbations).items()))}")
    print(f"  Source system size distribution: "
          f"{dict(sorted(Counter(p['system_size'] for p in perturbations).items()))}")
    print(f"\n  10 random samples:")
    for s in random.sample(perturbations, min(10, len(perturbations))):
        existing = "".join(s["existing_letters"])
        print(f"    {s['perturbed_name']:35s}  (system has {existing} → +{s['next_letter']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Pull 10 real planets + 10 perturbations only")
    args = parser.parse_args()

    print("Stage 1 (Exoplanet): Data Construction")
    print("=" * 60)

    n_real = 10 if args.smoke_test else 200
    n_pert = 10 if args.smoke_test else 200

    real_planets = fetch_real_planets(n_real)
    systems = fetch_multi_planet_systems()
    known_names = fetch_all_known_names()
    perturbations = generate_perturbations(systems, known_names, n_pert)
    assert_disjoint(perturbations, known_names)

    summarize(real_planets, perturbations)

    out_path = ROOT / "data" / "exoplanets.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "real_planets": real_planets,
        "perturbations": perturbations,
        "metadata": {
            "n_real": len(real_planets),
            "n_perturbations": len(perturbations),
            "properties": PROPERTIES,
            "seeds": CFG["seeds"],
        },
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
