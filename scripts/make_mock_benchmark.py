"""Generate a table with the exact shape of a PoseBusters full report.

    python -m scripts.make_mock_benchmark

Purpose: verify the mapping and the loader BEFORE downloading several GB.
The column names below are the real PoseBusters full-report names (verified
against posebusters 0.6.x), so a mapping that works here works on real output.

The VALUES are synthetic. Never report a metric produced from this file.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

OUT = "data/mock_benchmark_raw.csv"

BOOL_COLS = [
    "sanitization", "inchi_convertible", "all_atoms_connected", "no_radicals",
    "bond_lengths", "bond_angles", "internal_steric_clash",
    "aromatic_ring_flatness", "non-aromatic_ring_non-flatness",
    "double_bond_flatness", "internal_energy",
]


def main(n_structures: int = 60, poses_per_structure: int = 20, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_structures):
        pdb_id = f"{rng.integers(1,9)}{''.join(rng.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), 3))}"
        ligand_id = f"LIG{s:03d}"
        quality = rng.beta(2.0, 2.0)
        for p in range(poses_per_structure):
            good = rng.random() < quality
            rmsd = abs(rng.normal(1.2 if good else 6.0, 1.1))
            energy = rng.normal(-8.0 * quality, 1.5)
            row = {
                "pdb_id": pdb_id,
                "ligand_id": ligand_id,
                "file": f"{pdb_id}_pose_{p:02d}.sdf",
                "method": rng.choice(["vina", "diffdock", "af3"]),
                "rmsd": rmsd,
                "mol_pred_energy": energy,
                "energy_ratio": abs(rng.normal(1.0 + 0.8 * (1 - quality), 0.4)),
                "number_clashes": int(rng.poisson(0.3 + 3.0 * (1 - quality))),
                "shortest_noncovalent_relative_distance": float(
                    np.clip(rng.normal(0.55 + 0.4 * quality, 0.12), 0, 1)
                ),
                "number_bonds": int(rng.integers(12, 40)),
            }
            for b in BOOL_COLS:
                row[b] = bool(rng.random() < (0.55 + 0.45 * quality))
            rows.append(row)

    df = pd.DataFrame(rows)
    # a realistic share of predictions simply fail to produce output
    fail = rng.choice(len(df), size=int(0.03 * len(df)), replace=False)
    df.loc[fail, ["rmsd", "mol_pred_energy"]] = np.nan

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {df.shape[0]} rows x {df.shape[1]} columns, "
          f"{df.pdb_id.nunique()} structures, {int(df.rmsd.isna().sum())} failed rows")


if __name__ == "__main__":
    main()
