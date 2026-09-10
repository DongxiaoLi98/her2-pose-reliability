"""Inventory an extracted benchmark directory and build the PoseBusters input list.

    python -m scripts.build_bust_input data/posebusters_benchmark_set

Written to be layout-agnostic: it reports what it found before writing anything,
so you can see whether the guess is right instead of discovering it three hours
into a `bust` run.

Outputs data/bust_input.csv with the three columns `bust -t` expects:
    mol_pred   the pose to check
    mol_true   the crystal ligand (gives you rmsd -> the label)
    mol_cond   the protein (gives you the protein-distance checks)
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

TRUE_HINTS = ("_ligand.sdf", "_ligands.sdf", "ligand.sdf")
PROT_HINTS = ("_protein.pdb", "protein.pdb", "_protein_processed.pdb")
PRED_EXCLUDE = ("_ligand.sdf", "_ligands.sdf")


def inventory(root: Path) -> None:
    print(f"scanning {root}\n")
    subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    print(f"  top-level subdirectories: {len(subdirs)}")
    if subdirs:
        print(f"  first few: {[d.name for d in subdirs[:5]]}")
    ext = Counter(p.suffix.lower() for p in root.rglob("*") if p.is_file())
    print(f"  file extensions: {dict(ext.most_common(10))}")
    csvs = list(root.rglob("*.csv"))
    if csvs:
        print(f"\n  !! {len(csvs)} CSV file(s) already present -- check these first,")
        print("     they may already contain rmsd and QC columns:")
        for c in csvs[:10]:
            print(f"       {c}")
    print()


def build(root: Path, out: Path = Path("data/bust_input.csv")) -> None:
    inventory(root)

    rows, skipped = [], Counter()
    cases = [d for d in sorted(root.iterdir()) if d.is_dir()]
    for case in cases:
        sdfs = sorted(case.rglob("*.sdf"))
        pdbs = sorted(case.rglob("*.pdb")) + sorted(case.rglob("*.cif"))

        true = next((p for p in sdfs if p.name.endswith(TRUE_HINTS)), None)
        prot = next((p for p in pdbs if p.name.endswith(PROT_HINTS)), None)
        if prot is None and pdbs:
            prot = pdbs[0]
        preds = [p for p in sdfs if not p.name.endswith(PRED_EXCLUDE)]

        if true is None:
            skipped["no crystal ligand (mol_true)"] += 1
            continue
        if not preds:
            # redocking the crystal pose against itself is still a valid row:
            # it is the positive control the gate must never reject
            preds = [true]
            skipped["only crystal pose available"] += 1

        for p in preds:
            rows.append({
                "mol_pred": str(p),
                "mol_true": str(true),
                "mol_cond": str(prot) if prot else "",
            })

    if not rows:
        sys.exit(
            "no poses found. The layout differs from what this script expects.\n"
            "Run:  find <root> -maxdepth 3 | head -40\n"
            "and adjust TRUE_HINTS / PROT_HINTS above."
        )

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"wrote {out}: {len(df)} poses across {len(cases)} cases")
    print(f"  cases with a protein file: {(df['mol_cond'] != '').sum()} rows")
    for reason, n in skipped.items():
        print(f"  note: {reason}: {n} cases")
    print("\nnext:")
    print(f"  bust -t {out} --outfmt csv --full-report --output data/benchmark_raw.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m scripts.build_bust_input <extracted_dir>")
    build(Path(sys.argv[1]))
