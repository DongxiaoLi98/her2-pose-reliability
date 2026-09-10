"""Run the full pipeline on a public benchmark table.

    python -m scripts.make_mock_benchmark            # offline dry run
    python run_benchmark.py --table data/mock_benchmark_raw.csv

    # then, with the real download:
    python run_benchmark.py --table data/benchmark_raw.csv

Prints exactly the numbers the resume needs, and writes them to
artifacts/benchmark/resume_numbers.json.
"""
from __future__ import annotations

import argparse
import json

from src.batch_score import aggregate_candidates, score_batch
from src.evaluate import evaluate
from src.io_utils import write_json
from src.preprocess import preprocess
from src.register import register
from src.train import train

OUT = "artifacts/benchmark"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="downloaded benchmark CSV")
    ap.add_argument("--mapping", default="config/benchmark_mapping.yaml")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("== 1/5 preprocess ==")
    p = preprocess(output_prefix=args.out, input_uri=args.table,
                   source="benchmark", mapping_path=args.mapping, seed=args.seed)
    print(json.dumps({k: v for k, v in p.items() if not k.endswith("_uri")},
                     indent=2, default=str))

    print("\n== 2/5 train (GroupKFold by structure_id) ==")
    t = train(p["train_uri"], p["cutoffs_uri"], args.out, seed=args.seed)
    sel = t["model_cv"][t["selected_model"]]
    print("selected model:", t["selected_model"])
    print("rule baseline CV  false_pass_rate=%.4f macro_f1=%.4f"
          % (t["rule_baseline_cv"]["false_pass_rate"], t["rule_baseline_cv"]["macro_f1"]))
    for name, m in t["model_cv"].items():
        print("  %-20s pooled_fpr=%.4f worst_fold_fpr=%.4f macro_f1=%.4f pass_cut=%.2f fail_cut=%.2f"
              % (name, m["oof_false_pass_rate"], m["oof_worst_fold_false_pass_rate"],
                 m["macro_f1"], m["pass_cut"], m["fail_cut"]))

    print("\n== 3/5 evaluate on held-out structures ==")
    e = evaluate(p["test_uri"], t["model_uri"], p["cutoffs_uri"], args.out)
    r = e["evaluation_result"]
    hdr = "%-14s %10s %10s %10s %10s %9s" % ("", "fpr", "pass_yld", "pass_rcl", "review", "macroF1")
    row = "%-14s %10.4f %10.4f %10.4f %10.4f %9.4f"
    print(hdr)
    print(row % ("model", r["model"]["false_pass_rate"], r["model"]["pass_yield"],
                 r["model"]["pass_recall"], r["model"]["review_rate"], r["model"]["macro_f1"]))
    print(row % ("rule baseline", r["rule_baseline"]["false_pass_rate"],
                 r["rule_baseline"]["pass_yield"], r["rule_baseline"]["pass_recall"],
                 r["rule_baseline"]["review_rate"], r["rule_baseline"]["macro_f1"]))
    print("  a gate that passes almost nothing has a near-zero fpr and filters nothing;"
          "\n  read fpr only together with pass_yield / pass_recall.")
    print("cv worst fold %.4f  ->  held out %.4f   (gap %.4f)"
          % (sel["oof_worst_fold_false_pass_rate"], r["model"]["false_pass_rate"],
             r["model"]["false_pass_rate"] - sel["oof_worst_fold_false_pass_rate"]))

    print("\n== 4/5 conditional register ==")
    reg = register(r, t["model_uri"], args.out)
    print("registered:", reg["registered"], "| approval:", reg["approval_status"])

    print("\n== 5/5 batch score + candidate aggregation ==")
    s = score_batch(p["test_uri"], t["model_uri"], p["cutoffs_uri"], args.out)
    print("rows=%d  batch=%.3fs  %.3f ms/row  cpu=%s"
          % (s["rows_scored"], s["batch_seconds"], s["ms_per_row"], s["cpu_count"]))
    aggregate_candidates(s["scored_uri"], s["blocked_uri"], args.out)

    numbers = {
        "dataset_source": p["provenance"]["source_name"],
        "substitutions_made": p["provenance"].get("substitutions", []),
        "environment": t["environment"],
        "rows_total": p["n_raw_rows"],
        "rows_scorable": p["n_scorable_rows"],
        "rows_blocked": p["n_blocked_rows"],
        "blocked_breakdown": p["blocked_breakdown"],
        "n_classes": 3,
        "classes": ["Pass", "Review", "Fail"],
        "train_rows": p["n_train_rows"],
        "train_structures": p["n_train_structures"],
        "test_rows": p["n_test_rows"],
        "test_structures": p["n_test_structures"],
        "split": "leave-structures-out holdout; GroupKFold by structure_id inside train",
        "models_compared": ["rule_baseline"] + list(t["model_cv"].keys()),
        "selected_model": t["selected_model"],
        "selection_rule": t["selection_rule"],
        "cut_points_locked_on": "out-of-fold predictions, scored by worst fold",
        "pass_cut": t["locked_pass_cut"],
        "fail_cut": t["locked_fail_cut"],
        "heldout_model_false_pass_rate": r["model"]["false_pass_rate"],
        "heldout_baseline_false_pass_rate": r["rule_baseline"]["false_pass_rate"],
        "cv_pooled_false_pass_rate": sel["oof_false_pass_rate"],
        "cv_worst_fold_false_pass_rate": sel["oof_worst_fold_false_pass_rate"],
        # If this gap is large, the cut point does not transfer to a new
        # structure. Report it; do not tune it away.
        "cv_to_heldout_gap_false_pass_rate":
            r["model"]["false_pass_rate"] - sel["oof_worst_fold_false_pass_rate"],
        "heldout_model_pass_yield": r["model"]["pass_yield"],
        "heldout_baseline_pass_yield": r["rule_baseline"]["pass_yield"],
        "heldout_model_pass_recall": r["model"]["pass_recall"],
        "heldout_baseline_pass_recall": r["rule_baseline"]["pass_recall"],
        "heldout_model_review_rate": r["model"]["review_rate"],
        "heldout_baseline_review_rate": r["rule_baseline"]["review_rate"],
        # A lower false-pass rate only counts if the gate passes a comparable
        # amount. Require BOTH: at least as safe AND at least as productive.
        "beats_baseline_on_primary_metric":
            bool(r["model"]["false_pass_rate"] <= r["rule_baseline"]["false_pass_rate"]
                 and r["model"]["pass_recall"] >= r["rule_baseline"]["pass_recall"]),
        "safer_than_baseline": bool(r["model"]["false_pass_rate"] <= r["rule_baseline"]["false_pass_rate"]),
        "more_productive_than_baseline": bool(r["model"]["pass_recall"] >= r["rule_baseline"]["pass_recall"]),
        "heldout_model_macro_f1": r["model"]["macro_f1"],
        "heldout_baseline_macro_f1": r["rule_baseline"]["macro_f1"],
        "delta_false_pass_rate": r["delta_false_pass_rate"],
        "delta_macro_f1": r["delta_macro_f1"],
        "inference_rows": s["rows_scored"],
        "inference_ms_per_row": s["ms_per_row"],
        "inference_cpu_count": s["cpu_count"],
    }
    write_json(numbers, f"{args.out}/resume_numbers.json")

    print("\n" + "=" * 66)
    print("RESUME NUMBERS  ->  %s/resume_numbers.json" % args.out)
    print("=" * 66)
    for k, v in numbers.items():
        print(f"  {k:<38} {v}")


if __name__ == "__main__":
    main()
