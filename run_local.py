"""Run the whole pipeline locally, in the same order SageMaker will run it.

    python run_local.py

No AWS account, no credentials, no containers. Same functions the pipeline
steps wrap, so if this passes, the cloud version is a packaging problem, not a
logic problem.
"""
from __future__ import annotations

import json

from src.batch_score import aggregate_candidates, score_batch
from src.evaluate import evaluate
from src.preprocess import preprocess
from src.register import register
from src.train import train

OUT = "artifacts/local"


def main() -> None:
    print("== 1/5 preprocess ==")
    p = preprocess(output_prefix=OUT)
    print(json.dumps({k: v for k, v in p.items() if not k.endswith("_uri")}, indent=2))

    print("\n== 2/5 train (GroupKFold by structure_id) ==")
    t = train(p["train_uri"], p["cutoffs_uri"], OUT)
    print("selected model:", t["selected_model"])
    print("rule baseline CV  false_pass_rate=%.4f macro_f1=%.4f"
          % (t["rule_baseline_cv"]["false_pass_rate"], t["rule_baseline_cv"]["macro_f1"]))
    for name, m in t["model_cv"].items():
        print("  %-20s false_pass_rate=%.4f macro_f1=%.4f pass_cut=%.2f"
              % (name, m["false_pass_rate"], m["macro_f1"], m["pass_cut"]))

    print("\n== 3/5 evaluate on held-out structures ==")
    e = evaluate(p["test_uri"], t["model_uri"], p["cutoffs_uri"], OUT)
    r = e["evaluation_result"]
    print("held-out structures:", r["held_out_structures"], "rows:", r["held_out_rows"])
    print("model         false_pass_rate=%.4f macro_f1=%.4f"
          % (r["model"]["false_pass_rate"], r["model"]["macro_f1"]))
    print("rule baseline false_pass_rate=%.4f macro_f1=%.4f"
          % (r["rule_baseline"]["false_pass_rate"], r["rule_baseline"]["macro_f1"]))

    print("\n== 4/5 conditional register ==")
    reg = register(r, t["model_uri"], OUT)
    print("registered:", reg["registered"], "| approval:", reg["approval_status"])

    print("\n== 5/5 batch score + candidate aggregation ==")
    s = score_batch(p["test_uri"], t["model_uri"], p["cutoffs_uri"], OUT)
    print("rows=%d  batch=%.3fs  %.3f ms/row  cpu=%s"
          % (s["rows_scored"], s["batch_seconds"], s["ms_per_row"], s["cpu_count"]))
    summary_uri = aggregate_candidates(s["scored_uri"], s["blocked_uri"], OUT)
    print("candidate summary ->", summary_uri)


if __name__ == "__main__":
    main()
