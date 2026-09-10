# HER2 Pose Reliability Filter

A reliability gate that decides whether a predicted structural pose is
trustworthy enough to enter candidate ranking, and an MLOps pipeline around it.

The design question it answers: **a candidate must not be ranked highly because
of a pose that is physically implausible, and a tool failure must not look like
a weak biological result.**

## Layout

```
config/thresholds.v1.yaml   versioned, PROVISIONAL thresholds (not biological cut-offs)
src/contract.py             field contract, controlled values, direction normalisation
src/fixtures.py             synthetic pose rows (swap for real AEE/PFE rows)
src/gates.py                rankability gate + rule-based reliability baseline
src/split.py                grouped split by structure_id (no row-level leakage)
src/metrics.py              false_pass_rate (primary), macro-F1, Brier
src/preprocess.py           step 1 -- validate, gate, grouped split
src/train.py                step 2 -- GroupKFold model comparison, lock cut points
src/evaluate.py             step 3 -- one evaluation on held-out structures
src/register.py             step 4 -- conditional registration
src/batch_score.py          step 5 -- batch scoring + candidate aggregation
pipeline/build_pipeline.py  the same functions wrapped as a SageMaker Pipeline
tests/test_pipeline.py      contract, gate, leakage, idempotency, smoke tests
run_local.py                end-to-end local run, no AWS needed
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q
python run_local.py
```

Outputs land in `artifacts/local/`.

## Run on SageMaker

```bash
pip install -r requirements-cloud.txt
python -m pipeline.build_pipeline --bucket <your-bucket> --role <execution-role-arn> --start
```

## Claim boundary

`src/fixtures.py` generates synthetic rows so the pipeline is runnable before
real pose-level features exist. No metric produced from the fixture is a
biological result. The thresholds in `config/` are operational starting points
marked `calibration_status: uncalibrated`.
