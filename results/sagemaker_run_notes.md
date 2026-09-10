# SageMaker processing-job run, 2026-09-10

pipeline: her2-pose-reliability-proc-10-17-53-18
execution: 17ijd07t2vaa

preprocess / train / evaluate / check-false-pass-rate : Succeeded
register : did not start (AWS-side instance allocation, not a code failure)

The condition step read evaluation.json and routed to the register branch,
so the gate was exercised end to end.

Held out (21 complexes, 630 poses), run inside the managed sklearn 1.2 container:
  model  false_pass_rate 0.0067  pass_recall 0.6850  macro_f1 0.6737
  rule   false_pass_rate 0.0000  pass_recall 0.2835  macro_f1 0.4599

Container environment: python 3.9.21, numpy 1.24.1, pandas 1.1.3, scikit-learn 1.2.1
Locked cut points: pass 0.62, fail 0.65

Local environment: python 3.12.7, numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.0
Locked cut points: pass 0.56, fail 0.55
Local held out: model false_pass_rate 0.0101, pass_recall 0.6850, macro_f1 0.6786

The rule baseline reproduces exactly across both runtimes. The learned model
differs in the third decimal because the cut-point grid search lands on an
adjacent grid point under different numeric libraries. Resume numbers come
from the local run, where the dependency versions are pinned.
