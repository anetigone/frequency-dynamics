# Evidence and provenance

Paths below are relative to this directory. Existing results are read-only inputs to the draft and figure script.

The 2026-10-21 content-convergence version uses the encoder rebound and six-coordinate stability results as its core; see [independent core review](CORE_THEOREM_AUDIT_20261021.md) and [theorem-level literature comparison](THEOREM_LITERATURE_COMPARISON_20261021.md). The earlier projection certificate and detailed diagnostics below now appear in the supplement. The [fixed stochastic feasibility screen](STOCHASTIC_FEASIBILITY_20261021.md) rejects the present global-range instantiation; it adds no training and establishes no strict stochastic window.

| Manuscript content | Repository source |
|---|---|
| Supplementary projection fixed-gap theorem, initialization, threshold, interval proof | [THEORY_DRAFT.md](THEORY_DRAFT.md), §2.2, §4 and Appendix A |
| Exact certificate, 180-bit endpoints and 3,000 steps | [certificate.json](../../experiments/results/theory/certificate.json) |
| Reproduction algorithms | [verify.py](../../experiments/code/theory/verify.py), [interval_core.py](../../experiments/code/theory/interval_core.py) |
| 540 population trajectories and row-space means | [population findings](../../experiments/results/population/subspace_summary/findings.md), [summary](../../experiments/results/population/subspace_summary/summary.json) |
| Complete empirical protocol and 180 primary trajectories | [protocol](../../experiments/docs/FULL_EMPIRICAL_PROTOCOL.md), [closeout](../../experiments/docs/LOCAL_CLOSEOUT.md) |
| Main empirical displacement and repeatability | [main summary](../../experiments/results/empirical/full_sgd_main/summary.json) |
| Local empirical displacement | [local summary](../../experiments/results/empirical/full_sgd_local/summary.json) |
| Early/middle/late intervention results | [findings](../../experiments/results/empirical/full_sgd_interventions/findings.md), [summary](../../experiments/results/empirical/full_sgd_interventions/summary.json) |
| Five-seed CIFAR comparison | [summary](../../experiments/results/cifar/cifar_local_five/summary.json), [closeout](../../experiments/docs/LOCAL_CLOSEOUT.md) |
| Three-seed capacity diagnostics | [imported summary](../../experiments/results/cifar/capacity_development/summary.json), [interpretation](../../experiments/results/cifar/capacity_development/DIAGNOSTIC_REVIEW.md) |
| Frozen encoder: six states, three seeds | [imported summey.json](../../experiments/results/cifar/frozen_encoder_heads/summey.json), [interpretation](../../experiments/results/cifar/frozen_encoder_heads/FROZEN_HEAD_REVIEW.md) |
| Reduced-rank implementation | [frozen_encoder_probe.py](../../experiments/code/cifar/frozen_encoder_probe.py) |
| Natural-image objective decomposition and class learning | [diagnostic interpretation](NATURAL_IMAGE_DIAGNOSTIC_RESULTS_20261020.md) |
| DTD DC attribution, ablation and constrained training | [DC attribution](DTD_DOMINANT_DIRECTION_RESULTS_20261020.md), [ablation](DTD_DC_ABLATION_RESULTS_20261020.md), [DC-free diagnosis](DTD_DC_FREE_DIAGNOSIS_20261020.md) |
| CIFAR local direction coverage, reversals and effect sizes | [imported summary](../../experiments/results/cifar/risk_direction_development/summary.json), [protocol](../../experiments/docs/CIFAR_RISK_DIRECTION.md), [implementation](../../experiments/code/cifar/cifar_risk_direction.py) |
| Historical literature audit and current theorem-level comparison | [historical record](SOURCE_AUDIT.md), [current comparison](THEOREM_LITERATURE_COMPARISON_20261021.md) |

Bibliographic metadata was checked on the official NeurIPS page and versioned arXiv abstract pages on 2026-10-20. The CIFAR dataset page links the cited 2009 technical report. The subsequent theorem-level reading record is linked above; its stated reading scope does not establish exhaustive priority over all equivalent results.

The main abstract does not claim a natural-image improvement. Validation and test metrics are distinguished in every CIFAR comparison. The 98.84% frozen-head value is a ratio of mean validation MSE gaps, not a per-seed confidence statement or fraction of all information.
