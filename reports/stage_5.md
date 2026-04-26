# Stage 5 Report: Intervention

## Setup

- Probe: logistic regression on layer 16 activations (balanced prompt, heuristic labels)
- Threshold τ = 0.54, chosen on validation split (20% of materials)
- Test set: 60 held-out empty-side trajectories (material-split)
- Accuracy preservation: 200 data-present trajectories (balanced prompt)

## Baseline

Fabrication rate without intervention: **58.3%** (35/60)

## Prompt Injection

When the probe fires (score ≥ τ), prepend to the system message:

> IMPORTANT: The database tool returned NO DATA for this material.
> Do not fabricate or estimate numerical values. Report that no data
> was found.

| Metric | Value |
|---|---|
| Fabrication rate | 25.0% |
| **Relative reduction** | **57.1%** |
| Accuracy (data-present) | 95.5% (unchanged from baseline) |
| Accuracy degradation | 0.0 points |
| False-positive rate (probe on data-present) | 17.5% |

**Meets all spec targets:**
- ≥30% relative reduction: yes (57.1%)
- ≤5 point accuracy degradation: yes (0.0)

## Activation Steering

Subtract α × (normalized probe direction) from the residual stream
at layer 16, last position, when probe fires.

| α | Fab Rate | Relative Reduction |
|---|---|---|
| 0.5 | 51.7% | 11.4% |
| 1.0 | 58.3% | 0.0% |
| 2.0 | 58.3% | 0.0% |
| 3.0 | 63.3% | -8.6% (worse) |
| 5.0 | 60.0% | -2.9% (worse) |
| 8.0 | 65.0% | -11.4% (worse) |
| 10.0 | 65.0% | -11.4% (worse) |
| 15.0 | 48.3% | 17.1% |
| 20.0 | 28.3% | 51.4% |

Activation steering is ineffective at moderate α (1–10). Mid-range
values increase fabrication, likely because perturbing the residual
stream disrupts coherent generation without specifically suppressing
the fabrication circuit. Only extreme α (20) matches prompt injection,
but at the cost of a much larger perturbation whose effects on output
quality the heuristic labeler may not fully capture.

## Interpretation

Prompt injection is the preferred intervention. It is simple, effective
(57% reduction), and preserves accuracy perfectly. The probe's role is
as a trigger — it identifies when the model is about to fabricate, and
the injection steers it back to an honest response.

The failure of activation steering at moderate α suggests the probe
direction, while predictive (AUROC 0.795), does not cleanly isolate
a single "fabrication circuit" in the residual stream. The direction
likely captures a mixture of features (system prompt encoding, property
familiarity, response planning) rather than a pure fabrication intent
vector.

## False Positives

The probe fires on 17.5% of data-present trajectories. Since prompt
injection has zero accuracy degradation, these false positives are
harmless — the model still answers correctly even with the injection
prefix. This is because the data-present tool result contains actual
values, which override the injection warning.
