# Stage 6 Report: Transfer and Ablations

## Cross-Tool Transfer (Materials Project → ChemDB)

Probe trained on balanced-prompt Materials Project trajectories (layer 16),
applied zero-shot to trajectories from a synthetic "ChemDB" tool with
different schema, naming, and empty-result format (`null` instead of
`{"results": []}`).

| Metric | Value |
|---|---|
| **Cross-tool AUROC** | **0.702** |
| N trajectories | 200 |
| Fabrication rate | 6.0% |

**Passes spec threshold:** cross-tool AUROC > 0.65 (0.702 > 0.65).

The low fabrication rate (6%) under ChemDB suggests the different tool
schema and system prompt ("chemistry research assistant") makes the model
less likely to fabricate — the ChemDB framing is closer to the "neutral"
regime. Despite this class imbalance, the probe still achieves meaningful
discrimination (0.702), indicating transfer of the fabrication-intent
signal across tool boundaries.

## Cross-Template Transfer

Probe trained on templates 0-3, tested on:
- Template 4 (held out from training): "Report the {property} of {formula}
  from available materials data."
- 9 novel paraphrased templates never seen in training (3 per property).

| Metric | Value |
|---|---|
| **Cross-template AUROC** | **0.810** |
| N trajectories | 200 |
| Fabrication rate | 72.0% |

The cross-template AUROC (0.810) actually exceeds the within-template
test AUROC (0.795), suggesting the probe learned a template-invariant
fabrication signal rather than template-specific features. The higher
fabrication rate (72% vs 53%) under novel templates is expected — several
novel phrasings use "I need" framing which we identified in Stage 2 as
a fabrication trigger.

## Success Criteria Summary

| Criterion | Target | Result | Status |
|---|---|---|---|
| Probe peak AUROC (material-split) | > 0.75 | 0.795 | **Passed** |
| Intervention fabrication reduction | ≥ 30% relative | 57.1% | **Passed** |
| Cross-tool transfer AUROC | > 0.65 | 0.702 | **Passed** |
| Accuracy degradation | < 5 points | 0.0 points | **Passed** |

## Invalidation Conditions

| Condition | Status |
|---|---|
| Trivial regex > 0.95 | Passed (0.500) |
| Train/test material overlap | Passed (asserted disjoint) |
| Layer sweep flat or monotone | Passed (range 0.295) |
| Judge-vs-hand agreement < 85% | Passed (100%) |
| FABRICATE or ADMIT < 5% | Passed (53%/47%) |
| Wrong contrast dominates | Passed (within-prompt evaluation) |

No invalidation conditions triggered.
