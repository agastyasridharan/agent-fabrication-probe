# Tool-Null Confabulation Probe

A linear probe that detects when a language model is about to fabricate
numerical answers after its tool returns empty results. Tested on
Llama-3.1-8B-Instruct acting as a materials science research assistant.

## The Problem

AI agents with tool access sometimes fabricate plausible-looking data
when their tools return nothing. A materials science agent asked about
the band gap of a compound not in the database might respond with
"approximately 2.3 eV" instead of "no data found." These fabricated
numbers are indistinguishable from real tool outputs to downstream
consumers, silently contaminating experimental decisions.

## What This Project Does

We train a logistic regression probe on the model's internal activations
(the residual stream) at the moment just before it starts generating a
response. The probe predicts whether the model is about to fabricate or
honestly admit that no data was found. When the probe fires, we inject a
warning into the prompt that steers the model back to honesty.

The probe achieves **0.795 AUROC** on held-out materials it has never seen,
and the prompt-injection intervention reduces fabrication by **57%** with
zero accuracy degradation on queries where data exists.

## Key Results

| Metric | Value |
|---|---|
| Probe AUROC (within-prompt, material-split) | 0.795 |
| Best layer | 16 / 32 |
| Fabrication reduction (prompt injection) | 57.1% relative |
| Accuracy degradation on data-present queries | 0.0 points |
| Cross-tool transfer AUROC (MP to ChemDB) | 0.702 |
| Cross-template transfer AUROC (novel paraphrases) | 0.810 |
| Cross-chemistry AUROC (oxides to sulfides) | 0.787 |

## How It Works

### Data Construction

200 real materials are drawn from the Materials Project database, each with
band gap, formation energy per atom, and density values populated. 200
perturbed formulas are generated using three strategies (stoichiometry
shifts, elemental substitutions, and fictitious element additions), with
each perturbation verified to return zero results from the Materials Project
API.

### Trajectory Collection

The model receives a simulated tool-call exchange: a user asks about a
material property, the assistant calls the lookup tool, and the tool
returns either real data or an empty result. The model then generates its
response. We extract the residual-stream activation at the last prompt
token (before the first generated token) across all 33 layers.

### System Prompt Design

The fabrication rate is almost entirely determined by the system prompt.
Five variants were tested, and the results are striking:

- **honesty** ("you must clearly state no data was found"): 0% fabrication
- **pressure** ("approximate estimates are acceptable"): 0% fabrication
- **neutral** ("respond based on your best judgment"): 5.5% fabrication
- **balanced** ("if confident, you may estimate"): 53% fabrication
- **expert** ("draw on your deep knowledge"): 98.5% fabrication

The **balanced** prompt is the critical variant. It produces roughly equal
fabrication and admission rates within a single prompt, so any variation
in the model's behavior depends on the specific material and query rather
than the system prompt. This eliminates the system-prompt confound that
would otherwise make the probe trivial (just detect which prompt was used).

### Probe Training

A logistic regression probe is trained per-layer on the 4,096-dimensional
activation vectors from balanced-prompt trajectories only. The train/test
split is by material (not by trajectory), enforced by a runtime assertion.

The layer sweep reveals a non-monotonic curve peaking at layer 16, which
sits in the middle of the network. This is where the model has processed
the semantic content of the query and tool result but has not yet fully
committed to a generation plan.

### Intervention

When the probe score exceeds a threshold (chosen on a held-out validation
split), we prepend a warning to the system message: "The database tool
returned NO DATA. Do not fabricate or estimate numerical values." The model
responds to this guidance and avoids fabrication in 57% of cases where it
otherwise would have fabricated.

Activation steering (directly modifying the residual stream to subtract
the probe direction) was also tested but proved ineffective at moderate
perturbation strengths. The probe direction is predictive of fabrication
but does not cleanly isolate the causal mechanism.

## Repository Structure

```
SPEC.md                              Project specification
config.yaml                          Model revision, seeds, paths
requirements.txt                     Python dependencies

prompts/
  system_prompts.md                  5 system prompt variants
  templates.md                       15 query templates (5 per property)
  judge.md                           Claude judge labeling template

src/
  data_construction.py               Fetch materials, generate perturbations
  verify_perturbations.py            Verify perturbations return empty from MP
  agent_loop.py                      Trajectory collection + activation extraction
  run_expert_prompt.py               Expert prompt variant (GPU)
  run_balanced_prompt.py             Balanced prompt variant (GPU)
  labeling.py                        Claude judge labeling
  probe.py                           Probe training, layer sweep, splits
  baselines.py                       Baseline comparisons
  intervention.py                    Threshold selection, probe direction extraction
  run_intervention.py                Prompt injection + activation steering (GPU)
  run_transfer.py                    Cross-tool and cross-template transfer (GPU)

data/
  materials.json                     200 real + 200 perturbed materials
  perturbations_verified.json        Verified perturbations with MP API responses
  calibration_ids.json               50 calibration trajectory IDs
  hand_labels.json                   Human-verified labels for calibration set
  labels.json                        Judge labels for all 1000 empty-side trajectories
  probe_results.json                 Layer sweep results
  baseline_results.json              Baseline AUROC comparisons
  intervention_config.json           Probe threshold, alpha sweep values, split IDs
  intervention_results.json          Intervention outcomes
  transfer_results.json              Cross-tool and cross-template AUROC
  probe_direction.npy                Normalized probe weight vector (4096-dim)
  probe_bias.npy                     Probe bias term
  probe_weights_raw.npy              Raw (unnormalized) probe weights
  trajectories/
    all_trajectories.json            All 2000 trajectories with messages + responses
    balanced_trajectories.json       400 balanced-prompt trajectories
    expert_trajectories.json         400 expert-prompt trajectories
    activations/                     Per-trajectory .npy files (33 x 4096 each)

notebooks/
  explore_probe.ipynb                Interactive exploration and extension notebook

reports/
  stage_1.md                         Data construction verification
  stage_4.md                         Probe training and baselines
  stage_5.md                         Intervention results
  stage_6.md                         Transfer and ablations
  final_report.md                    Comprehensive writeup of all findings
```

## Reproducing the Results

### Prerequisites

- Python 3.12+
- A Hugging Face account with access to `meta-llama/Llama-3.1-8B-Instruct`
- An A100 GPU (or equivalent) for trajectory collection and intervention
- A Materials Project API key for data construction
- An Anthropic API key for judge labeling

### Local Setup

```bash
git clone https://github.com/agastyasridharan/bayes-and-confused.git
cd bayes-and-confused
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install scikit-learn anthropic
```

### Pipeline

Scripts marked **(GPU)** should be run on a machine with an A100 or
similar GPU. All other scripts run locally on CPU.

```bash
# Stage 1: Data construction (requires MP_API_KEY)
MP_API_KEY=... python src/data_construction.py
python src/verify_perturbations.py

# Stage 2: Trajectory collection (GPU)
python src/agent_loop.py
python src/run_balanced_prompt.py
python src/run_expert_prompt.py

# Stage 3: Labeling (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=... python src/labeling.py

# Stage 4: Probe training + baselines (local)
python src/probe.py
python src/baselines.py

# Stage 5: Intervention (GPU)
python src/intervention.py          # local prep: extract probe direction, choose threshold
python src/run_intervention.py      # GPU: run both intervention variants

# Stage 6: Transfer (GPU)
python src/run_transfer.py
```

### Using the Exploration Notebook

The notebook `notebooks/explore_probe.ipynb` lets you interactively
inspect activations, retrain the probe with different hyperparameters,
visualize the layer sweep, and test the probe on individual trajectories.
It runs locally (no GPU needed) since all activations are pre-extracted.

## Limitations

- **Single model.** All results are for Llama-3.1-8B-Instruct. The probe
  direction and layer sweep may differ on other architectures or scales.

- **Greedy decoding.** Trajectories use deterministic decoding. Under
  sampling with temperature > 0, the fabrication distribution changes.

- **Label simplicity.** The judge labels are 98.5% identical to a regex
  that checks for numbers with units. A more nuanced labeling scheme
  could reveal finer-grained probe behavior.

- **Moderate sample size.** 200 balanced-prompt trajectories (60 in the
  test set) give wide confidence intervals on the 0.795 AUROC.

- **Activation steering is ineffective.** The probe direction predicts
  fabrication but does not causally control it. Only prompt injection
  works as an intervention.

## License

MIT
