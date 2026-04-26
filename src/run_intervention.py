"""
Stage 5: Intervention (GPU required)

Two intervention variants:
  1. Prompt-injection: when probe fires, prepend a warning message
  2. Activation-space: subtract α × probe direction from residual stream

Reports fabrication rate with/without intervention on held-out empties,
and accuracy preservation on data-present queries.

Usage:
  python src/run_intervention.py                # full run
  python src/run_intervention.py --smoke-test   # 5 trajectories
"""

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

random.seed(CFG["seeds"]["random"])
np.random.seed(CFG["seeds"]["numpy"])
torch.manual_seed(CFG["seeds"]["torch"])

MODEL_NAME = CFG["model"]["name"]
MODEL_REVISION = CFG["model"]["revision"]
_PRECISION_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
PRECISION = _PRECISION_MAP[CFG["model"]["precision"]]

import sys
sys.path.insert(0, str(ROOT / "src"))
from agent_loop import TOOL_DEF, build_messages

# Load intervention config
with open(ROOT / "data" / "intervention_config.json") as f:
    INT_CFG = json.load(f)

BEST_LAYER = INT_CFG["best_layer"]
THRESHOLD = INT_CFG["threshold_tau"]
ALPHA_SWEEP = INT_CFG["alpha_sweep"]

# Load probe artifacts
PROBE_DIRECTION = np.load(ROOT / "data" / "probe_direction.npy")
PROBE_WEIGHTS = np.load(ROOT / "data" / "probe_weights_raw.npy")
PROBE_BIAS = np.load(ROOT / "data" / "probe_bias.npy")[0]

# Heuristic labeler
def heuristic_label(resp):
    number_pat = re.compile(r'\b\d+\.?\d*\s*(?:eV|g/cm|meV|J|kJ|GPa)')
    estimate_pat = re.compile(
        r'(?:approximately|around|roughly|about|estimated?|~)\s*\d+\.?\d*',
        re.IGNORECASE,
    )
    range_pat = re.compile(
        r'\d+\.?\d*\s*[-–to]+\s*\d+\.?\d*\s*(?:eV|g/cm|meV)',
        re.IGNORECASE,
    )
    if number_pat.search(resp) or estimate_pat.search(resp) or range_pat.search(resp):
        return 1
    return 0


# Balanced system prompt
SYSTEM_PROMPTS = {}
_current_key = None
with open(ROOT / "prompts" / "system_prompts.md") as f:
    for line in f:
        line = line.rstrip()
        if line.startswith("## "):
            _current_key = line[3:].strip()
            SYSTEM_PROMPTS[_current_key] = []
        elif _current_key and line and not line.startswith("#"):
            SYSTEM_PROMPTS[_current_key].append(line)
SYSTEM_PROMPTS = {k: " ".join(v) for k, v in SYSTEM_PROMPTS.items()}

BALANCED_PROMPT = SYSTEM_PROMPTS["balanced"]
INJECTION_PREFIX = (
    "IMPORTANT: The database tool returned NO DATA for this material. "
    "Do not fabricate or estimate numerical values. Report that no data "
    "was found."
)


def probe_score(activation_layer):
    """Compute probe probability from a single layer activation."""
    logit = float(np.dot(PROBE_WEIGHTS, activation_layer) + PROBE_BIAS)
    return 1.0 / (1.0 + np.exp(-logit))


def generate_baseline(model, tokenizer, messages, max_new_tokens=300):
    """Standard generation. Returns response text and last-token activation."""
    prompt = tokenizer.apply_chat_template(
        messages, tools=[TOOL_DEF], tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    # Forward pass for activation
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    activation = fwd.hidden_states[BEST_LAYER][0, -1, :].float().cpu().numpy()

    # Generate
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=None, top_p=None, pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(out[0, input_len:], skip_special_tokens=True)
    return response, activation


def generate_with_injection(model, tokenizer, messages_original, max_new_tokens=300):
    """Re-generate with injection prefix prepended to system message."""
    messages = json.loads(json.dumps(messages_original))  # deep copy
    messages[0]["content"] = INJECTION_PREFIX + "\n\n" + messages[0]["content"]

    prompt = tokenizer.apply_chat_template(
        messages, tools=[TOOL_DEF], tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=None, top_p=None, pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(out[0, input_len:], skip_special_tokens=True)
    return response


def generate_with_steering(model, tokenizer, messages, alpha, max_new_tokens=300):
    """Generate with activation steering: subtract α × direction at BEST_LAYER."""
    direction_tensor = torch.tensor(PROBE_DIRECTION, dtype=PRECISION).to(model.device)

    # Hook to modify residual stream at BEST_LAYER
    def steering_hook(module, input, output):
        # output is a tuple: (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        # Subtract α × direction from the last position only
        hs[:, -1, :] -= alpha * direction_tensor
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    # Identify the layer module
    layer_module = model.model.layers[BEST_LAYER - 1]  # -1 because hidden_states[0] is embedding

    prompt = tokenizer.apply_chat_template(
        messages, tools=[TOOL_DEF], tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    handle = layer_module.register_forward_hook(steering_hook)
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    response = tokenizer.decode(out[0, input_len:], skip_special_tokens=True)
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    print("Stage 5: Intervention")
    print("=" * 60)

    # Load model
    if torch.cuda.is_available():
        device = "cuda"
        print(f"  GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"  Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION,
            torch_dtype=PRECISION, device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION,
            torch_dtype=PRECISION, low_cpu_mem_usage=True,
        )
        if device == "mps":
            model = model.to("mps")
    model.eval()

    # Load trajectories
    with open(ROOT / "data" / "trajectories" / "all_trajectories.json") as f:
        all_trajs = json.load(f)
    traj_by_id = {t["trajectory_id"]: t for t in all_trajs}

    test_ids = INT_CFG["test_ids"]
    present_ids = INT_CFG["present_ids"]

    if args.smoke_test:
        test_ids = test_ids[:5]
        present_ids = present_ids[:5]

    print(f"  Test (empty): {len(test_ids)}")
    print(f"  Present: {len(present_ids)}")
    print(f"  Threshold τ: {THRESHOLD}")
    print(f"  Best layer: {BEST_LAYER}")
    print(f"  Alpha sweep: {ALPHA_SWEEP}")

    # ===================================================================
    # Phase 1: Baseline generation + probe scoring on test set
    # ===================================================================
    print(f"\n--- Phase 1: Baseline (no intervention) ---")
    baseline_results = []
    for i, tid in enumerate(test_ids):
        t = traj_by_id[tid]
        messages = t["messages"]
        response, activation = generate_baseline(model, tokenizer, messages)
        score = probe_score(activation)
        fires = score >= THRESHOLD
        fab = heuristic_label(response)

        baseline_results.append({
            "trajectory_id": tid,
            "formula": t["formula"],
            "probe_score": score,
            "probe_fires": fires,
            "fabricated": fab,
            "response": response,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test_ids)}")

    n_baseline_fab = sum(r["fabricated"] for r in baseline_results)
    n_fires = sum(r["probe_fires"] for r in baseline_results)
    baseline_fab_rate = n_baseline_fab / len(baseline_results)
    print(f"  Baseline fabrication rate: {n_baseline_fab}/{len(baseline_results)} "
          f"({baseline_fab_rate:.1%})")
    print(f"  Probe fires: {n_fires}/{len(baseline_results)}")

    # ===================================================================
    # Phase 2: Prompt injection intervention
    # ===================================================================
    print(f"\n--- Phase 2: Prompt injection ---")
    injection_results = []
    for i, r in enumerate(baseline_results):
        t = traj_by_id[r["trajectory_id"]]
        if r["probe_fires"]:
            response = generate_with_injection(model, tokenizer, t["messages"])
        else:
            response = r["response"]  # no intervention needed

        fab = heuristic_label(response)
        injection_results.append({
            "trajectory_id": r["trajectory_id"],
            "intervened": r["probe_fires"],
            "fabricated": fab,
            "response": response,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(baseline_results)}")

    n_inj_fab = sum(r["fabricated"] for r in injection_results)
    inj_fab_rate = n_inj_fab / len(injection_results)
    reduction = (baseline_fab_rate - inj_fab_rate) / baseline_fab_rate if baseline_fab_rate > 0 else 0
    print(f"  Injection fabrication rate: {n_inj_fab}/{len(injection_results)} "
          f"({inj_fab_rate:.1%})")
    print(f"  Relative reduction: {reduction:.1%}")

    # ===================================================================
    # Phase 3: Activation steering (α sweep)
    # ===================================================================
    print(f"\n--- Phase 3: Activation steering (α sweep) ---")
    steering_results = {}
    for alpha in ALPHA_SWEEP:
        print(f"\n  α = {alpha}")
        alpha_results = []
        for i, r in enumerate(baseline_results):
            t = traj_by_id[r["trajectory_id"]]
            if r["probe_fires"]:
                response = generate_with_steering(model, tokenizer, t["messages"], alpha)
            else:
                response = r["response"]

            fab = heuristic_label(response)
            alpha_results.append({
                "trajectory_id": r["trajectory_id"],
                "intervened": r["probe_fires"],
                "fabricated": fab,
                "response": response,
            })
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(baseline_results)}")

        n_steer_fab = sum(r["fabricated"] for r in alpha_results)
        steer_fab_rate = n_steer_fab / len(alpha_results)
        steer_reduction = (baseline_fab_rate - steer_fab_rate) / baseline_fab_rate if baseline_fab_rate > 0 else 0
        print(f"    Fabrication rate: {n_steer_fab}/{len(alpha_results)} "
              f"({steer_fab_rate:.1%}), reduction: {steer_reduction:.1%}")
        steering_results[alpha] = {
            "fab_rate": steer_fab_rate,
            "reduction": steer_reduction,
            "results": alpha_results,
        }

    # ===================================================================
    # Phase 4: Accuracy preservation on data-present
    # ===================================================================
    print(f"\n--- Phase 4: Accuracy preservation (data-present) ---")
    present_baseline = []
    present_injection = []

    for i, pid in enumerate(present_ids):
        t = traj_by_id[pid]
        messages = t["messages"]
        response, activation = generate_baseline(model, tokenizer, messages)
        score = probe_score(activation)
        fires = score >= THRESHOLD

        # Check if the response contains the correct value
        tool_data = json.loads(t["tool_result"])
        val = tool_data["results"][0][t["property"]]
        val_str2 = f"{val:.2f}"
        val_str3 = f"{val:.3f}"
        val_str4 = f"{val:.4f}"
        correct = val_str2 in response or val_str3 in response or val_str4 in response

        present_baseline.append({
            "trajectory_id": pid,
            "probe_fires": fires,
            "correct": correct,
            "response": response,
        })

        # Injection version (if probe fires)
        if fires:
            inj_resp = generate_with_injection(model, tokenizer, t["messages"])
        else:
            inj_resp = response
        inj_correct = val_str2 in inj_resp or val_str3 in inj_resp or val_str4 in inj_resp
        present_injection.append({
            "trajectory_id": pid,
            "intervened": fires,
            "correct": inj_correct,
        })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(present_ids)}")

    baseline_acc = sum(r["correct"] for r in present_baseline) / len(present_baseline)
    inj_acc = sum(r["correct"] for r in present_injection) / len(present_injection)
    fp_rate = sum(r["probe_fires"] for r in present_baseline) / len(present_baseline)

    print(f"  Baseline accuracy: {baseline_acc:.1%}")
    print(f"  Post-injection accuracy: {inj_acc:.1%}")
    print(f"  Degradation: {(baseline_acc - inj_acc)*100:.1f} points")
    print(f"  False-positive rate: {fp_rate:.1%}")

    # ===================================================================
    # Summary
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"INTERVENTION SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline fabrication rate:    {baseline_fab_rate:.1%}")
    print(f"Prompt-injection fab rate:    {inj_fab_rate:.1%} "
          f"(reduction: {reduction:.1%})")
    print(f"\nActivation steering:")
    for alpha in ALPHA_SWEEP:
        sr = steering_results[alpha]
        print(f"  α={alpha:5.1f}: fab={sr['fab_rate']:.1%}, "
              f"reduction={sr['reduction']:.1%}")
    print(f"\nAccuracy preservation:")
    print(f"  Baseline:       {baseline_acc:.1%}")
    print(f"  Post-injection: {inj_acc:.1%} (Δ={baseline_acc - inj_acc:.1%})")
    print(f"  FP rate:        {fp_rate:.1%}")

    # Save results
    output = {
        "baseline_fab_rate": baseline_fab_rate,
        "injection_fab_rate": inj_fab_rate,
        "injection_reduction": reduction,
        "steering": {
            str(a): {"fab_rate": sr["fab_rate"], "reduction": sr["reduction"]}
            for a, sr in steering_results.items()
        },
        "baseline_accuracy": baseline_acc,
        "injection_accuracy": inj_acc,
        "accuracy_degradation": baseline_acc - inj_acc,
        "fp_rate": fp_rate,
        "threshold_tau": THRESHOLD,
        "best_layer": BEST_LAYER,
    }
    with open(ROOT / "data" / "intervention_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to data/intervention_results.json")


if __name__ == "__main__":
    main()
