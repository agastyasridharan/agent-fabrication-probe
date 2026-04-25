"""
Stage 2: Trajectory Collection + Activation Extraction

Runs Llama-3.1-8B-Instruct with a simulated Materials Project tool to collect
agent trajectories AND extract residual-stream activations at the last prompt
token (before generation) across all layers.

Empty side:   200 perturbations × 3 system prompts = 600 trajectories.
Data-present: 200 real materials  × 3 system prompts = 600 trajectories.
Total: 1200 trajectories.

Each trajectory uses a deterministically rotated (property, template) pair.

Activations are saved as per-trajectory .npy files with shape
[n_layers+1, hidden_dim] (e.g. [33, 4096] for Llama-3.1-8B).

Usage:
  python src/agent_loop.py               # full 1200-trajectory run
  python src/agent_loop.py --smoke-test  # 5-trajectory diagnostic
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Templates and system prompts (loaded from frozen files)
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[str, str] = {}
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

TEMPLATES: dict[str, list[str]] = {}
_current_prop = None
with open(ROOT / "prompts" / "templates.md") as f:
    for line in f:
        line = line.rstrip()
        if line.startswith("## "):
            _current_prop = line[3:].strip()
            TEMPLATES[_current_prop] = []
        elif _current_prop and (
            line.startswith('"')
            or (line and line[0].isdigit() and '"' in line)
        ):
            start = line.index('"') + 1
            end = line.rindex('"')
            TEMPLATES[_current_prop].append(line[start:end])

assert all(len(v) == 5 for v in TEMPLATES.values()), (
    f"Expected 5 templates per property, got: "
    f"{ {k: len(v) for k, v in TEMPLATES.items()} }"
)

PROPERTIES = list(TEMPLATES.keys())

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "lookup_material_property",
        "description": (
            "Look up a property of a material in the Materials Project "
            "database. Returns a list of matching entries, or an empty list "
            "if the material is not found."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "formula": {
                    "type": "string",
                    "description": "Chemical formula of the material",
                },
                "property": {
                    "type": "string",
                    "enum": ["band_gap", "formation_energy_per_atom", "density"],
                    "description": "The property to look up",
                },
            },
            "required": ["formula", "property"],
        },
    },
}


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------

def build_messages(
    system_prompt: str,
    user_query: str,
    formula: str,
    prop: str,
    tool_result_content: str,
) -> list[dict]:
    """Build the full message list including the simulated tool exchange."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_material_property",
                        "arguments": json.dumps(
                            {"formula": formula, "property": prop}
                        ),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": tool_result_content},
    ]


def make_empty_tool_result() -> str:
    return json.dumps({"results": []})


def make_data_present_tool_result(material: dict, prop: str) -> str:
    return json.dumps({
        "results": [
            {
                "material_id": material["material_id"],
                "formula": material["formula"],
                prop: material[prop],
            }
        ]
    })


# ---------------------------------------------------------------------------
# Core: activation extraction + generation
# ---------------------------------------------------------------------------

def extract_and_generate(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 300,
) -> tuple[str, np.ndarray, str, str]:
    """Run a forward pass to extract activations, then generate a response.

    Returns:
        response:           decoded model response text
        activations:        np array of shape [n_layers+1, hidden_dim]
        last_token_decoded: the decoded string of the token at extraction position
        prompt_text:        the full templated prompt string
    """
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tools=[TOOL_DEF],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    # --- Activation extraction: forward pass on prompt only ---
    with torch.no_grad():
        fwd_out = model(**inputs, output_hidden_states=True)

    # hidden_states: tuple of (n_layers+1) tensors, each [1, seq_len, hidden_dim]
    # Extract the last prompt token from every layer
    activations = torch.stack(
        [layer[0, -1, :].float().cpu() for layer in fwd_out.hidden_states]
    ).numpy()  # [n_layers+1, hidden_dim]

    last_token_id = inputs["input_ids"][0, -1].item()
    last_token_decoded = tokenizer.decode([last_token_id])

    # --- Generation ---
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response, activations, last_token_decoded, prompt_text


# ---------------------------------------------------------------------------
# Trajectory assignment
# ---------------------------------------------------------------------------

def assign_property_template(idx: int) -> tuple[str, int, str]:
    """Deterministically assign a (property, template_index, template_str)
    cycling through all 15 combinations."""
    combos = [(p, t_idx) for p in PROPERTIES for t_idx in range(5)]
    p, t_idx = combos[idx % len(combos)]
    return p, t_idx, TEMPLATES[p][t_idx]


# ---------------------------------------------------------------------------
# Collect one trajectory
# ---------------------------------------------------------------------------

def collect_trajectory(
    model, tokenizer, trajectory_id: str, side: str,
    formula: str, prop: str, tmpl_idx: int, user_query: str,
    system_prompt_variant: str, system_prompt: str,
    tool_result: str, act_dir: Path,
    extra_fields: dict | None = None,
) -> dict:
    """Run extraction + generation for one trajectory and save activation."""
    messages = build_messages(system_prompt, user_query, formula, prop, tool_result)
    response, activations, last_tok, prompt_text = extract_and_generate(
        model, tokenizer, messages
    )

    # Save activation
    act_path = act_dir / f"{trajectory_id}.npy"
    np.save(act_path, activations)

    record = {
        "trajectory_id": trajectory_id,
        "side": side,
        "formula": formula,
        "property": prop,
        "template_index": tmpl_idx,
        "system_prompt_variant": system_prompt_variant,
        "system_prompt": system_prompt,
        "user_query": user_query,
        "tool_result": tool_result,
        "assistant_response": response,
        "activation_path": str(act_path.relative_to(ROOT)),
        "activation_shape": list(activations.shape),
        "extraction_token": last_tok,
        "messages": messages,
    }
    if extra_fields:
        record.update(extra_fields)
    return record


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(model, tokenizer, perturbations, real_materials, act_dir):
    """Run 5 trajectories (mix of sides and prompts) with full diagnostics."""
    print("\n" + "=" * 60)
    print("SMOKE TEST: 5 trajectories")
    print("=" * 60)

    sp_keys = list(SYSTEM_PROMPTS.keys())

    # Pick 5 diverse trajectories:
    #   0: empty / neutral
    #   1: empty / pressure
    #   2: empty / honesty
    #   3: data_present / neutral
    #   4: data_present / pressure
    test_specs = [
        ("empty",        0, sp_keys[0]),  # neutral
        ("empty",        1, sp_keys[1]),  # pressure
        ("empty",        2, sp_keys[2]),  # honesty
        ("data_present", 0, sp_keys[0]),  # neutral
        ("data_present", 1, sp_keys[1]),  # pressure
    ]

    for i, (side, data_idx, sp_key) in enumerate(test_specs):
        prop, tmpl_idx, tmpl_str = assign_property_template(i)

        if side == "empty":
            pert = perturbations[data_idx]
            formula = pert["perturbed_formula"]
            tool_result = make_empty_tool_result()
            extra = {
                "source_formula": pert["source_formula"],
                "perturbation_type": pert["perturbation_type"],
            }
        else:
            mat = real_materials[data_idx]
            formula = mat["formula"]
            tool_result = make_data_present_tool_result(mat, prop)
            extra = {"material_id": mat["material_id"]}

        user_query = tmpl_str.format(formula=formula)
        system_prompt = SYSTEM_PROMPTS[sp_key]

        tid = f"smoke_{i}"
        messages = build_messages(system_prompt, user_query, formula, prop, tool_result)
        response, activations, last_tok, prompt_text = extract_and_generate(
            model, tokenizer, messages
        )
        act_path = act_dir / f"{tid}.npy"
        np.save(act_path, activations)

        print(f"\n{'─'*60}")
        print(f"TRAJECTORY {i}: {side} / {sp_key} / {formula} / {prop}")
        print(f"{'─'*60}")
        print(f"\n[FULL PROMPT]\n{prompt_text}")
        print(f"\n[EXTRACTION TOKEN] '{last_tok}'")
        print(f"[ACTIVATION SHAPE] {activations.shape}")
        print(f"\n[TOOL RESULT] {tool_result}")
        print(f"\n[RESPONSE]\n{response}")
        print()

    # Verify all 5 extraction tokens are identical
    tokens = set()
    for i in range(5):
        act = np.load(act_dir / f"smoke_{i}.npy")
        # Re-derive token — just collect from the print above
        # (we printed them; here just verify shapes)
        assert act.shape[0] == CFG["probe"]["n_layers"], (
            f"Expected {CFG['probe']['n_layers']} layers, got {act.shape[0]}"
        )

    print("=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

def run_full(model, tokenizer, perturbations, real_materials, act_dir, out_dir):
    """Collect all 1200 trajectories."""
    sp_keys = list(SYSTEM_PROMPTS.keys())
    all_trajectories = []
    t0 = time.time()

    # --- Empty side: 200 × 3 = 600 ---
    n_empty_target = len(perturbations) * len(sp_keys)
    print(f"\n--- Empty-side trajectories (target: {n_empty_target}) ---")

    for pert_idx, pert in enumerate(perturbations):
        formula = pert["perturbed_formula"]
        for sp_idx, sp_key in enumerate(sp_keys):
            traj_idx = pert_idx * len(sp_keys) + sp_idx
            prop, tmpl_idx, tmpl_str = assign_property_template(traj_idx)
            user_query = tmpl_str.format(formula=formula)

            rec = collect_trajectory(
                model, tokenizer,
                trajectory_id=f"empty_{traj_idx:04d}",
                side="empty",
                formula=formula,
                prop=prop,
                tmpl_idx=tmpl_idx,
                user_query=user_query,
                system_prompt_variant=sp_key,
                system_prompt=SYSTEM_PROMPTS[sp_key],
                tool_result=make_empty_tool_result(),
                act_dir=act_dir,
                extra_fields={
                    "source_formula": pert["source_formula"],
                    "source_material_id": pert.get("source_material_id", ""),
                    "perturbation_type": pert["perturbation_type"],
                },
            )
            all_trajectories.append(rec)

            if (traj_idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (traj_idx + 1) / elapsed
                eta = (n_empty_target - traj_idx - 1) / rate
                print(f"  {traj_idx+1}/{n_empty_target}  "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    print(f"  Collected {n_empty_target} empty-side trajectories.")

    # --- Data-present side: 200 × 3 = 600 ---
    n_present_target = len(real_materials) * len(sp_keys)
    print(f"\n--- Data-present trajectories (target: {n_present_target}) ---")
    t1 = time.time()

    for mat_idx, mat in enumerate(real_materials):
        formula = mat["formula"]
        for sp_idx, sp_key in enumerate(sp_keys):
            traj_idx = mat_idx * len(sp_keys) + sp_idx
            prop, tmpl_idx, tmpl_str = assign_property_template(traj_idx)
            user_query = tmpl_str.format(formula=formula)

            rec = collect_trajectory(
                model, tokenizer,
                trajectory_id=f"present_{traj_idx:04d}",
                side="data_present",
                formula=formula,
                prop=prop,
                tmpl_idx=tmpl_idx,
                user_query=user_query,
                system_prompt_variant=sp_key,
                system_prompt=SYSTEM_PROMPTS[sp_key],
                tool_result=make_data_present_tool_result(mat, prop),
                act_dir=act_dir,
                extra_fields={"material_id": mat["material_id"]},
            )
            all_trajectories.append(rec)

            if (traj_idx + 1) % 50 == 0:
                elapsed = time.time() - t1
                rate = (traj_idx + 1) / elapsed
                eta = (n_present_target - traj_idx - 1) / rate
                print(f"  {traj_idx+1}/{n_present_target}  "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    print(f"  Collected {n_present_target} data-present trajectories.")

    # --- Save ---
    out_path = out_dir / "all_trajectories.json"
    with open(out_path, "w") as f:
        json.dump(all_trajectories, f, indent=2)

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Total: {len(all_trajectories)} trajectories in {total_time:.0f}s")
    print(f"Saved to {out_path}")

    # Quick stats
    for sp_key in sp_keys:
        subset = [t for t in all_trajectories
                  if t["side"] == "empty" and t["system_prompt_variant"] == sp_key]
        lens = [len(t["assistant_response"].split()) for t in subset]
        print(f"  empty/{sp_key}: n={len(subset)}, mean_words={np.mean(lens):.0f}")

    for sp_key in sp_keys:
        subset = [t for t in all_trajectories
                  if t["side"] == "data_present" and t["system_prompt_variant"] == sp_key]
        lens = [len(t["assistant_response"].split()) for t in subset]
        print(f"  present/{sp_key}: n={len(subset)}, mean_words={np.mean(lens):.0f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 5-trajectory diagnostic instead of full run")
    args = parser.parse_args()

    print("Stage 2: Trajectory Collection + Activation Extraction")
    print("=" * 60)

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  Precision: {CFG['model']['precision']}")
        if not torch.cuda.is_bf16_supported():
            print("  WARNING: bf16 not supported on this GPU")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("  Device: MPS (Apple Silicon)")
        print("  NOTE: spec requires A100+bf16; MPS acceptable for smoke test")
    else:
        device = "cpu"
        print("  WARNING: No GPU detected, running on CPU (very slow)")

    # Load model
    print(f"\nLoading {MODEL_NAME} (revision {MODEL_REVISION[:12]}...)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            torch_dtype=PRECISION,
            device_map="auto",
        )
    else:
        # MPS / CPU: load to CPU first, then move (avoids large single alloc)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            torch_dtype=PRECISION,
            low_cpu_mem_usage=True,
        )
        if device == "mps":
            model = model.to("mps")
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_layers = model.config.num_hidden_layers  # 32 transformer layers
    hidden_dim = model.config.hidden_size       # 4096
    print(f"  Loaded: {n_params/1e9:.1f}B params, {n_layers} layers, "
          f"hidden_dim={hidden_dim}")
    print(f"  output_hidden_states will produce {n_layers + 1} tensors "
          f"(embedding + {n_layers} layers)")

    # Load data
    with open(ROOT / "data" / "perturbations_verified.json") as f:
        perturbations = json.load(f)
    with open(ROOT / "data" / "materials.json") as f:
        data = json.load(f)
    real_materials = data["real_materials"]
    print(f"  {len(perturbations)} perturbations, {len(real_materials)} real materials")

    # Output dirs
    out_dir = ROOT / "data" / "trajectories"
    act_dir = out_dir / "activations"
    out_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        run_smoke_test(model, tokenizer, perturbations, real_materials, act_dir)
    else:
        run_full(model, tokenizer, perturbations, real_materials, act_dir, out_dir)


if __name__ == "__main__":
    main()
