"""
train_grpo.py — GRPO training for the Hiring Fleet environment.

Stage 1: standard GRPO on the 12-component reward.
         Teaches role discipline, fraud detection, and report quality.

Usage:
    export ENV_URL="https://ishikamahadar-resume-env.hf.space"   # or local
    python train_grpo.py

Recommended platform: Kaggle (free T4, 16 GB VRAM) or RunPod RTX 3090.
Model: Qwen/Qwen2.5-1.5B-Instruct
"""

import os, json, re, random, requests, textwrap
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
ENV_URL    = os.getenv("ENV_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
HF_TOKEN   = os.getenv("HF_TOKEN", "")          # HF hub token for gated models

TASK_TYPES  = ["easy", "medium", "hard"]
N_EPISODES  = 36          # one full pass over the dataset per epoch
GROUP_SIZE  = 8           # GRPO group size (completions per prompt)
MAX_STEPS   = 15          # hard cap per episode (hard tier max)
OUTPUT_DIR  = "grpo_fleet_output"

# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders — one per specialist role
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a hiring fleet agent. You will receive an observation describing your
    current role, the job description, visible resume sections, and available actions.

    Respond with a single JSON action object. Do NOT add any explanation outside the JSON.
    The JSON must contain "action_type" and any required fields for that action.

    Examples:
      {"action_type": "verify_credential"}
      {"action_type": "check_reference", "reference_id": "ref2"}
      {"action_type": "view_section", "section": "experience"}
      {"action_type": "ask_clarification", "question": "Can you clarify the Kafka usage?"}
      {"action_type": "submit_specialist_report", "findings": "...", "has_issues": true, "specialist_confidence": 0.85}
      {"action_type": "read_reports", "report_target": "fraud_specialist"}
      {"action_type": "request_reinvestigation", "reinvestigation_target": "fraud_specialist", "reinvestigation_reason": "..."}
      {"action_type": "submit_final_decision", "decision": "reject", "fraud_flag": true, "confidence": 0.9, "fraud_reasoning": "..."}
""")


def obs_to_prompt(obs: dict) -> str:
    """Convert a FleetObservation dict to a human-readable prompt string."""
    lines = [
        f"PHASE: {obs.get('current_phase', 'unknown')}",
        f"STEPS REMAINING: {obs.get('steps_remaining', 0)} (total: {obs.get('total_steps_remaining', 0)})",
        f"VIOLATIONS SO FAR: {obs.get('violations_count', 0)}",
        "",
        f"ROLE INSTRUCTIONS:\n{obs.get('role_instructions', '')}",
        "",
        f"JOB DESCRIPTION:\n{obs.get('job_description', '')}",
    ]

    visible = obs.get("visible_sections", {})
    if visible:
        lines.append("\nVISIBLE RESUME SECTIONS:")
        for sec, content in visible.items():
            lines.append(f"  [{sec.upper()}]\n{content}")

    reports = obs.get("specialist_reports", [])
    if reports:
        lines.append("\nSPECIALIST REPORTS SO FAR:")
        for r in reports:
            lines.append(
                f"  [{r['specialist_role'].upper()}] has_issues={r['has_issues']} "
                f"confidence={r['confidence']}\n  {r['findings']}"
            )

    for key, label in [
        ("reference_response",   "REFERENCE CHECK RESULT"),
        ("verification_result",  "CREDENTIAL VERIFICATION"),
        ("clarification_response", "CLARIFICATION RESPONSE"),
    ]:
        val = obs.get(key)
        if val:
            lines.append(f"\n{label}:\n{val}")

    read_details = obs.get("read_report_details", {})
    if read_details:
        lines.append("\nFULL REPORT DETAILS (read via read_reports):")
        for role, detail in read_details.items():
            lines.append(f"  [{role.upper()}]\n{detail}")

    feedback = obs.get("feedback")
    if feedback:
        lines.append(f"\nFEEDBACK: {feedback}")

    lines.append(f"\nAVAILABLE ACTIONS: {obs.get('available_actions', [])}")
    lines.append("\nRespond with a single JSON action object:")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Action parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_action(text: str, available: list) -> dict:
    """Extract JSON action from model output. Fall back gracefully."""
    # Try to find JSON in the output
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if match:
        try:
            action = json.loads(match.group())
            if "action_type" in action and action["action_type"] in available:
                return action
        except json.JSONDecodeError:
            pass

    # Fallback: pick the safest available action
    priority = [
        "submit_final_decision", "submit_specialist_report",
        "verify_credential", "check_reference",
        "read_reports", "view_section", "ask_clarification",
    ]
    for at in priority:
        if at in available:
            return _fallback_action(at)
    return {"action_type": available[0]} if available else {"action_type": "submit_specialist_report"}


def _fallback_action(action_type: str) -> dict:
    defaults = {
        "submit_specialist_report": {
            "action_type": "submit_specialist_report",
            "findings": "Insufficient information gathered.",
            "has_issues": False,
            "specialist_confidence": 0.3,
        },
        "submit_final_decision": {
            "action_type": "submit_final_decision",
            "decision": "reject",
            "fraud_flag": False,
            "confidence": 0.3,
            "fraud_reasoning": "",
        },
        "check_reference":  {"action_type": "check_reference",  "reference_id": "ref1"},
        "view_section":     {"action_type": "view_section",      "section": "experience"},
        "ask_clarification": {"action_type": "ask_clarification", "question": "Can you clarify your experience?"},
        "read_reports":     {"action_type": "read_reports",      "report_target": "fraud_specialist"},
        "verify_credential": {"action_type": "verify_credential"},
    }
    return defaults.get(action_type, {"action_type": action_type})


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(task_type: str, seed: int, episode_id: str, generate_fn) -> tuple[float, list]:
    """
    Run one full fleet episode, calling generate_fn(prompt) → text for each step.
    Returns (total_reward, list_of_(prompt, completion) pairs).
    """
    # Reset
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task_type": task_type, "seed": seed, "episode_id": episode_id},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    obs  = data["observation"]

    total_reward = 0.0
    pairs = []   # (prompt_str, completion_str) for GRPO

    for _step in range(MAX_STEPS):
        if obs.get("current_phase") == "complete" or data.get("done"):
            total_reward += data.get("reward", 0.0)
            break

        available = obs.get("available_actions", [])
        if not available:
            break

        prompt     = obs_to_prompt(obs)
        completion = generate_fn(prompt)
        action     = parse_action(completion, available)
        action["episode_id"] = episode_id

        resp = requests.post(
            f"{ENV_URL}/step",
            json={"action": action},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        total_reward += data.get("reward", 0.0)
        obs           = data["observation"]
        pairs.append((prompt, completion))

        if data.get("done"):
            break

    return total_reward, pairs


# ─────────────────────────────────────────────────────────────────────────────
# GRPO reward function (called by TRL trainer)
# ─────────────────────────────────────────────────────────────────────────────

def make_reward_fn(prompts_to_rewards: dict):
    """Return a reward function that looks up pre-computed episode rewards."""
    def reward_fn(completions, prompts=None, **kwargs):
        rewards = []
        for prompt, completion in zip(prompts or [], completions):
            key = (prompt, completion)
            rewards.append(prompts_to_rewards.get(key, 0.0))
        return rewards
    return reward_fn


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder — generates GRPO training samples
# ─────────────────────────────────────────────────────────────────────────────

def build_grpo_dataset(model, tokenizer, n_episodes: int = N_EPISODES):
    """
    Run n_episodes against the live environment using the current model,
    collect (prompt, completion, reward) triples, and return a HF Dataset.
    """
    from datasets import Dataset

    device = next(model.parameters()).device

    def generate(prompt: str) -> str:
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)

    records = []
    prompt_reward_map = {}

    for i in range(n_episodes):
        task   = TASK_TYPES[i % len(TASK_TYPES)]
        seed   = random.randint(0, 99)
        ep_id  = f"grpo-ep-{i}-seed-{seed}"

        try:
            reward, pairs = run_episode(task, seed, ep_id, generate)
            # Distribute reward equally across all steps in the episode
            per_step_reward = reward / max(len(pairs), 1)
            for prompt, completion in pairs:
                records.append({
                    "prompt":     [{"role": "system", "content": SYSTEM_PROMPT},
                                   {"role": "user",   "content": prompt}],
                    "completion": completion,
                    "reward":     per_step_reward,
                })
                prompt_reward_map[(prompt, completion)] = per_step_reward
            print(f"  episode {i:3d} | {task:6s} | reward={reward:.4f} | steps={len(pairs)}")
        except Exception as e:
            print(f"  episode {i:3d} | {task:6s} | ERROR: {e}")

    return Dataset.from_list(records), prompt_reward_map


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN or None,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN or None,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.train()

    # ── Verify environment is reachable ──────────────────────────────────────
    try:
        health = requests.get(f"{ENV_URL}/health", timeout=10).json()
        print(f"Environment health: {health}")
    except Exception as e:
        raise RuntimeError(f"Cannot reach environment at {ENV_URL}: {e}")

    # ── Build initial dataset ─────────────────────────────────────────────────
    print(f"\nCollecting {N_EPISODES} rollout episodes …")
    dataset, reward_map = build_grpo_dataset(model, tokenizer)
    print(f"Dataset size: {len(dataset)} prompt-completion pairs")

    # ── GRPO config ──────────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        max_grad_norm=0.1,
        num_generations=GROUP_SIZE,      # completions per prompt for GRPO
        max_completion_length=256,
        temperature=0.7,
        beta=0.04,                       # KL penalty coefficient
        logging_steps=10,
        save_steps=100,
        report_to="none",               # set to "wandb" if you have wandb
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
    )

    # ── Reward function ───────────────────────────────────────────────────────
    reward_fn = make_reward_fn(reward_map)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    print("\nStarting GRPO training …")
    trainer.train()

    print(f"\nSaving model to {OUTPUT_DIR}/final")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print("Done.")


if __name__ == "__main__":
    main()
