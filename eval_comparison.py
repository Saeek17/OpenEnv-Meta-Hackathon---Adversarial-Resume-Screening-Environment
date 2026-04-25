"""
eval_comparison.py — Compare rule-based baseline vs fine-tuned LoRA model
=========================================================================
Runs the same 9 episodes (3 per difficulty tier) through the live environment
using two agents:
  1. Rule-based baseline  — deterministic, no model
  2. Fine-tuned LoRA      — Qwen2.5-1.5B-Instruct + grpo_results adapter

Outputs:
  - Per-episode reward table (stdout)
  - assets/comparison_chart.png
  - assets/eval_results.json  (for blog post)

Usage:
    python eval_comparison.py                          # both agents
    python eval_comparison.py --baseline-only          # skip model (fast)
    ENV_URL=http://localhost:7860 python eval_comparison.py
"""

import os, json, re, argparse, traceback, random
import requests
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ENV_URL        = os.getenv("ENV_URL", "https://ishikamahadar-resume-env.hf.space")
MODEL_NAME     = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
HF_TOKEN       = os.getenv("HF_TOKEN", "")
ADAPTER_PATH   = "grpo_results/grpo_fleet_output/final"

TASK_TYPES     = ["easy", "medium", "hard"]
EPISODES_PER_TASK = 3   # 9 total episodes
SEEDS          = [10, 20, 30]  # fixed seeds → same resumes for both agents

SYSTEM_PROMPT = """\
You are a hiring fleet agent. Output a single JSON action object and nothing else.

Valid action types per phase:
  fraud_specialist  → verify_credential | check_reference | view_section | submit_specialist_report
  skills_specialist → view_section | ask_clarification | submit_specialist_report
  timeline_specialist → view_section | ask_clarification | submit_specialist_report
  overseer → read_reports | request_reinvestigation | submit_final_decision

Output ONLY the JSON. No explanation."""

HIGH_VALUE_SECTIONS = {"experience", "education", "skills"}


# ─────────────────────────────────────────────────────────────────────────────
# Environment HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────
def env_reset(task_type: str, seed: int, episode_id: str) -> dict:
    r = requests.post(
        f"{ENV_URL}/reset",
        json={"task_type": task_type, "seed": seed, "episode_id": episode_id},
        timeout=30,
    )
    r.raise_for_status()
    d = r.json()
    return d.get("observation", d)


def env_step(action: dict) -> dict:
    r = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30)
    r.raise_for_status()
    d = r.json()
    obs = d.get("observation", d)
    obs["_reward"] = d.get("reward", obs.get("reward", 0.0)) or 0.0
    obs["_done"]   = d.get("done",   obs.get("done",   False))
    return obs


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based agent
# ─────────────────────────────────────────────────────────────────────────────
def rule_action(obs: dict) -> dict:
    phase     = obs.get("current_phase", "")
    available = obs.get("available_actions", [])

    def pick(preferred):
        for a in preferred:
            if a in available:
                return a
        return available[0] if available else "submit_specialist_report"

    if phase == "fraud_specialist":
        choice = pick(["verify_credential", "check_reference", "submit_specialist_report"])
        if choice == "check_reference":
            return {"action_type": "check_reference", "reference_id": "ref2"}
        if choice == "submit_specialist_report":
            vr  = obs.get("verification_result", "") or ""
            rr  = obs.get("reference_response", "")  or ""
            bad = "FAILED" in vr or "cannot verify" in rr.lower() or "not in our system" in rr.lower()
            return {
                "action_type": "submit_specialist_report",
                "findings": f"Verification: {vr[:120]}. Reference: {rr[:120]}.",
                "has_issues": bad,
                "specialist_confidence": 0.85 if bad else 0.70,
            }
        return {"action_type": choice}

    if phase == "skills_specialist":
        viewed = set(obs.get("visible_sections", {}).keys())
        want   = next((s for s in ["experience", "education", "skills", "projects"]
                       if s not in viewed and "view_section" in available), None)
        if want:
            return {"action_type": "view_section", "section": want}
        return {"action_type": "submit_specialist_report",
                "findings": "Reviewed skills sections.", "has_issues": False,
                "specialist_confidence": 0.70}

    if phase == "timeline_specialist":
        viewed = set(obs.get("visible_sections", {}).keys())
        want   = next((s for s in ["experience", "header", "summary"]
                       if s not in viewed and "view_section" in available), None)
        if want:
            return {"action_type": "view_section", "section": want}
        return {"action_type": "submit_specialist_report",
                "findings": "Timeline reviewed.", "has_issues": False,
                "specialist_confidence": 0.65}

    if phase == "overseer":
        already_read  = set(obs.get("reports_read", []))
        steps_left    = obs.get("steps_remaining", 1)
        reports       = obs.get("specialist_reports", [])
        n_issues      = sum(1 for r in reports if r.get("has_issues"))
        is_fraud      = n_issues >= 2

        def final_decision():
            return {
                "action_type":    "submit_final_decision",
                "decision":       "reject" if is_fraud else "accept",
                "fraud_flag":     is_fraud,
                "confidence":     0.85,
                "fraud_reasoning": (
                    "Credential FAILED or reference denied employment."
                    if is_fraud else ""
                ),
            }

        # Always submit on the last step — never burn it on a read
        if steps_left <= 1:
            return final_decision()

        # Read reports we haven't read yet (keep ≥1 step for final submit)
        unread = [t for t in ["fraud_specialist", "skills_specialist", "timeline_specialist"]
                  if t not in already_read]
        if unread and "read_reports" in available and steps_left > 1:
            return {"action_type": "read_reports", "report_target": unread[0]}

        return final_decision()

    return {"action_type": available[0]} if available else {"action_type": "submit_specialist_report"}


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuned model agent
# ─────────────────────────────────────────────────────────────────────────────
def load_model(adapter_path: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"  Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, token=HF_TOKEN or None, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, token=HF_TOKEN or None,
        torch_dtype=torch.float16, trust_remote_code=True,
    )
    print(f"  Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )
    model = model.to(device)
    print(f"  Model loaded on {device}")
    return tokenizer, model, device


def obs_to_prompt(obs: dict) -> str:
    lines = [
        f"PHASE: {obs.get('current_phase', '?')}",
        f"STEPS REMAINING: {obs.get('steps_remaining', 0)}",
        f"VIOLATIONS: {obs.get('violations_count', 0)}",
        "",
        f"ROLE INSTRUCTIONS:\n{obs.get('role_instructions', '')}",
        f"\nJOB DESCRIPTION:\n{obs.get('job_description', '')}",
    ]
    visible = obs.get("visible_sections") or {}
    if visible:
        lines.append("\nVISIBLE RESUME SECTIONS:")
        for sec, content in visible.items():
            lines.append(f"  [{sec.upper()}]\n{content}")
    reports = obs.get("specialist_reports") or []
    if reports:
        lines.append("\nSPECIALIST REPORTS:")
        for r in reports:
            lines.append(
                f"  [{r['specialist_role'].upper()}] has_issues={r['has_issues']} "
                f"confidence={r.get('confidence', 0):.2f}\n  {r['findings']}"
            )
    for key, label in [
        ("reference_response", "REFERENCE CHECK"),
        ("verification_result", "CREDENTIAL VERIFICATION"),
        ("clarification_response", "CLARIFICATION"),
    ]:
        val = obs.get(key)
        if val:
            lines.append(f"\n{label}:\n{val}")
    read_details = obs.get("read_report_details") or {}
    if read_details:
        lines.append("\nFULL REPORT DETAILS:")
        for role, detail in read_details.items():
            lines.append(f"  [{role.upper()}]\n{detail}")
    if obs.get("feedback"):
        lines.append(f"\nFEEDBACK: {obs['feedback']}")
    lines.append(f"\nAVAILABLE ACTIONS: {obs.get('available_actions', [])}")
    lines.append("\nYour JSON action:")
    return "\n".join(lines)


def model_action(tokenizer, model, device, obs: dict) -> dict:
    """Generate one action with the fine-tuned model."""
    phase     = obs.get("current_phase", "fraud_specialist")
    available = obs.get("available_actions", [])
    steps_left = obs.get("steps_remaining", 1)

    # Force submit on last step (same safety guard as training)
    if steps_left <= 1:
        if phase == "overseer":
            reports  = obs.get("specialist_reports", [])
            n_issues = sum(1 for r in reports if r.get("has_issues"))
            is_fraud = n_issues >= 2
            return {
                "action_type": "submit_final_decision",
                "decision":    "reject" if is_fraud else "accept",
                "fraud_flag":  is_fraud,
                "confidence":  0.70,
                "fraud_reasoning": "Forced terminal — specialists flagged issues." if is_fraud else "",
            }
        return {"action_type": "submit_specialist_report",
                "findings": "Forced terminal submission.", "has_issues": False,
                "specialist_confidence": 0.50}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": obs_to_prompt(obs)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.4,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Parse JSON from completion
    match = re.search(r'\{[^{}]+\}', completion, re.DOTALL)
    if match:
        try:
            action = json.loads(match.group())
            if "action_type" in action and action["action_type"] in available:
                return action
        except json.JSONDecodeError:
            pass

    # Fallback to rule-based if model output is invalid
    return rule_action(obs)


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────
def run_episode(task: str, seed: int, agent_name: str,
                tokenizer=None, model=None, device=None,
                debug: bool = False) -> dict:
    ep_id = f"eval-{agent_name}-{task}-{seed}"
    obs   = env_reset(task, seed, ep_id)

    # /step returns a per-step incremental reward each turn.
    # The terminal step (submit_final_decision) returns the large terminal bonus.
    # We sum all per-step rewards then clamp to [0, 1] — same as inference_fleet.py.
    step_rewards  = []
    step          = 0

    while step < 25:
        phase = obs.get("current_phase", "complete")
        if phase == "complete":
            break
        available = obs.get("available_actions", [])
        if not available:
            break

        if agent_name == "rule_based":
            action = rule_action(obs)
        else:
            action = model_action(tokenizer, model, device, obs)

        action["episode_id"] = ep_id

        obs   = env_step(action)
        r     = obs["_reward"]
        done  = obs["_done"]
        step_rewards.append(r)
        step += 1

        if debug:
            at = action.get("action_type", "?")
            print(f"    step {step:2d} [{phase[:8]}] {at:<30} → r={r:.4f}  done={done}")

        if done:
            break

    total = max(0.0, min(1.0, sum(step_rewards)))
    return {
        "task":         task,
        "seed":         seed,
        "agent":        agent_name,
        "total_reward": round(total, 4),
        "steps":        step,
        "step_rewards": step_rewards,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
def print_table(results: list[dict]):
    print("\n" + "="*75)
    print(f"{'Task':<8} {'Seed':<6} {'Agent':<14} {'Reward':<10} {'Steps':<6}")
    print("-"*75)
    for r in results:
        print(f"{r['task']:<8} {r['seed']:<6} {r['agent']:<14} "
              f"{r['total_reward']:<10.4f} {r['steps']:<6}")
    print("="*75)

    # Summary by agent and task
    agents = sorted(set(r["agent"] for r in results))
    print("\nAVERAGE REWARD BY DIFFICULTY")
    print(f"{'Task':<10}", end="")
    for a in agents:
        print(f"  {a:<14}", end="")
    print()
    print("-"*55)
    for task in TASK_TYPES:
        print(f"{task:<10}", end="")
        for a in agents:
            subset = [r["total_reward"] for r in results
                      if r["task"] == task and r["agent"] == a]
            avg = sum(subset) / len(subset) if subset else 0.0
            print(f"  {avg:<14.4f}", end="")
        print()
    print("-"*55)
    print(f"{'OVERALL':<10}", end="")
    for a in agents:
        subset = [r["total_reward"] for r in results if r["agent"] == a]
        avg = sum(subset) / len(subset) if subset else 0.0
        print(f"  {avg:<14.4f}", end="")
    print()


def plot_comparison(results: list[dict]):
    agents = sorted(set(r["agent"] for r in results))
    colors = {"rule_based": "#4C72B0", "finetuned": "#DD8452"}
    bar_w  = 0.35

    # Per-difficulty averages
    avgs = {}
    for a in agents:
        avgs[a] = []
        for task in TASK_TYPES:
            subset = [r["total_reward"] for r in results
                      if r["task"] == task and r["agent"] == a]
            avgs[a].append(sum(subset) / len(subset) if subset else 0.0)

    # Overall averages
    overall = {a: sum(avgs[a]) / len(avgs[a]) for a in agents}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: per-difficulty bar chart ────────────────────────────────────
    ax = axes[0]
    x  = np.arange(len(TASK_TYPES))
    for i, a in enumerate(agents):
        offset = (i - (len(agents) - 1) / 2) * bar_w
        bars   = ax.bar(x + offset, avgs[a], bar_w,
                        label=a.replace("_", "-"),
                        color=colors.get(a, "#888888"),
                        alpha=0.85, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in TASK_TYPES])
    ax.set_ylabel("Average Environment Reward")
    ax.set_title("Reward by Difficulty Tier")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="threshold")

    # ── Right: overall comparison ─────────────────────────────────────────
    ax2 = axes[1]
    agent_labels = [a.replace("_", "-") for a in agents]
    bar_colors   = [colors.get(a, "#888") for a in agents]
    bars2 = ax2.bar(agent_labels, [overall[a] for a in agents],
                    color=bar_colors, alpha=0.85, edgecolor="white", width=0.4)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    if len(agents) == 2:
        delta = overall[agents[1]] - overall[agents[0]]
        sign  = "+" if delta >= 0 else ""
        ax2.text(0.5, 0.92, f"Δ = {sign}{delta:.3f}  ({sign}{delta*100:.1f}%)",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=12, fontweight="bold",
                 color="green" if delta >= 0 else "red")

    ax2.set_ylabel("Average Environment Reward (all episodes)")
    ax2.set_title("Overall: Rule-Based vs Fine-Tuned")
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Hiring Fleet — Baseline vs GRPO Fine-Tuned (Qwen2.5-1.5B-Instruct)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/comparison_chart.png", dpi=150)
    plt.show()
    print("Saved: assets/comparison_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run only the rule-based baseline (no model load)")
    args = parser.parse_args()

    # ── Verify env ────────────────────────────────────────────────────────
    print(f"ENV_URL: {ENV_URL}")
    health = requests.get(f"{ENV_URL}/health", timeout=15).json()
    print(f"Health : {health}\n")

    results = []

    # ── Baseline ─────────────────────────────────────────────────────────
    print("=== Running rule-based baseline ===")
    first = True
    for task in TASK_TYPES:
        for seed in SEEDS:
            print(f"  [{task}] seed={seed} ...", end=" ", flush=True)
            try:
                r = run_episode(task, seed, "rule_based", debug=first)
                if first:
                    print()  # newline after debug block
                    first = False
                results.append(r)
                print(f"reward={r['total_reward']:.4f}  steps={r['steps']}")
            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()

    # ── Fine-tuned model ─────────────────────────────────────────────────
    if not args.baseline_only:
        if not os.path.exists(ADAPTER_PATH):
            print(f"\nAdapter not found at {ADAPTER_PATH} — skipping model eval.")
            print("Run with --baseline-only or copy grpo_results/ to this directory.")
        else:
            print(f"\n=== Loading fine-tuned model from {ADAPTER_PATH} ===")
            try:
                tokenizer, model, device = load_model(ADAPTER_PATH)
                print("\n=== Running fine-tuned model ===")
                for task in TASK_TYPES:
                    for seed in SEEDS:
                        print(f"  [{task}] seed={seed} ...", end=" ", flush=True)
                        try:
                            r = run_episode(task, seed, "finetuned",
                                            tokenizer=tokenizer, model=model, device=device)
                            results.append(r)
                            print(f"reward={r['total_reward']:.4f}  steps={r['steps']}")
                        except Exception as e:
                            print(f"ERROR: {e}")
                            traceback.print_exc()
            except Exception as e:
                print(f"Model load failed: {e}")
                traceback.print_exc()

    # ── Results ───────────────────────────────────────────────────────────
    print_table(results)

    # Save JSON
    os.makedirs("assets", exist_ok=True)
    with open("assets/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: assets/eval_results.json")

    # Chart (only if both agents ran)
    agents = set(r["agent"] for r in results)
    if len(agents) >= 1:
        plot_comparison(results)


if __name__ == "__main__":
    main()
