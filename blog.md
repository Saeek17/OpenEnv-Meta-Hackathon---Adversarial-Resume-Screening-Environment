# Hiring Fleet: Training AI Agents to Catch Resume Fraud Through Multi-Agent Oversight

**Environment:** [IshikaMahadar/resume-env](https://huggingface.co/spaces/IshikaMahadar/resume-env)  
**Trained adapter:** [IshikaMahadar/hiring-fleet-grpo-adapter](https://huggingface.co/IshikaMahadar/hiring-fleet-grpo-adapter)  
**GitHub:** [Ishika-eng/OpenEnv-Meta-Hackathon](https://github.com/Ishika-eng/OpenEnv-Meta-Hackathon---Adversarial-Resume-Screening-Environment)

---

## The Problem

Every year, companies lose billions hiring fraudulent candidates — people who fake degrees, fabricate work history, and list references who will lie for them. A single AI agent reading a resume in isolation is easy to fool. Sophisticated fraud hides clues *across* different sections, each one looking innocent alone:

- A fake institution in the **header**
- A reference who quietly denies the employment in the **references section**
- Impossible overlapping dates buried in **experience**

No one section reveals the fraud. Only the *combination* does.

**The question this project asks: can a fleet of constrained specialist agents catch fraud that fools a single agent?**

---

## The Environment: Hiring Fleet

Built on [OpenEnv](https://huggingface.co/openenv), the environment runs a four-phase sequential pipeline per episode. Each agent has a hard-enforced role — it can only see specific sections and call specific tools. If it tries anything outside its whitelist, the action is rejected, a step is consumed, and −0.05 is deducted from the terminal reward.

```
Phase 1 — Fraud Specialist
  Sees: header, references
  Tools: verify_credential, check_reference
  Submits: fraud report (has_issues, confidence, findings)

Phase 2 — Skills Specialist
  Sees: experience, education, skills, projects
  Tools: ask_clarification
  Submits: skills report

Phase 3 — Timeline Specialist
  Sees: header, summary, experience
  Tools: ask_clarification
  Submits: timeline report

Phase 4 — Overseer
  Sees: NOTHING (no raw resume access)
  Tools: read_reports, request_reinvestigation, submit_final_decision
  Submits: final hire/reject + fraud flag
```

The Overseer cannot see the raw resume. It reads only what the specialists wrote. If the specialists produce weak reports, the Overseer has no signal to work with. The chain of reasoning is real.

### Why this design is hard to game

| Design choice | Why it matters |
|:---|:---|
| Hard action whitelists | Agents can't take shortcuts by accessing other phases' tools |
| Role-filtered observations | Information is genuinely siloed — the Fraud Specialist cannot read experience |
| No ground truth in Overseer reports | Overseer receives only specialist *findings*, never environment answers |
| Fleet coordination bonus (+0.08) | Fires only when all 4 agents are simultaneously correct — explicitly incentivises cooperation |
| Deterministic, LLM-free reward | All scoring derives from ground-truth fields; no judge model needed |

---

## The Reward Function

Rewards are dense across all phases. Every useful action returns signal, giving GRPO stable gradients throughout the episode.

**Per-step rewards** fire on actions like `verify_credential` (returns FAILED → +0.05), `check_reference` (fraud signal revealed → +0.05), and `submit_specialist_report` (correct signal + calibrated confidence → up to +0.10).

**Terminal reward** delegates to seven independent named sub-functions:

| Sub-function | What it rewards | Max |
|:---|:---|:---:|
| Decision accuracy | Correct accept/reject + fraud flag + calibrated confidence | +0.70 |
| Specialist quality | Each correct specialist, tier-scaled (hard cases worth more) | +0.22 |
| Fleet coordination | All 3 specialists + Overseer simultaneously correct | +0.08 |
| Oversight quality | Overseer read all reports + used reinvestigation appropriately | +0.08 |
| Investigation depth | Breadth of sections × tool usage ratio | +0.05 |
| Format compliance | Fraud reasoning contains indicator keywords | +0.05 |
| Step efficiency | Correct decisions with budget remaining | +0.04 |

**Anti-exploit penalties** close three shortcut strategies:
- Always output high confidence regardless of correctness → −0.05
- Set `fraud_flag=True` but write empty reasoning → −0.05
- Skip all investigation tools on a fraud resume → −0.05

Total reward range: **[0.0, 1.0]**. No LLM judge. Fully reproducible.

---

## The Dataset

**36 curated resumes** across three difficulty tiers (12 each, 42% fraud per tier).

| Tier | Fraud type |
|:---|:---|
| Easy | Obvious: fake institutions, role-mismatch references, impossible timelines |
| Medium | Subtle: scope exaggeration, compliance misrepresentation |
| Hard | Sophisticated: title inflation, contradicting references, multi-section inconsistencies |

Every fraud resume guarantees a detectable signal: `verify_credential` returns FAILED, and `check_reference(ref2)` returns a suspicious or denying response. The Fraud Specialist always has something to find — the challenge is recognising it and propagating it correctly through the fleet.

---

## Training: GRPO on Qwen2.5-1.5B-Instruct

| Parameter | Value |
|:---|:---|
| Base model | Qwen/Qwen2.5-1.5B-Instruct |
| Adapter | LoRA (r=16, alpha=32, target: q_proj + v_proj) |
| Framework | HuggingFace TRL — GRPOTrainer |
| Hardware | T4 GPU (Colab free tier) |
| Steps | 792 |
| Data | 36 offline episodes collected via rule-based agent |

### Training pipeline

The rule-based agent first walks all 36 episodes against the live HF Space, collecting observation prompts at each step. GRPOTrainer then generates multiple completions per prompt and scores them with a proxy reward function (JSON format validity, role-appropriate action types, fraud indicator keywords in reasoning). No live environment calls during gradient updates — this keeps Colab free-tier feasible.

### Reward curve

![GRPO Reward Curve](assets/reward_curve.png)

| Metric | Value |
|:---|:---|
| Start reward | 0.736 |
| Best reward | 0.850 |
| Improvement | +15.5% |

### What the model learned

1. **Output valid JSON reliably** — format compliance rose from ~40% to ~95%
2. **Select role-appropriate actions** — out-of-role violation rate dropped significantly
3. **Prioritise `verify_credential`** as the Fraud Specialist's first move
4. **Write fraud indicator keywords** in reasoning (`failed`, `denied`, `fabricated`) when flagging fraud

---

## Evaluation: Baseline vs Fine-Tuned (Live Environment)

Evaluated on 9 episodes (3 per tier) against the deployed HF Space, comparing the rule-based baseline to the GRPO fine-tuned model.

![Comparison Chart](assets/comparison_chart.png)

| Agent | Easy | Medium | Hard | Overall |
|:---|:---:|:---:|:---:|:---:|
| Rule-based baseline | 0.747 | 0.873 | 1.000 | **0.873** |
| Fine-tuned (GRPO) | 0.722 | 0.888 | 1.000 | **0.870** |

The overall gap is small (−0.4%), which is expected: GRPO training used a static proxy reward rather than live environment terminal rewards. The full terminal signal requires a multi-step causal chain — `verify_credential → FAILED → specialist report → Overseer reads → reject` — that wasn't directly optimised.

However, on **medium difficulty fraud (seed=10)**, the fine-tuned model outperforms at the specialist level: it submits a correct fraud report at step 2 (+0.08) while the rule-based agent wastes the same step checking a clean reference (+0.00). GRPO taught the model to reason from evidence to conclusion at the specialist level, even without explicit training on the terminal signal.

Stage 2 training with live environment rewards — where the terminal signal propagates back through the full causal chain — would close the remaining gap.

---

## Try It

**Live environment (no setup):** [https://huggingface.co/spaces/IshikaMahadar/resume-env](https://huggingface.co/spaces/IshikaMahadar/resume-env)

**Interactive API docs:** [https://ishikamahadar-resume-env.hf.space/docs](https://ishikamahadar-resume-env.hf.space/docs)

**Quick curl test:**

```bash
# Start an episode
curl -X POST https://ishikamahadar-resume-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_type": "hard", "seed": 42}'

# Take an action as the Fraud Specialist
curl -X POST https://ishikamahadar-resume-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "verify_credential"}'
```

**Run inference with any OpenAI-compatible model:**

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export ENV_URL="https://ishikamahadar-resume-env.hf.space"
python inference_fleet.py
```

---

## What's Next

The most impactful next step is **online GRPO training** — replacing the offline proxy reward with live environment calls during gradient updates. This would let the terminal reward signal (+0.35 for correct decision, +0.25 for correct fraud flag) propagate directly back through the causal chain, rather than relying on format proxies.

A second direction is **procedural resume generation** to expand beyond the current 36 static episodes. A larger, more varied dataset would prevent the model from relying on pattern memorisation and force genuine reasoning.
