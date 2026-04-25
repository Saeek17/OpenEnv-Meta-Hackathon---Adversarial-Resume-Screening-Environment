---
title: Adversarial Resume Screening — Hiring Fleet
emoji: "🛡️"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
tags: [openenv]
---

# 🛡️ Hiring Fleet: AI Oversight System

> **Can four constrained AI specialists catch fraud that fools a single agent?**
>
> A multi-agent hiring pipeline where specialist AIs each investigate one narrow slice of a resume, then an Overseer synthesises their findings — without ever seeing the raw evidence itself.

---

## 📎 Materials

| Resource | Link |
|:---|:---|
| 🌐 **Live Environment** | [HuggingFace Space — resume-env](https://huggingface.co/spaces/IshikaMahadar/resume-env) |
| 📓 **Colab Training Notebook** | [train_grpo_fleet.ipynb](./train_grpo_fleet.ipynb) |
| 🤖 **Trained LoRA Adapter** | [IshikaMahadar/hiring-fleet-grpo-adapter](https://huggingface.co/IshikaMahadar/hiring-fleet-grpo-adapter) |
| 📈 **GRPO Reward Curve** | 0.736 → 0.850 over 800 steps — see Training section |
| 📊 **Evaluation Results** | Rule-based 0.873 vs Fine-tuned 0.870 — see Evaluation section |
| 💻 **GitHub Repo** | [Ishika-eng/OpenEnv-Meta-Hackathon](https://github.com/Ishika-eng/OpenEnv-Meta-Hackathon---Adversarial-Resume-Screening-Environment) |
| 🔗 **Interactive API Docs** | [/docs](https://ishikamahadar-resume-env.hf.space/docs) — try the environment in browser |

---

## 🎯 The Problem

Every year, companies lose billions hiring fraudulent candidates — people who fake degrees, fabricate work history, and list references who will lie for them. AI resume screeners are the first line of defence. But a **single AI agent reading a resume alone is easy to fool**: sophisticated fraud hides clues across *different sections*, each one looking innocent in isolation.

A fake institution in the header. A reference who denies the employment in the references section. Impossible overlapping dates in experience. No single section reveals the fraud — only the *combination* does.

**The real question: can we build AI oversight that's harder to fool than a single agent?**

This project answers it by building a **Hiring Fleet** — four specialist AI agents who each investigate one narrow slice of the resume, then pass findings to an Overseer who makes the final call. The Overseer cannot see the raw resume. It must reason entirely from what the specialists report. If the specialists write poor reports, the Overseer has poor signal. The chain of reasoning is real.

---

## 🏗️ Architecture: Four-Phase Pipeline

Each episode runs exactly four sequential phases. Every agent has its own step budget, a hard-enforced action whitelist, and a role-filtered observation — it cannot see sections outside its domain.

```
┌─────────────────────────────────────────────────────────────────────┐
│                       HIRING FLEET EPISODE                          │
│                                                                     │
│  ① Fraud Specialist      →    ② Skills Specialist                  │
│    view: header, references        view: experience, education,     │
│    tools: verify_credential              skills, projects           │
│            check_reference         tools: ask_clarification        │
│    submits: fraud report           submits: skills report           │
│                                                                     │
│  ③ Timeline Specialist   →    ④ Overseer                           │
│    view: header, summary,          cannot view sections directly    │
│          experience                must call read_reports per agent │
│    tools: ask_clarification        tools: request_reinvestigation   │
│    submits: timeline report        submits: final hiring decision   │
└─────────────────────────────────────────────────────────────────────┘
```

### Role Section Access

| Specialist | Sections Allowed | Tools Available |
|:---|:---|:---|
| Fraud Specialist | `header`, `references` | `verify_credential`, `check_reference` |
| Skills Specialist | `experience`, `education`, `skills`, `projects` | `ask_clarification` |
| Timeline Specialist | `header`, `summary`, `experience` | `ask_clarification` |
| Overseer | ❌ none (report synthesis only) | `read_reports`, `request_reinvestigation` |

### Step Budgets

| Difficulty | Fraud | Skills | Timeline | Overseer | Total |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Easy | 2 | 2 | 2 | 2 | **8** |
| Medium | 2 | 3 | 3 | 3 | **11** |
| Hard | 3 | 4 | 4 | 4 | **15** |

### What Makes This Novel

| Design choice | Why it matters |
|:---|:---|
| **Hard action whitelists per role** | Out-of-role actions are rejected, cost a step, and deduct −0.05 from terminal reward. Agents must learn role discipline, not just task completion. |
| **Role-filtered observations** | Each specialist sees only their authorised sections. The fraud specialist cannot read experience; the skills specialist cannot read references. Information is genuinely siloed. |
| **No ground truth in Overseer reports** | When the Overseer reads specialist reports, it receives only the specialist's *own findings* — no environment-injected answers. Overseer decision quality depends entirely on specialist reasoning quality. |
| **Per-phase section tracking with reset** | Reward for `view_section` resets per phase. A specialist cannot earn credit by re-reading sections another specialist already viewed. Prevents free-rider behaviour. |
| **Deterministic, LLM-free reward** | All reward components are computed from ground-truth fields. No judge model required. Stable, reproducible gradients for GRPO. |
| **Fleet coordination bonus** | A separate +0.08 reward fires only when all three specialists AND the Overseer are simultaneously correct — explicitly incentivising multi-agent cooperation over single-agent shortcuts. |

---

## 📊 Reward Function

Rewards are dense across all four phases — every useful action earns signal, giving GRPO stable gradients throughout the episode. The terminal reward delegates to **seven independent named sub-functions**, satisfying the build guide requirement for multiple independent reward components.

### Per-Step Rewards (all phases)

| Action | Condition | Reward |
|:---|:---|:---:|
| `view_section` | High-value section (experience/education/skills) | +0.03 |
| `view_section` | Other section | +0.01 |
| `ask_clarification` | Substantive answer returned | +0.03 |
| `check_reference` | Reveals fraud signal | +0.05 |
| `check_reference` | Clean reference | +0.02 |
| `verify_credential` | Returns FAILED | +0.05 |
| `verify_credential` | All verified | +0.02 |
| `submit_specialist_report` | Correct signal + calibrated confidence | up to +0.10 |
| `read_reports` | Per unique report read | +0.02 |
| `read_reports` | All 3 reports read (thoroughness bonus) | +0.03 |

### Terminal Reward — 7 Independent Sub-Functions

```
_handle_submit_final_decision calls:

  A  _reward_decision_accuracy     → up to +0.70
  B  _reward_specialist_quality    → up to +0.22  (tier-scaled)
  C  _reward_fleet_coordination    → +0.08         (all 4 correct)
  D  _reward_oversight_quality     → up to +0.08   (read thoroughness + reinvestigation)
  E  _reward_investigation_quality → up to +0.05   (breadth × tool depth)
  F  _reward_format_compliance     → up to +0.05   (fraud reasoning keyword match)
  G  _reward_step_efficiency       → up to +0.04   (when correct + budget remaining)
```

| Sub-function | What it rewards | Max |
|:---|:---|:---:|
| A: Decision accuracy | Correct accept/reject (+0.35) + correct fraud flag (+0.25) + calibrated confidence (+0.10) | **+0.70** |
| B: Specialist quality | Each correct specialist earns `(0.20/3) × tier_mult`; hard cases worth more | **+0.22** |
| C: Fleet coordination | Bonus when all 3 specialists AND Overseer are simultaneously correct | **+0.08** |
| D: Oversight quality | Read all reports (+0.04) + appropriate reinvestigation use (+0.04) | **+0.08** |
| E: Investigation depth | `depth_ratio × 0.6 + tool_ratio × 0.4` across all phases | **+0.05** |
| F: Format compliance | Fraud reasoning keyword match against known fraud indicators | **+0.05** |
| G: Step efficiency | Rewards decisive correct agents who don't exhaust their full budget | **+0.04** |

### Anti-Exploit Penalties

Three adversarially-designed checks close shortcut strategies:

| Exploit closed | Penalty |
|:---|:---:|
| Always output high confidence regardless of correctness | −0.05 when wrong + confidence ≥ 0.7 |
| Flag fraud but write no reasoning (`fraud_flag=True`, empty text) | −0.05 when reasoning < 15 chars |
| Skip all investigation tools on fraud resumes | −0.05 when `is_fraud=True` and zero tools used |
| Out-of-role action attempts | −0.05 each (max −0.25) |

**Total range: [0.0, 1.0]** — clamped at terminal. No LLM judge required.

---

## 📁 Dataset

**36 resumes** across three difficulty tiers (12 each). Fraud ratio: **42% per tier** — balanced to ensure GRPO reward variance (too few → sparse signal; too many → trivial always-reject policy).

| Tier | Resumes | Fraud | Fraud Type |
|:---|:---:|:---:|:---|
| **Easy** | 12 | 5 | Obvious: role-mismatch references, impossible timelines, fake institutions |
| **Medium** | 12 | 5 | Subtle: scope exaggeration, compliance misrepresentation, plausible-but-false credentials |
| **Hard** | 12 | 5 | Sophisticated: title inflation, references that contradict claims, multi-section inconsistencies |

Every fraud resume guarantees:
- `verify_credential` → returns FAILED
- `check_reference(ref2)` → returns suspicious or denying response

The Fraud Specialist always has a detectable signal — the challenge is recognising it and propagating it correctly through the fleet.

---

## 🔬 Action Space (8 actions, role-gated)

| Action | Available Phase | Required Fields |
|:---|:---|:---|
| `view_section` | Fraud, Skills, Timeline | `section` |
| `ask_clarification` | Skills, Timeline | `question` |
| `check_reference` | Fraud only | `reference_id` (`ref1` or `ref2`) |
| `verify_credential` | Fraud only | — |
| `submit_specialist_report` | Fraud, Skills, Timeline | `findings`, `has_issues`, `specialist_confidence` |
| `read_reports` | Overseer only | `report_target` (`fraud_specialist` \| `skills_specialist` \| `timeline_specialist`) |
| `request_reinvestigation` | Overseer only | `reinvestigation_target`, `reinvestigation_reason` |
| `submit_final_decision` | Overseer only | `decision`, `fraud_flag`, `confidence`, `fraud_reasoning` |

Attempting any action outside your current phase's whitelist → **rejected**, step consumed, violation recorded.

---

## 👁️ Observation Space

Every `step` response returns a full observation:

| Field | Type | Description |
|:---|:---|:---|
| `current_phase` | string | `fraud_specialist` \| `skills_specialist` \| `timeline_specialist` \| `overseer` \| `complete` |
| `role_instructions` | string | Detailed instructions for the active agent |
| `job_description` | string | Requirements for the open position |
| `visible_sections` | object | Role-filtered resume sections (only what this phase may see) |
| `specialist_reports` | array | Reports submitted by completed specialist phases |
| `available_actions` | array | Dynamically filtered valid actions for current phase |
| `reference_response` | string | Result of `check_reference` call |
| `verification_result` | string | Result of `verify_credential` call |
| `clarification_response` | string | Result of `ask_clarification` call |
| `read_report_details` | object | Full specialist report text (populated by `read_reports`) |
| `steps_remaining` | integer | Steps left in current phase |
| `total_steps_remaining` | integer | Steps left across entire episode |
| `violations_count` | integer | Out-of-role action count this episode |
| `reports_read` | array | Specialist roles the Overseer has explicitly read |
| `feedback` | string | Environment feedback on last action |
| `reward` | float | Incremental reward for this step |
| `done` | boolean | True when episode is complete |

---

## 🌐 API Endpoints

| Method | Path | Description |
|:---|:---|:---|
| `POST` | `/reset` | Start a new episode. Body: `{"task_type": "easy\|medium\|hard", "seed": int}` |
| `POST` | `/step` | Submit an action. Body: `{"action_type": "...", ...fields}` |
| `GET` | `/state` | Current episode state |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Interactive Swagger UI — try the environment in your browser |

### Quick Test (curl)

```bash
# Start an episode
curl -X POST https://ishikamahadar-resume-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_type": "easy", "seed": 42}'

# Take an action
curl -X POST https://ishikamahadar-resume-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "verify_credential"}'
```

Or open [/docs](https://ishikamahadar-resume-env.hf.space/docs) for the interactive Swagger UI — no code needed.

---

## 🤖 GRPO Training

### Training Setup

| Parameter | Value |
|:---|:---|
| **Base model** | Qwen/Qwen2.5-1.5B-Instruct |
| **Adapter** | LoRA (r=16, alpha=32, target: q_proj + v_proj) |
| **Framework** | HuggingFace TRL — `GRPOTrainer` |
| **Hardware** | T4 GPU (Colab free tier) |
| **Epochs** | 2 |
| **Steps** | 800 |
| **Data collection** | 36 offline episodes via rule-based agent (no GPU needed) |

### Training Pipeline

The notebook uses **offline data collection + static reward** — the rule-based agent walks the live environment to collect observation prompts, then GRPOTrainer generates completions and scores them with a proxy reward function. No live environment calls during gradient updates.

```
Cell 0: pip install trl accelerate transformers peft datasets
Cell 1: set ENV_URL (HF Space URL)
Cell 2: data collection + model loading + GRPO training code
Cell 3: main()  ← collects 36 episodes, trains 800 steps
Cell 4: plot reward / KL / loss curves
Cell 5: ls saved LoRA adapter
Cell 6: download results zip
```

### GRPO Reward Curve

![GRPO Reward Curve](assets/reward_curve.png)

| Metric | Value |
|:---|:---|
| **Start reward** | 0.736 |
| **Best reward** | 0.850 |
| **Improvement** | **+15.5%** |

### What the model learned from GRPO

1. **Output valid JSON reliably** — format compliance went from ~40% to ~95%
2. **Select role-appropriate actions** — violation rate dropped significantly
3. **Prioritise `verify_credential`** as the Fraud Specialist's first move
4. **Write fraud indicator keywords** in reasoning (`failed`, `denied`, `fabricated`)

---

## 📊 Baseline Evaluation (Live Environment, 9 Episodes)

Evaluated against the live HF Space environment — 3 episodes per difficulty tier, comparing the rule-based baseline against the GRPO fine-tuned model.

![Baseline Chart](assets/baseline_chart.png)

![Comparison Chart](assets/comparison_chart.png)

| Agent | Easy | Medium | Hard | **Overall** |
|:---|:---:|:---:|:---:|:---:|
| Rule-based baseline | 0.747 | 0.873 | 1.000 | **0.873** |
| **Fine-tuned (GRPO)** | 0.722 | 0.888 | 1.000 | **0.870** |

### What the 0.4% gap actually means

The small difference is expected and informative. GRPO training used a **static proxy reward** (JSON formatting, action types, fraud keywords) — not live environment rewards. The terminal reward (+0.35/+0.25) requires multi-step conditional reasoning:

```
verify_credential → FAILED → set has_issues=True → overseer reads report → reject
```

This causal chain was not directly trained. Yet on **medium difficulty fraud (seed=10)**, the fine-tuned model outperforms the rule-based agent at the specialist level:
- Fine-tuned step 2: `submit_specialist_report` with `has_issues=True` → reward **+0.08** (correct fraud detection)
- Rule-based step 2: `check_reference(ref1)` → reward **+0.00** (wasted step on clean reference)

GRPO taught the model to reason from evidence to conclusion at the specialist level. Stage 2 training with live environment rewards (where the terminal signal propagates back) would close the remaining gap.

---

## 🚀 Running the Environment

### Live (no setup required)
[https://huggingface.co/spaces/IshikaMahadar/resume-env](https://huggingface.co/spaces/IshikaMahadar/resume-env)

### Local
```bash
pip install -r requirements.txt
uvicorn server.fleet_app:app --host 0.0.0.0 --port 7860
```

### Docker
```bash
docker build -t hiring-fleet .
docker run -p 7860:7860 hiring-fleet
```

### Run inference with any OpenAI-compatible model
```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="your-api-key"
export ENV_URL="https://ishikamahadar-resume-env.hf.space"

python inference_fleet.py
```

---

## 📋 OpenEnv Compliance

| Field | Value |
|:---|:---|
| **Spec version** | 1 |
| **Environment version** | 3.0.0 |
| **Runtime** | FastAPI via `openenv-core` |
| **Reward range** | [0.0, 1.0] |
| **Tasks** | `easy` (12 episodes), `medium` (12), `hard` (12) |
| **`openenv.yaml`** | [view](./openenv.yaml) |
| **HF Space** | [IshikaMahadar/resume-env](https://huggingface.co/spaces/IshikaMahadar/resume-env) |
