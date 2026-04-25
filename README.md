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
> A multi-agent hiring pipeline where specialist AIs investigate resumes in sequence and an Overseer synthesises their findings into a final decision.

---

## 📎 Materials

| Resource | Link |
|:---|:---|
| 🌐 **Live Environment** | [HuggingFace Space — resume-env](https://huggingface.co/spaces/IshikaMahadar/resume-env) |
| 📓 **Colab Training Notebook** | [train_grpo_fleet.ipynb](./train_grpo_fleet.ipynb) |
| 🤖 **Trained LoRA Adapter** | [`grpo_results/grpo_fleet_output/final/`](./grpo_results/grpo_fleet_output/final/) |
| 📈 **Reward Curve** | See below — 0.736 → 0.850 over 800 steps |
| 💻 **GitHub Repo** | [Ishika-eng/OpenEnv-Meta-Hackathon](https://github.com/Ishika-eng/OpenEnv-Meta-Hackathon---Adversarial-Resume-Screening-Environment) |

---

## 🎯 The Problem

Every year, companies lose billions hiring fraudulent candidates — people who fake degrees, fabricate work history, and list references who will lie for them. AI-based resume screeners are increasingly the first line of defence. But a single AI agent, reading a resume alone, is easy to fool: sophisticated fraud hides clues in *different sections* of the resume, each one looking innocent in isolation.

**The real question is: can we build AI oversight that's harder to fool than a single agent?**

This project answers that question by building a **Hiring Fleet** — a team of four specialist AI agents who each investigate one narrow slice of the resume, then pass their findings to an Overseer who makes the final call. This mirrors how real enterprise HR works: separate teams for background checks, skills assessment, and employment verification, all feeding a hiring manager.

---

## 🏗️ How It Works: Four-Phase Pipeline

Each episode of the environment runs exactly four sequential phases. Every agent has:
- Its own **step budget** (limited moves)
- A **hard-enforced action whitelist** (can't use another specialist's tools)
- A **role-filtered view** (can only see the resume sections relevant to their job)

```
┌──────────────────────────────────────────────────────────────┐
│                    HIRING FLEET EPISODE                      │
│                                                              │
│  ① Fraud Specialist   →   ② Skills Specialist               │
│    verify credentials        check technical fit             │
│    call references           ask clarifying questions        │
│                                                              │
│  ③ Timeline Specialist  →   ④ Overseer                      │
│    find employment gaps       reads all 3 reports            │
│    spot date conflicts        issues final verdict           │
└──────────────────────────────────────────────────────────────┘
```

### What makes this novel

| Design choice | Why it matters |
|:---|:---|
| **Hard action whitelists per role** | Attempting an out-of-role action is rejected, costs a step, and deducts from reward. The agent must learn role discipline. |
| **Role-filtered observations** | Each specialist only sees the sections they're authorised to view. Skills specialist can't read what the Fraud specialist revealed. |
| **Per-phase section tracking** | Each specialist earns reward for viewing their allowed sections *independently*. Phase transitions reset the counter — preventing free-riders. |
| **Overseer synthesis** | The Overseer cannot view sections directly. It must call `read_reports` explicitly to get each specialist's enriched findings — rewarded for reading all three. |
| **Deterministic, LLM-free reward** | All 12 reward components are computed from ground-truth fields. No judge model required. Stable gradients for GRPO. |

### Step Budgets

| Difficulty | Fraud | Skills | Timeline | Overseer | Total |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Easy | 2 | 2 | 2 | 2 | **8** |
| Medium | 2 | 3 | 3 | 3 | **11** |
| Hard | 3 | 4 | 4 | 4 | **15** |

---

## 📊 Reward Function (12 components)

Rewards are dense across all four phases — every useful action earns something, which gives GRPO stable signal throughout the episode.

### Per-step rewards

| Action | Condition | Reward |
|:---|:---|:---:|
| `view_section` | High-value (experience/education/skills) | +0.03 |
| `view_section` | Other | +0.01 |
| `ask_clarification` | Substantive answer | +0.03 |
| `check_reference` | Reveals fraud signal | +0.05 |
| `check_reference` | Clean reference | +0.02 |
| `verify_credential` | Reveals FAILED | +0.05 |
| `verify_credential` | All verified | +0.02 |
| `read_reports` | Per report read | +0.02 |
| `read_reports` | All 3 read (thoroughness bonus) | +0.03 |

### Terminal rewards (Overseer decision)

| Component | Reward |
|:---|:---:|
| Correct accept/reject decision | **+0.35** |
| Correct fraud flag | **+0.25** |
| Confident + correct (≥0.7) | +0.10 |
| Fraud reasoning quality (keyword match) | +0.10 |
| Reinvestigation used + correct | +0.05 |
| Fleet coordination (all specialists + overseer correct) | +0.03 |
| Step efficiency (correct + unused budget) | up to +0.04 |
| Out-of-role violation | **−0.05 each** |

**Total range: [0.0, 1.0]** — clamped at terminal. No LLM judge needed.

---

## 📁 Dataset

**36 resumes** across three difficulty tiers (12 each). **Fraud ratio: 42% per tier** — balanced for GRPO reward variance (too few fraud cases → sparse signal; too many → trivial reject policy).

| Tier | Fraud | Description |
|:---|:---:|:---|
| **Easy** | 5/12 | Obvious fraud: role-mismatch references, impossible timelines, fake institutions |
| **Medium** | 5/12 | Embellished resumes: scope exaggeration, compliance misrepresentation, plausible-but-false credentials |
| **Hard** | 5/12 | Sophisticated fabrication: title inflation, references that contradict claims, multi-section inconsistencies |

Every fraud resume guarantees `verify_credential` returns FAILED and `check_reference(ref2)` returns a suspicious or denying response — so the Fraud Specialist always has a detectable signal.

---

## 🤖 GRPO Training

### Results

| Metric | Value |
|:---|:---|
| **Model** | Qwen/Qwen2.5-1.5B-Instruct + LoRA (r=16) |
| **Training** | GRPO, 2 epochs, 800 steps, T4 GPU |
| **Start reward** | 0.736 |
| **Best reward** | 0.850 |
| **Improvement** | **+15.5%** |

![GRPO Reward Curve](assets/reward_curve.png)

The reward curve shows the model learning to:
1. Output valid JSON action objects reliably
2. Select role-appropriate actions (no violations)
3. Prioritise `verify_credential` as the Fraud Specialist's first move
4. Include fraud indicator keywords in reasoning (`failed`, `denied`, `fabricated`)

### How to train

Open [train_grpo_fleet.ipynb](./train_grpo_fleet.ipynb) in Google Colab (T4 GPU, free tier sufficient).

```
Cell 0: pip install deps
Cell 1: set ENV_URL
Cell 2: paste all training code
Cell 3: main()   ← runs data collection + training
Cell 4: plot reward / KL / loss curves
Cell 5: ls saved adapter
Cell 6: download results zip
```

The notebook uses **offline data collection** (rule-based agent walks the environment — no GPU needed) then **GRPOTrainer generates completions internally** and scores them with a static reward function. No live environment calls during the gradient update step.

---

## 🚀 Running the Environment

### Live (no setup)
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

### API Endpoints

| Method | Path | Description |
|:---|:---|:---|
| POST | `/reset` | Start a new episode |
| POST | `/step` | Submit an action, get next observation + reward |
| GET | `/state` | Current episode state |
| GET | `/health` | Health check |

---

## 🔬 Technical Details

### Action Space (8 actions, role-gated)

| Action | Phase | Key Fields |
|:---|:---|:---|
| `view_section` | All specialists | `section` |
| `ask_clarification` | Skills, Timeline | `question` |
| `check_reference` | Fraud only | `reference_id` (ref1/ref2) |
| `verify_credential` | Fraud only | — |
| `submit_specialist_report` | All specialists | `findings`, `has_issues`, `specialist_confidence` |
| `read_reports` | Overseer only | `report_target` |
| `request_reinvestigation` | Overseer only | `reinvestigation_target`, `reinvestigation_reason` |
| `submit_final_decision` | Overseer only | `decision`, `fraud_flag`, `confidence`, `fraud_reasoning` |

### Observation Space

Each observation includes: `current_phase`, `role_instructions`, `job_description`, `visible_sections` (role-filtered), `specialist_reports`, `available_actions`, `reference_response`, `verification_result`, `clarification_response`, `steps_remaining`, `total_steps_remaining`, `violations_count`, `reports_read`, `read_report_details`, `feedback`.

---

## 📋 OpenEnv Compliance

- **Spec version**: 1
- **`openenv.yaml`**: [view](./openenv.yaml)
- **Deployment**: [IshikaMahadar/resume-env](https://huggingface.co/spaces/IshikaMahadar/resume-env)
- **Version**: 3.0.0
