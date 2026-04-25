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

# Hiring Fleet: AI Oversight System

> **Round 2 — Fleet AI / Scalable Oversight sub-theme**
> Multi-agent resume screening where specialist AIs investigate in sequence and an Overseer synthesises their reports.

---

## Overview & Motivation

Automated hiring systems are increasingly targeted by **adversarial resumes** — CVs crafted with fabricated credentials, inflated titles, and keyword stuffing to bypass AI filters. While Round 1 used a single agent to catch these, a single investigator can be fooled by a sophisticated forgery that hides its flaws across different resume sections.

Round 2 addresses this with a **Hiring Fleet**: a pipeline of specialist agents, each with a narrow, focused scope, whose independent findings are synthesised by a human-like Overseer. This mirrors how real enterprise HR workflows use dedicated background-check teams before a final hiring authority decides.

The key research question: **can scalable AI oversight — multiple constrained specialists feeding an Overseer — outperform a single capable agent at detecting complex fraud?**

---

## Architecture: Four-Phase Fleet

Each episode runs four sequential phases. Every phase is a separate agent with its own step budget, role instructions, and **hard-enforced action whitelist**.

```
┌─────────────────────────────────────────────────────────────┐
│                    HIRING FLEET EPISODE                     │
│                                                             │
│  ① Fraud Specialist  →  ② Skills Specialist                │
│       (fraud/ref checks)      (technical fit)               │
│                                                             │
│  ③ Timeline Specialist  →  ④ Overseer                      │
│       (chronology/gaps)       (synthesises + decides)       │
└─────────────────────────────────────────────────────────────┘
```

### Phase Budgets

| Tier | Fraud | Skills | Timeline | Overseer | **Total** |
|:---|:---:|:---:|:---:|:---:|:---:|
| Easy | 2 | 2 | 2 | 2 | **8** |
| Medium | 2 | 3 | 3 | 3 | **11** |
| Hard | 3 | 4 | 4 | 4 | **15** |

Each specialist submits a structured `SpecialistReport` (findings, `has_issues`, confidence) before the next phase begins. If a specialist exhausts their budget without submitting, the environment auto-submits with `has_issues=False` and `confidence=0.3`.

---

## Specialist Roles & Constraints

### Role-Based Action Whitelists

Each specialist can only use the actions relevant to their function. Attempting an out-of-role action is rejected, costs a step, and increments `violations_count`. Each violation deducts **0.05** from the final reward.

| Phase | Allowed Actions | Allowed Sections |
|:---|:---|:---|
| `fraud_specialist` | `check_reference`, `verify_credential`, `view_section`, `submit_specialist_report` | `header`, `references` |
| `skills_specialist` | `view_section`, `ask_clarification`, `submit_specialist_report` | `experience`, `education`, `skills`, `projects` |
| `timeline_specialist` | `view_section`, `ask_clarification`, `submit_specialist_report` | `header`, `summary`, `experience` |
| `overseer` | `read_reports`, `request_reinvestigation`, `submit_final_decision` | *(report summaries only)* |

**Per-phase section isolation**: `_sections_viewed` resets at each phase transition. Each specialist earns reward for viewing their own allowed sections independently — the timeline specialist can view `experience` even if skills already viewed it. Re-viewing the same section within a phase returns a hint but costs no step.

**Observation role-filtering**: a skills specialist cannot see sections revealed by the fraud specialist. Each agent only sees what they're supposed to see.

### Overseer Capabilities

The Overseer operates differently from specialists:
- **`read_reports`** — explicitly requests the full enriched text of one specialist report (`report_target: fraud_specialist | skills_specialist | timeline_specialist`). Each read earns +0.02; reading all three earns an additional +0.03 thoroughness bonus.
- **`request_reinvestigation`** — sends one specialist phase back for a second pass (used at most once). Earns +0.05 if the final decision is correct.
- **`submit_final_decision`** — terminal action with `decision`, `fraud_flag`, `confidence`, `fraud_reasoning`.

---

## Observation & Action Spaces

### `FleetObservation`

| Field | Type | Description |
|:---|:---|:---|
| `task_type` | `"easy"\|"medium"\|"hard"` | Difficulty tier |
| `current_phase` | `"fraud_specialist"\|...\|"overseer"\|"complete"` | Active agent |
| `role_instructions` | `string` | Instructions for the current agent |
| `job_description` | `string` | Role requirements |
| `visible_sections` | `Dict[str, str]` | Role-filtered resume sections |
| `specialist_reports` | `List[SpecialistReport]` | Reports from completed phases |
| `available_actions` | `List[str]` | Dynamically filtered valid actions |
| `clarification_response` | `string\|null` | Answer to last clarification |
| `reference_response` | `string\|null` | Last reference check result |
| `verification_result` | `string\|null` | Last credential verification |
| `steps_remaining` | `int` | Steps left in current phase |
| `total_steps_remaining` | `int` | Steps left across all phases |
| `violations_count` | `int` | Out-of-role violations this episode |
| `reports_read` | `List[str]` | Specialist roles the overseer has read |
| `read_report_details` | `Dict[str, str]` | Enriched report text per specialist |
| `feedback` | `string\|null` | Environment hints / violation messages |

### `FleetAction`

| Action | Phase | Key Fields |
|:---|:---|:---|
| `view_section` | Specialists | `section` |
| `ask_clarification` | Skills / Timeline | `question` |
| `check_reference` | Fraud | `reference_id` (ref1/ref2) |
| `verify_credential` | Fraud | — |
| `submit_specialist_report` | Specialists | `findings`, `has_issues`, `specialist_confidence` |
| `read_reports` | Overseer | `report_target` |
| `request_reinvestigation` | Overseer | `reinvestigation_target`, `reinvestigation_reason` |
| `submit_final_decision` | Overseer | `decision`, `fraud_flag`, `confidence`, `fraud_reasoning` |

---

## Reward Function

Rewards accumulate across all four phases. All values are in **[0.0, 1.0]** (clamped).

### Per-Step Rewards (investigation quality)

| Action | Condition | Reward |
|:---|:---|:---:|
| `view_section` | High-value section (experience/education/skills) | +0.03 |
| `view_section` | Other section | +0.01 |
| `ask_clarification` | Substantive answer | +0.03 |
| `ask_clarification` | Generic answer | +0.01 |
| `check_reference` | Reveals fraud signal | +0.05 |
| `check_reference` | Clean reference | +0.02 |
| `check_reference` | Reference not found | +0.00 |
| `verify_credential` | Reveals FAILED | +0.05 |
| `verify_credential` | All verified | +0.02 |
| `read_reports` | Per specialist report read | +0.02 |
| `read_reports` | All 3 reports read (bonus) | +0.03 |

### Specialist Report Scoring

Each `submit_specialist_report` earns a base bonus if the specialist's `has_issues` flag matches ground truth, **scaled by difficulty tier**:

```
per_specialist_bonus = (0.20 / 3) × tier_multiplier
tier_multiplier = easy: 1.0 | medium: 1.1 | hard: 1.3
```

### Terminal Decision Reward (Overseer)

| Component | Condition | Reward |
|:---|:---|:---:|
| Decision correct | `decision` matches ground truth | +0.35 |
| Decision wrong | Mismatch | −0.35 |
| Fraud flag correct | `fraud_flag` matches `is_fraud` | +0.25 |
| Fraud flag wrong | Mismatch | −0.25 |
| Confidence calibration | ≥ 0.7 confidence + both correct | +0.10 |
| Fraud reasoning quality | Mentions fraud indicator keywords | +0.10 |
| Reinvestigation bonus | Used + final decision correct | +0.05 |
| Fleet coordination bonus | All 3 specialists correct + overseer correct | **+0.03** |
| Step efficiency bonus | Both correct + unused budget | **+0.04 × (1 − steps_used/max_steps)** |
| Violation penalty | Per out-of-role action | **−0.05 each** |

**Total reward range**: [0.0, 1.0] per episode. Negative intermediates are clamped to 0.0 at terminal.

---

## Dataset

36 resumes across three difficulty tiers (12 each). **Fraud ratio: 42% per tier** (5/12), balanced for GRPO reward variance.

| Tier | Episodes | Total Steps | Description |
|:---|:---:|:---:|:---|
| **Easy** | 12 | 8 | Clear match/mismatch, obvious fraud (impossible timelines, fake institutions, role-mismatch references) |
| **Medium** | 12 | 11 | Subtle skill gaps, partial matches, embellished but plausible resumes |
| **Hard** | 12 | 15 | Title inflation, scope exaggeration, references that contradict claims, sophisticated fabrication |

Every resume includes:
- `required_skills` — extracted from job description, used for skills enrichment
- `employment_gaps` — derived from fraud indicators, used for timeline enrichment
- `reference_check_results` — ref1 (always present) and ref2 (present on fraud resumes, often a denial)
- `verification_data` — at least one `False` entry on all fraud resumes (so `verify_credential` gives the correct 0.05 signal)

Graders are **fully deterministic** — all reward computation uses ground-truth fields (`is_fraud`, `fraud_indicators`, `employment_gaps`, `required_skills`). No LLM judge required.

---

## Setup & Usage

### Local Server

```bash
pip install -r requirements.txt
uvicorn server.fleet_app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t hiring-fleet .
docker run -p 7860:7860 hiring-fleet
```

### Running Fleet Inference

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="your-api-key"
export ENV_URL="http://localhost:7860"   # or your HF Space URL

python inference_fleet.py
```

`inference_fleet.py` runs 9 fleet episodes (3 per difficulty tier), routing each phase to the correct specialist/overseer prompt. It emits structured `[PHASE]`/`[STEP]`/`[END]` logs and handles action parsing, fallback actions, and overseer report-reading automatically.

### Running Tests

```bash
# End-to-end tests (no server / no LLM needed)
python test_e2e_local.py

# Day 3 overseer unit tests
python test_day3_overseer.py
```

---

## GRPO Training

The environment is fully ready for GRPO training. All reward signals are deterministic and dense enough for stable gradient estimation.

### Prerequisites

```bash
pip install trl accelerate transformers torch datasets
```

### Recommended Model

`Qwen/Qwen2.5-1.5B-Instruct` — fits in ~3 GB VRAM, strong instruction-following, fast iteration.

### Platform Recommendations

| Platform | GPU | VRAM | Cost | Notes |
|:---|:---|:---|:---|:---|
| **Kaggle** | T4 ×2 | 32 GB | Free (30 h/week) | Best for first run |
| **Google Colab Pro** | T4 / A100 | 16–40 GB | ~$10/month | Easy fallback |
| **RunPod** | RTX 3090 | 24 GB | ~$0.44/hr | Best value for full runs |
| Mac M-series | MPS | shared | Free | Slow; no mixed precision |

A single T4 (16 GB) is sufficient for `Qwen2.5-1.5B-Instruct` with GRPO group_size=8.

### Running GRPO Training

```bash
export ENV_URL="https://ishikamahadar-resume-env.hf.space"  # or local
python train_grpo.py
```

### Two-Stage Training Roadmap

| Stage | When | Focus |
|:---|:---|:---|
| **Stage 1** (now) | First run | Role discipline, fraud detection, report quality — standard GRPO on the 12-component reward |
| **Stage 2** (after Stage 1 converges) | Theme #2 integration | Long-horizon planning bonus: reward coherent trajectories where specialist `has_issues` flags correctly predict the overseer's final decision across the full 4-phase episode |

Stage 2 aligns with **Theme #2 — Long-Horizon Planning + Scale AI Bonus**. The dense intermediate rewards (per specialist report) and delayed terminal reward (overseer decision) make this environment naturally suited for long-horizon RL. After Stage 1 establishes a base policy, Stage 2 adds a trajectory-level coherence bonus that incentivises agents to plan investigations with the final verdict in mind from the first phase.

---

## API Endpoints

| Method | Path | Description |
|:---|:---|:---|
| POST | `/reset` | Start a new episode, returns initial `FleetObservation` |
| POST | `/step` | Submit a `FleetAction`, returns next `FleetObservation` with reward |
| GET | `/state` | Current internal `FleetState` |
| GET | `/health` | Health check |
| GET | `/` | Web UI |

---

## Round 2 Changes vs Round 1

| | Round 1 | Round 2 |
|:---|:---|:---|
| **Architecture** | Single agent | 4-phase multi-agent fleet |
| **Action space** | 5 actions (flat) | 8 actions (role-gated) |
| **Observation** | Flat resume view | Role-filtered per specialist |
| **Decision maker** | Agent directly decides | Overseer synthesises 3 reports |
| **Role enforcement** | None | Hard whitelist + violation penalty |
| **Reward range** | [−1.0, 1.0] | [0.0, 1.0] |
| **Reward components** | 8 | 12 (tier-scaled, efficiency, coordination) |
| **Step budget** | easy=6 / med=8 / hard=10 | easy=8 / med=11 / hard=15 |
| **Inference script** | `inference.py` | `inference_fleet.py` |
| **Test coverage** | Minimal | 78 tests (59 unit + 19 E2E) |
| **Fraud balance** | ~17% easy/medium | 42% all tiers |
| **Section tracking** | Global across phases | Per-phase (each specialist earns independently) |
| **Re-view behavior** | Free (no step cost) | Free (no step cost) |

---

**OpenEnv Compliance**: v3.0.0
**Deployment**: [IshikaMahadar/resume-env](https://huggingface.co/spaces/IshikaMahadar/resume-env)
