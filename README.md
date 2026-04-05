---
title: Adversarial Resume Screening Environment
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
tags: [openenv]
---

# Adversarial Resume Screening 📄♎

## 💡 Overview & Motivation
As automated hiring systems become ubiquitous, they become targets for **Adversarial Resumes**—CVs specifically crafted with fabricated credentials or "keyword stuffing" to trick AI filters.  

The **Adversarial Resume Screening Environment** provides a benchmark for evaluating the robustness of agentic AI in HR technology. Unlike simple parsers, agents in this environment must engage with complex, multi-modal evidence (Experience vs. Age, Credential Validation) to make fair and accurate hiring decisions while identifying fraudulent submissions.

---

## 🛠️ System Specifications

### 📝 Observation Space
The agent receives a `ResumeObservation` object with:
- **`resume_text`**: The raw text of the candidate's CV.
- **`job_description`**: The specific requirements and "red lines" for the role.
- **`task_type`**: Metadata indicating the complexity level for benchmarking.

### 🛠️ Action Space
The agent provides a `ResumeAction` decision:
- **`decision`**: `accept` or `reject`.
- **`fraud_flag`**: `true` if suspicious data points are found.
- **`confidence`**: (float 0.0-1.0) Indicating certainty level.

### 💰 Reward Function (v1.1)
The environment uses a multi-faceted reward signal (0.0 to 1.0):
- **Decision Accuracy (+0.5)**: Awarded for matching the ground-truth hiring decision.
- **Fraud Detection (+0.3)**: Awarded for correctly identifying (or correctly clearing) fraud.
- **Confidence Calibration (+0.2)**: Awarded for high-confidence correct decisions.

---

## 🎮 Task Difficulty & Graders

| Task Level | Description | Expected Difficulty |
| :--- | :--- | :--- |
| **Easy** | Clear match or clear mismatch with no fraud. | ⭐ (Low) |
| **Medium** | Subtle skill mismatches requiring careful JD analysis. | ⭐⭐ (Medium) |
| **Hard** | Adversarial cases: Professional CVs with impossible dates or fake credentials. | ⭐⭐⭐ (High) |

---

## 🚀 Setup & Usage

### 1. Local Setup
```bash
# Install requirements
pip install -r requirements.txt

# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 2. Docker Setup
```bash
docker build -t resume-env .
docker run -p 7860:7860 resume-env
```

### 3. Inference & Evaluation
Configure your `.env` with your LLM provider (OpenAI/Groq) and the `ENV_URL` pointing to your Space, then run:
```bash
python3 inference.py
```

---

## 📊 Baseline Scores
Evaluated using **Llama-3.3-70b** on Groq across the full 5-resume benchmark set.

| Metric | Score |
| :--- | :--- |
| **Success Rate** | 100% |
| **Aggregate Score** | **1.000** / 1.000 |
| **Avg. Reward/Step** | 1.00 |

---
**OpenEnv Compliance**: Level 3 (Production Ready) ✅  
**Deployment**: [ishikamahadar-resume-env.hf.space](https://huggingface.co/spaces/IshikaMahadar/resume-env)
