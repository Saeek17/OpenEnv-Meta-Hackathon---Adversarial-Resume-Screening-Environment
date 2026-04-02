---
title: Adversarial Resume Screening
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
---

# Adversarial Resume Screening Environment

A production-grade OpenEnv environment where an AI agent evaluates resumes against job descriptions and detects fraud.

## Project Structure
- `models.py`: Pydantic models for Actions, Observations, and State.
- `server/environment.py`: Core logic and reward computation.
- `server/app.py`: FastAPI server entry point.
- `client.py`: Client interface for AI agents.
- `inference.py`: Evaluation script using OpenAI.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Start the server: `uvicorn server.app:app`
3. Run inference: `python3 inference.py`
