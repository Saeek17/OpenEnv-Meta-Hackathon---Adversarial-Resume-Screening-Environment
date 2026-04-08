"""
Inference Script for Adversarial Resume Screening Environment
=============================================================
Multi-step agent that investigates resumes by viewing sections, checking
references, verifying credentials, and asking clarifications before making
a hiring decision.

MANDATORY VARIABLES:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import sys
import textwrap
import warnings
from typing import List, Optional

warnings.filterwarnings("ignore")
# Suppress noisy library logs but keep stderr usable for real errors
import logging
logging.disable(logging.WARNING)

from openai import OpenAI
from dotenv import load_dotenv

from client import ResumeEnv
from models import ResumeAction

env_path = os.path.join(os.getcwd(), ".env")
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
ENV_URL = os.getenv("ENV_URL", "https://ishikamahadar-resume-env.hf.space")

BENCHMARK = "adversarial-resume-screening"
TASK_TYPES = ["easy", "medium", "hard"]
EPISODES_PER_TASK = 3  # 3 per tier = 9 total episodes
TEMPERATURE = 0.4
MAX_TOKENS = 600

# Scoring
SUCCESS_THRESHOLD = 0.4
MAX_REWARD_PER_EPISODE = 1.0
TOTAL_EPISODES = len(TASK_TYPES) * EPISODES_PER_TASK
MAX_TOTAL_REWARD = TOTAL_EPISODES * MAX_REWARD_PER_EPISODE

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert recruitment specialist evaluating resumes in a multi-step investigation environment.

Each turn you must output EXACTLY one JSON action. Available action types:
1. view_section: Reveal a resume section. Fields: {"action_type":"view_section","section":"<name>"}
   Sections: header, summary, experience, education, skills, projects, references
2. ask_clarification: Ask the candidate a question. Fields: {"action_type":"ask_clarification","question":"<text>"}
3. check_reference: Contact a reference. Fields: {"action_type":"check_reference","reference_id":"ref1"}
4. verify_credential: Verify education/employment/certifications. Fields: {"action_type":"verify_credential"}
5. submit_decision: Final decision. Fields: {"action_type":"submit_decision","decision":"accept|reject","fraud_flag":true|false,"confidence":0.0-1.0,"fraud_reasoning":"explanation if fraud"}

STRATEGY:
- First, view experience, education, and skills sections to understand the candidate.
- Then check references and verify credentials to validate claims.
- If anything seems suspicious, ask clarification questions.
- Only submit your decision after gathering sufficient evidence.
- Be thorough but efficient — you have a limited step budget.

Respond with ONLY valid JSON, no other text.
""")


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def build_user_prompt(obs_data: dict, step: int, history: List[str]) -> str:
    visible = obs_data.get("visible_sections", {})
    sections_text = ""
    for name, content in visible.items():
        sections_text += f"\n--- {name.upper()} ---\n{content}\n"

    clarification = obs_data.get("clarification_response") or ""
    reference = obs_data.get("reference_response") or ""
    verification = obs_data.get("verification_result") or ""

    extra = ""
    if clarification:
        extra += f"\nClarification response: {clarification}"
    if reference:
        extra += f"\nReference check result: {reference}"
    if verification:
        extra += f"\nCredential verification: {verification}"

    history_text = "\n".join(history[-5:]) if history else "None"
    remaining = obs_data.get("steps_remaining", 0)
    feedback = obs_data.get("feedback", "")

    return textwrap.dedent(f"""\
Step {step} | Steps remaining: {remaining}
Feedback: {feedback}

Job Description:
{obs_data.get('job_description', '')}

Revealed Sections:{sections_text}
{extra}

Previous actions:
{history_text}

Choose your next action (JSON only). If steps_remaining <= 1, you MUST submit_decision now.
""")


def parse_model_action(client: OpenAI, obs_data: dict, step: int, history: List[str]) -> dict:
    user_prompt = build_user_prompt(obs_data, step, history)

    fallback_actions = [
        {"action_type": "view_section", "section": "experience"},
        {"action_type": "view_section", "section": "education"},
        {"action_type": "view_section", "section": "skills"},
        {"action_type": "check_reference", "reference_id": "ref1"},
        {"action_type": "verify_credential"},
        {"action_type": "submit_decision", "decision": "reject", "fraud_flag": False, "confidence": 0.5,
         "fraud_reasoning": "insufficient information"},
    ]

    remaining = obs_data.get("steps_remaining", 0)
    if remaining <= 1:
        fallback = {"action_type": "submit_decision", "decision": "reject",
                     "fraud_flag": False, "confidence": 0.5, "fraud_reasoning": ""}
    elif step - 1 < len(fallback_actions):
        fallback = fallback_actions[step - 1]
    else:
        fallback = {"action_type": "submit_decision", "decision": "reject",
                     "fraud_flag": False, "confidence": 0.5, "fraud_reasoning": ""}

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            stream=False,
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)

        if "action_type" not in parsed:
            return fallback

        # Validate and clean up
        action = {"action_type": parsed["action_type"]}
        at = parsed["action_type"]

        if at == "view_section":
            action["section"] = parsed.get("section", "experience")
        elif at == "ask_clarification":
            action["question"] = parsed.get("question", "Can you elaborate on your experience?")
        elif at == "check_reference":
            action["reference_id"] = parsed.get("reference_id", "ref1")
        elif at == "verify_credential":
            pass
        elif at == "submit_decision":
            action["decision"] = parsed.get("decision", "reject")
            action["fraud_flag"] = bool(parsed.get("fraud_flag", False))
            action["confidence"] = float(parsed.get("confidence", 0.5))
            action["fraud_reasoning"] = parsed.get("fraud_reasoning", "")
        else:
            return fallback

        return action

    except Exception:
        return fallback


def action_to_str(action: dict) -> str:
    at = action.get("action_type", "unknown")
    if at == "view_section":
        return f"view_section({action.get('section', '')})"
    elif at == "ask_clarification":
        q = (action.get("question", ""))[:40]
        return f"ask_clarification({q})"
    elif at == "check_reference":
        return f"check_reference({action.get('reference_id', '')})"
    elif at == "verify_credential":
        return "verify_credential()"
    elif at == "submit_decision":
        d = action.get("decision", "")
        f = action.get("fraud_flag", False)
        c = action.get("confidence", 0)
        return f"submit_decision({d},fraud={f},conf={c:.2f})"
    return f"{at}()"


async def run_episode(client: OpenAI, env: ResumeEnv, task_type: str, episode_num: int) -> tuple:
    """Run a single multi-step episode. Returns (steps_taken, episode_rewards)."""
    observation = await env.reset()
    obs_data = observation.model_dump() if hasattr(observation, "model_dump") else observation.__dict__

    history: List[str] = []
    episode_rewards: List[float] = []
    step = 0

    task_name = f"resume-{task_type}-{episode_num}"
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        while True:
            step += 1
            done = obs_data.get("done", False)
            if done:
                break

            action_dict = parse_model_action(client, obs_data, step, history)
            action_str = action_to_str(action_dict)

            action = ResumeAction(**action_dict)
            observation = await env.step(action)
            obs_data = observation.model_dump() if hasattr(observation, "model_dump") else observation.__dict__

            reward = obs_data.get("reward", 0.0) or 0.0
            done = obs_data.get("done", False)

            episode_rewards.append(reward)
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)
            history.append(f"Step {step}: {action_str} -> reward={reward:+.2f}")

            if done:
                break

            if step >= 15:  # safety cap
                break

    except Exception as e:
        episode_rewards.append(0.0)
        log_step(step=step, action="error", reward=0.0, done=True, error=str(e))

    total = sum(episode_rewards)
    score = max(0.0, min(1.0, total / MAX_REWARD_PER_EPISODE)) if MAX_REWARD_PER_EPISODE > 0 else 0.0
    success = score >= SUCCESS_THRESHOLD
    log_end(success=success, steps=step, score=score, rewards=episode_rewards)

    return step, episode_rewards


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = ResumeEnv(base_url=ENV_URL)

    all_rewards: List[float] = []
    total_steps = 0

    for task_type in TASK_TYPES:
        for ep in range(1, EPISODES_PER_TASK + 1):
            steps, rewards = await run_episode(client, env, task_type, ep)
            total_steps += steps
            all_rewards.extend(rewards)

    if hasattr(env, "close"):
        try:
            await env.close()
        except Exception:
            pass

    overall_score = sum(all_rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
    overall_score = max(0.0, min(1.0, overall_score))


if __name__ == "__main__":
    asyncio.run(main())
