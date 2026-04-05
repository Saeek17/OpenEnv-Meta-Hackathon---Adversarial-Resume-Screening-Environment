"""
Inference Script for Resume Screening Environment
===================================
MANDATORY
- Environment variables required:
    API_BASE_URL   The API endpoint for the LLM
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Your Hugging Face / API key (or API_KEY)
    
- The inference script must be named `inference.py`
- Must use OpenAI Client for all LLM calls

STDOUT FORMAT
- Exactly three line types to stdout:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin
    - One [STEP] line per step, immediately after env.step()
    - One [END] line after env.close(), always emitted
    - reward and rewards formatted to 2 decimal places
    - done and success are lowercase booleans: true or false
    - error is raw error string or null
    - All fields on single line
    - Score in [0, 1]
"""

import asyncio
import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI
from dotenv import load_dotenv

from client import ResumeEnv
from models import ResumeAction

# Load environment variables explicitly from the current directory
env_path = os.path.join(os.getcwd(), '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv()

# Deployment variables as defined in the mandatory submission checklist
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

# Local fallback for the environment server
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

# Task Configuration
TASK_NAME = os.getenv("RESUME_TASK", "resume-screening")
BENCHMARK = os.getenv("RESUME_BENCHMARK", "resume-eval")
NUM_EPISODES = 5

# Model Parameters
TEMPERATURE = 0.7
MAX_TOKENS = 500

# Scoring Configuration
SUCCESS_SCORE_THRESHOLD = 0.6
MAX_REWARD_PER_EPISODE = 1.0
MAX_TOTAL_REWARD = NUM_EPISODES * MAX_REWARD_PER_EPISODE

# System Prompt
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert recruitment specialist. Your task is to evaluate resumes against job descriptions.
    
    For each resume, you must:
    1. Determine if the candidate should be ACCEPTED or REJECTED
    2. Detect any potential FRAUD indicators
    3. Provide a CONFIDENCE score for your decision
    
    Guidelines:
    - Accept candidates who closely match the job requirements
    - Reject candidates with significant skill gaps or mismatches
    - Flag fraud if you detect fabricated experience, fake credentials, or inconsistencies
    - Confidence should reflect how certain you are (0.0 = very uncertain, 1.0 = absolutely certain)
    
    Respond ONLY with valid JSON in this exact format:
    {
        "decision": "accept" or "reject",
        "fraud_flag": true or false,
        "confidence": 0.0 to 1.0
    }
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(
        f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(
    episode: int,
    job_description: str,
    resume_text: str,
    history: List[str]
) -> str:
    history_block = "\n".join(history[-3:]) if history else "None"
    return textwrap.dedent(
        f"""
        Episode: {episode}
        
        Job Description:
        {job_description}
        
        Resume Text:
        {resume_text}
        
        Previous Evaluations:
        {history_block}
        
        Evaluate this resume and provide your decision in JSON format.
        """
    ).strip()


def get_model_decision(
    client: OpenAI,
    episode: int,
    job_description: str,
    resume_text: str,
    history: List[str]
) -> dict:
    fallback_action = {
        "decision": "reject",
        "fraud_flag": False,
        "confidence": 0.5
    }
    
    user_prompt = build_user_prompt(episode, job_description, resume_text, history)
    
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
        
        content = json.loads(response.choices[0].message.content or "{}")
        
        return {
            "decision": content.get("decision", "reject").lower(),
            "fraud_flag": bool(content.get("fraud_flag", False)),
            "confidence": float(content.get("confidence", 0.5))
        }
        
    except Exception:
        return fallback_action


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = ResumeEnv(base_url=ENV_URL)
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        for episode in range(1, NUM_EPISODES + 1):
            observation = await env.reset()
            
            action_dict = get_model_decision(
                client=client,
                episode=episode,
                job_description=observation.job_description,
                resume_text=observation.resume_text,
                history=history
            )
            
            action = ResumeAction(**action_dict)
            action_str = f"{action.decision}(fraud={action.fraud_flag},conf={action.confidence:.2f})"
            
            observation = await env.step(action)
            
            reward = getattr(observation, 'reward', 0.0)
            done = getattr(observation, 'done', episode == NUM_EPISODES)
            
            rewards.append(reward)
            steps_taken = episode
            
            log_step(episode, action_str, reward, done, None)
            history.append(f"Ep {episode}: {action.decision} -> rew={reward:+.2f}")
            
            # Note: We continue for NUM_EPISODES to get a full aggregate score
        
        total_reward = sum(rewards)
        score = min(max(total_reward / MAX_TOTAL_REWARD, 0.0), 1.0) if MAX_TOTAL_REWARD > 0 else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD
        
    except Exception:
        success = False
    
    finally:
        if hasattr(env, 'close'):
            await env.close()
        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())
