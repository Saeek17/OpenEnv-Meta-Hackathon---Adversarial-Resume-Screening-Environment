import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from openenv.core.env_server import Environment
from models import ResumeObservation, ResumeAction, ResumeState


VALID_SECTIONS = ["header", "summary", "experience", "education", "skills", "projects", "references"]
MAX_STEPS_BY_DIFFICULTY = {"easy": 6, "medium": 8, "hard": 10}

# Sections that are more relevant for job evaluation (higher investigation reward)
HIGH_VALUE_SECTIONS = ["experience", "education", "skills"]


def load_dataset(path: str) -> Dict[str, list]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ["easy", "medium", "hard"]:
        if key not in data:
            raise ValueError(f"Dataset missing required category: {key}")
    return data


class ResumeScreeningEnvironment(Environment[ResumeObservation, ResumeAction, ResumeState]):
    """
    Multi-step resume screening environment. The agent investigates a resume
    by viewing sections, checking references, verifying credentials, and asking
    clarifying questions before submitting a final hiring decision.
    """
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self.data_path = "data/resumes.json"
        self.dataset = load_dataset(self.data_path)

        # Per-episode state
        self._task_type: str = "easy"
        self._current_index: int = 0
        self._sample: Optional[dict] = None
        self._step_count: int = 0
        self._max_steps: int = 8
        self._sections_viewed: List[str] = []
        self._clarifications_asked: int = 0
        self._references_checked: int = 0
        self._verifications_done: int = 0
        self._investigation_score: float = 0.0
        self._done: bool = False

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, task_type: Optional[str] = None, seed: Optional[int] = None, **kwargs) -> ResumeObservation:
        if seed is not None:
            random.seed(seed)

        # Select task type
        if task_type and task_type in self.dataset and len(self.dataset[task_type]) > 0:
            self._task_type = task_type
        else:
            available = [t for t in self.dataset if len(self.dataset[t]) > 0]
            self._task_type = random.choice(available)

        pool = self.dataset[self._task_type]
        self._current_index = random.randint(0, len(pool) - 1)
        self._sample = pool[self._current_index]
        self._max_steps = MAX_STEPS_BY_DIFFICULTY[self._task_type]

        # Reset episode state
        self._step_count = 0
        self._sections_viewed = []
        self._clarifications_asked = 0
        self._references_checked = 0
        self._verifications_done = 0
        self._investigation_score = 0.0
        self._done = False

        # Initial observation: job description + header only
        visible = {"header": self._sample["resume_sections"]["header"]}
        self._sections_viewed.append("header")

        return ResumeObservation(
            task_type=self._task_type,
            phase="initial",
            job_description=self._sample["job_description"],
            visible_sections=visible,
            available_actions=self._get_available_actions(),
            steps_remaining=self._max_steps,
            feedback="Episode started. Review the job description and candidate header. Use investigation actions to gather more information before submitting your decision.",
            done=False,
            reward=0.0,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action: ResumeAction) -> ResumeObservation:
        if self._done or self._sample is None:
            return self._terminal_obs(reward=0.0, feedback="Episode already ended.")

        self._step_count += 1
        remaining = self._max_steps - self._step_count

        if action.action_type == "view_section":
            return self._handle_view_section(action, remaining)
        elif action.action_type == "ask_clarification":
            return self._handle_ask_clarification(action, remaining)
        elif action.action_type == "check_reference":
            return self._handle_check_reference(action, remaining)
        elif action.action_type == "verify_credential":
            return self._handle_verify_credential(action, remaining)
        elif action.action_type == "submit_decision":
            return self._handle_submit_decision(action)
        else:
            return self._obs(reward=0.0, remaining=remaining, feedback="Unknown action type.")

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------
    def _handle_view_section(self, action: ResumeAction, remaining: int) -> ResumeObservation:
        section = (action.section or "").lower().strip()
        sections = self._sample["resume_sections"]

        if section not in VALID_SECTIONS:
            return self._obs(reward=0.0, remaining=remaining,
                             feedback=f"Invalid section '{section}'. Choose from: {', '.join(VALID_SECTIONS)}")

        if section in self._sections_viewed:
            return self._obs(reward=0.0, remaining=remaining,
                             feedback=f"Section '{section}' already viewed. No new information.")

        self._sections_viewed.append(section)
        reward = 0.03 if section in HIGH_VALUE_SECTIONS else 0.01
        self._investigation_score += reward

        if remaining <= 0:
            return self._auto_timeout()

        return self._obs(reward=reward, remaining=remaining,
                         feedback=f"Revealed section: {section}")

    def _handle_ask_clarification(self, action: ResumeAction, remaining: int) -> ResumeObservation:
        self._clarifications_asked += 1
        question = (action.question or "").lower().strip()
        answers = self._sample.get("clarification_answers", {})

        # Fuzzy match: find the answer whose key has the most overlap with the question
        best_key, best_score = None, 0
        for key in answers:
            key_words = set(key.replace("_", " ").lower().split())
            question_words = set(question.split())
            overlap = len(key_words & question_words)
            if overlap > best_score:
                best_score = overlap
                best_key = key

        if best_key and best_score > 0:
            response_text = answers[best_key]
            reward = 0.03
        else:
            response_text = "Thank you for your question. I don't have specific information to address that."
            reward = 0.01

        self._investigation_score += reward

        if remaining <= 0:
            return self._auto_timeout()

        return ResumeObservation(
            task_type=self._task_type,
            phase="investigation",
            job_description=self._sample["job_description"],
            visible_sections=self._get_visible_sections(),
            available_actions=self._get_available_actions(),
            clarification_response=response_text,
            steps_remaining=remaining,
            feedback=f"Clarification answer received.",
            done=False,
            reward=reward,
        )

    def _handle_check_reference(self, action: ResumeAction, remaining: int) -> ResumeObservation:
        self._references_checked += 1
        ref_id = (action.reference_id or "ref1").lower().strip()
        refs = self._sample.get("reference_check_results", {})

        if ref_id in refs:
            ref_data = refs[ref_id]
            response_text = f"{ref_data['name']}: {ref_data['response']}"
        else:
            available_refs = list(refs.keys())
            if available_refs:
                ref_data = refs[available_refs[0]]
                response_text = f"{ref_data['name']}: {ref_data['response']}"
            else:
                response_text = "No references available for this candidate."

        # Higher reward if resume is actually fraudulent (reference check is more valuable)
        is_fraud = self._sample["ground_truth"].get("is_fraud", False)
        reward = 0.05 if is_fraud else 0.02
        self._investigation_score += reward

        if remaining <= 0:
            return self._auto_timeout()

        return ResumeObservation(
            task_type=self._task_type,
            phase="investigation",
            job_description=self._sample["job_description"],
            visible_sections=self._get_visible_sections(),
            available_actions=self._get_available_actions(),
            reference_response=response_text,
            steps_remaining=remaining,
            feedback=f"Reference check completed.",
            done=False,
            reward=reward,
        )

    def _handle_verify_credential(self, action: ResumeAction, remaining: int) -> ResumeObservation:
        self._verifications_done += 1
        verification = self._sample.get("verification_data", {})

        parts = []
        for key, value in verification.items():
            label = key.replace("_", " ").title()
            status = "VERIFIED" if value else "FAILED"
            parts.append(f"{label}: {status}")
        result_text = "; ".join(parts) if parts else "No verification data available."

        # Higher reward if verification reveals fraud
        has_failed = any(not v for v in verification.values())
        reward = 0.05 if has_failed else 0.02
        self._investigation_score += reward

        if remaining <= 0:
            return self._auto_timeout()

        return ResumeObservation(
            task_type=self._task_type,
            phase="investigation",
            job_description=self._sample["job_description"],
            visible_sections=self._get_visible_sections(),
            available_actions=self._get_available_actions(),
            verification_result=result_text,
            steps_remaining=remaining,
            feedback=f"Credential verification completed.",
            done=False,
            reward=reward,
        )

    def _handle_submit_decision(self, action: ResumeAction) -> ResumeObservation:
        self._done = True
        gt = self._sample["ground_truth"]

        decision = (action.decision or "reject").lower()
        fraud_flag = action.fraud_flag if action.fraud_flag is not None else False
        confidence = action.confidence if action.confidence is not None else 0.5
        confidence = max(0.0, min(1.0, confidence))
        fraud_reasoning = action.fraud_reasoning or ""

        reward = 0.0

        # Decision accuracy: +0.35 correct, -0.35 wrong
        if decision == gt["decision"]:
            reward += 0.35
        else:
            reward -= 0.35

        # Fraud detection: +0.25 correct, -0.25 wrong
        if fraud_flag == gt["is_fraud"]:
            reward += 0.25
        else:
            reward -= 0.25

        # Confidence calibration: +0.10 if confident and both correct
        if (decision == gt["decision"] and fraud_flag == gt["is_fraud"]
                and confidence >= 0.7):
            reward += 0.10

        # Investigation thoroughness bonus: +0.10 if agent did real investigation
        investigated = (len(self._sections_viewed) >= 3
                        and (self._references_checked > 0 or self._verifications_done > 0))
        if investigated:
            reward += 0.10

        # Fraud reasoning quality: +0.10 if reasoning mentions actual indicators
        fraud_indicators = gt.get("fraud_indicators", [])
        if fraud_indicators and fraud_reasoning:
            reasoning_lower = fraud_reasoning.lower()
            matched = sum(1 for ind in fraud_indicators if ind.replace("_", " ") in reasoning_lower)
            if matched > 0:
                reward += 0.10

        # Early termination penalty: submit on step 1 without investigation
        if self._step_count == 1 and len(self._sections_viewed) <= 1:
            reward -= 0.15

        # Clamp to [-1.0, 1.0]
        final_reward = max(-1.0, min(1.0, reward))

        return ResumeObservation(
            task_type=self._task_type,
            phase="decision_made",
            job_description=self._sample["job_description"],
            visible_sections=self._get_visible_sections(),
            available_actions=[],
            steps_remaining=0,
            feedback=f"Decision submitted. Episode complete.",
            done=True,
            reward=final_reward,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_visible_sections(self) -> Dict[str, str]:
        if self._sample is None:
            return {}
        sections = self._sample["resume_sections"]
        return {s: sections[s] for s in self._sections_viewed if s in sections}

    def _get_available_actions(self) -> List[str]:
        actions = ["view_section", "ask_clarification", "check_reference",
                    "verify_credential", "submit_decision"]
        return actions

    def _obs(self, reward: float, remaining: int, feedback: str) -> ResumeObservation:
        return ResumeObservation(
            task_type=self._task_type,
            phase="investigation",
            job_description=self._sample["job_description"],
            visible_sections=self._get_visible_sections(),
            available_actions=self._get_available_actions(),
            steps_remaining=remaining,
            feedback=feedback,
            done=False,
            reward=reward,
        )

    def _terminal_obs(self, reward: float, feedback: str) -> ResumeObservation:
        return ResumeObservation(
            task_type=self._task_type,
            phase="decision_made",
            job_description=self._sample.get("job_description", "") if self._sample else "",
            visible_sections=self._get_visible_sections(),
            available_actions=[],
            steps_remaining=0,
            feedback=feedback,
            done=True,
            reward=reward,
        )

    def _auto_timeout(self) -> ResumeObservation:
        self._done = True
        return ResumeObservation(
            task_type=self._task_type,
            phase="decision_made",
            job_description=self._sample["job_description"],
            visible_sections=self._get_visible_sections(),
            available_actions=[],
            steps_remaining=0,
            feedback="Step budget exhausted without submitting a decision. Episode ended with zero reward.",
            done=True,
            reward=0.0,
        )

    @property
    def state(self) -> ResumeState:
        return ResumeState(
            current_index=self._current_index,
            task_type=self._task_type,
            step_count=self._step_count,
            max_steps=self._max_steps,
            sections_viewed=self._sections_viewed.copy(),
            clarifications_asked=self._clarifications_asked,
            references_checked=self._references_checked,
            verifications_done=self._verifications_done,
            investigation_score=self._investigation_score,
        )
