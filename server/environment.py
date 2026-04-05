import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

from openenv.core.env_server import Environment
from models import ResumeObservation, ResumeAction, ResumeState

def load_dataset(path: str) -> Dict[str, list]:
    """
    Loads and validates the resume dataset from a JSON file.
    """
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {json_path}")
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Validate keys
    required_keys = ["easy", "medium", "hard"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Dataset missing required category: {key}")
            
    return data

class ResumeScreeningEnvironment(Environment[ResumeObservation, ResumeAction, ResumeState]):
    """
    Core environment logic for the Adversarial Resume Screening task.
    """
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self.data_path = "data/resumes.json"
        self.dataset = load_dataset(self.data_path)
        
        # Initialize state variables
        self._current_task_type: str = "easy"
        self._current_index: int = 0
        self._current_sample = None
        self._step_count: int = 0

    def reset(self) -> ResumeObservation:
        """
        Resets the environment by randomly selecting a task type and a resume.
        Ensures a clean state on every reset.
        """
        # 1. Randomly select task_type from dataset
        available_task_types = [t for t in self.dataset.keys() if len(self.dataset[t]) > 0]
        if not available_task_types:
            raise ValueError("Dataset is empty across all task types.")
            
        self._current_task_type = random.choice(available_task_types)
        
        # 2. Randomly pick one entry (sample)
        self._current_index = random.randint(0, len(self.dataset[self._current_task_type]) - 1)
        self._current_sample = self.dataset[self._current_task_type][self._current_index]
        
        # 3. Reset step_count
        self._step_count = 0
        
        # 4. Return ResumeObservation with required fields
        return ResumeObservation(
            resume_text=self._current_sample["resume"],
            job_description=self._current_sample["job"],
            task_type=self._current_task_type
        )

    def step(self, action: ResumeAction) -> ResumeObservation:
        """
        Evaluates the agent's action and ends the episode.
        Returns ONLY the ResumeObservation object (with reward/done populated).
        """
        if self._current_sample is None:
            return self._get_empty_observation(done=True)

        # 1. Validation Logic
        is_invalid = (
            action.decision not in ["accept", "reject"] or
            not (0.0 <= action.confidence <= 1.0) or
            not isinstance(action.fraud_flag, bool)
        )
        
        if is_invalid:
            return self._get_empty_observation(done=True, reward=0.0)

        self._step_count += 1
        
        # 2. Extract ground truth
        expected_decision = self._current_sample.get("expected_decision", "reject")
        is_fraud_truth = self._current_sample.get("is_fraud", False)

        # 3. Reward Calculation (STRICT)
        reward = 0.0

        # decision correct: +0.5
        if action.decision == expected_decision:
            reward += 0.5
        else:
            # wrong decision: -0.3
            reward -= 0.3

        # fraud_flag correct: +0.3
        if action.fraud_flag == is_fraud_truth:
            reward += 0.3
        else:
            # wrong fraud: -0.3
            reward -= 0.3

        # CONFIDENCE REWARD LOGIC: ONLY +0.2 if Decision, Fraud, and Confidence >= 0.7 are all true
        if (action.decision == expected_decision and 
            action.fraud_flag == is_fraud_truth and 
            action.confidence >= 0.7):
            reward += 0.2

        # 4. Final Reward Normalization (max 0.0, min 1.0)
        final_reward = float(max(0.0, min(1.0, reward)))

        # Return the observation object itself (OpenEnv wrapper extracts reward/done from it)
        return self._get_empty_observation(done=True, reward=final_reward)

    def _get_empty_observation(self, done: bool = False, reward: float = 0.0) -> ResumeObservation:
        """Helper to return an observation with empty fields after the episode ends."""
        return ResumeObservation(
            resume_text="",
            job_description="",
            task_type=self._current_task_type if self._current_task_type in ["easy", "medium", "hard"] else "easy",
            done=done,
            reward=reward
        )

    @property
    def state(self) -> ResumeState:
        """
        Returns the current internal state with all required fields.
        """
        return ResumeState(
            current_index=self._current_index,
            task_type=self._current_task_type,
            step_count=self._step_count
        )
