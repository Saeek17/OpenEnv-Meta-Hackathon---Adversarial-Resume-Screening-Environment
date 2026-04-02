from openenv.core.env_server import Action, Observation, State
from pydantic import Field
from typing import Literal, Optional

class ResumeObservation(Observation):
    """
    Observation provided to the agent for each resume screening task.
    """
    resume_text: str = Field(..., description="The full text of the resume.")
    job_description: str = Field(..., description="The job description to evaluate the resume against.")
    task_type: Literal["easy", "medium", "hard"] = Field(..., description="The difficulty level of the current task.")
    done: bool = Field(False, description="Whether the episode is finished.")
    reward: Optional[float] = Field(0.0, description="The reward for the last action.")

class ResumeAction(Action):
    """
    Action taken by the agent after evaluating a resume.
    """
    decision: Literal["accept", "reject"] = Field(..., description="The hiring decision for the candidate.")
    fraud_flag: bool = Field(..., description="Whether the resume is detected as fraudulent.")
    confidence: float = Field(..., description="The agent's confidence in its decision (0.0 to 1.0).")

class ResumeState(State):
    """
    Internal state of the environment, used for tracking and evaluation.
    """
    current_index: int = Field(..., description="The index of the current resume in the dataset.")
    task_type: str = Field(..., description="The difficulty level of the current task.")
    step_count: int = Field(..., description="The number of actions taken in the current episode.")

# Rebuild models for Pydantic v2 compatibility
ResumeObservation.model_rebuild()
ResumeAction.model_rebuild()
ResumeState.model_rebuild()
