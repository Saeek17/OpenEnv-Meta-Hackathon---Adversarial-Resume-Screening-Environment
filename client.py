from typing import Tuple, Dict, Any

from openenv.core.env_client import EnvClient
from models import ResumeObservation, ResumeAction, ResumeState

class ResumeEnv(EnvClient[ResumeObservation, ResumeAction, ResumeState]):
    """
    Client-side interface for interacting with the Resume Screening Environment.
    """

    def _step_payload(self, action: ResumeAction) -> Dict[str, Any]:
        """
        Converts ResumeAction to a dictionary for the API payload.
        """
        return action.model_dump()

    def _parse_result(self, response: Dict[str, Any]) -> Tuple[ResumeObservation, float, bool, Dict[str, Any]]:
        """
        Converts the API response into a Tuple containing ResumeObservation, reward, done, and info.
        """
        observation = ResumeObservation(**response["observation"])
        reward = float(response["reward"])
        done = bool(response["done"])
        info = response.get("info", {})
        
        return observation, reward, done, info

    def _parse_state(self, response: Dict[str, Any]) -> ResumeState:
        """
        Converts the API response into a ResumeState object.
        """
        return ResumeState(**response)

    def _parse_reset(self, response: Dict[str, Any]) -> ResumeObservation:
        """
        Converts the API response from /reset into a ResumeObservation.
        """
        return ResumeObservation(**response)
