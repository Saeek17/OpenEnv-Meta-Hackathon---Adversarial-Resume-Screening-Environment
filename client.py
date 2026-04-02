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

    def _parse_result(self, response: Dict[str, Any]) -> ResumeObservation:
        """
        Converts the API response into a ResumeObservation.
        Handles both nested and flattened observation data.
        """
        data = response.copy()
        
        # Flatten nested observation if present
        if "observation" in data and isinstance(data["observation"], dict):
            obs_data = data.pop("observation")
            data.update(obs_data)
            
        # Ensure reward and done have default values if missing or None
        if data.get("reward") is None:
            data["reward"] = 0.0
        if data.get("done") is None:
            data["done"] = False
            
        return ResumeObservation(**data)

    def _parse_state(self, response: Dict[str, Any]) -> ResumeState:
        """
        Converts the API response into a ResumeState object.
        """
        return ResumeState(**response)

    def _parse_reset(self, response: Dict[str, Any]) -> ResumeObservation:
        """
        Converts the API response from /reset into a ResumeObservation.
        """
        return self._parse_result(response)
