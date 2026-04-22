from openenv.core.env_server import Action, Observation, State
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional


# ============================================================
# Original Resume Screening Models
# ============================================================

class ResumeObservation(Observation):
    """
    Observation provided to the agent at each step of a resume screening episode.
    The agent starts with the job description and header, then reveals more info
    through investigation actions before making a final decision.
    """
    task_type: Literal["easy", "medium", "hard"] = Field(
        ..., description="Difficulty level of the current task."
    )
    phase: Literal["initial", "investigation", "decision_made"] = Field(
        "initial", description="Current phase of the episode."
    )
    job_description: str = Field(
        "", description="The job description to evaluate the resume against."
    )
    visible_sections: Dict[str, str] = Field(
        default_factory=dict,
        description="Resume sections revealed so far (section_name -> content)."
    )
    available_actions: List[str] = Field(
        default_factory=list,
        description="Action types the agent can take next."
    )
    clarification_response: Optional[str] = Field(
        None, description="Response to the last clarification question."
    )
    reference_response: Optional[str] = Field(
        None, description="Result of the last reference check."
    )
    verification_result: Optional[str] = Field(
        None, description="Result of the last credential verification."
    )
    steps_remaining: int = Field(
        0, description="Number of steps left in the episode budget."
    )
    feedback: Optional[str] = Field(
        None, description="Environment feedback or warnings."
    )


class ResumeAction(Action):
    """
    Multi-type action for resume investigation and decision-making.
    The agent chooses an action_type and provides the relevant fields.
    """
    action_type: Literal[
        "view_section",
        "ask_clarification",
        "check_reference",
        "verify_credential",
        "submit_decision"
    ] = Field(..., description="Type of action to take.")

    # Session routing — set by inference client to prevent concurrent-session collisions
    episode_id: Optional[str] = Field(
        None, description="Episode ID for session routing (set by client, ignored in scoring)."
    )

    # For view_section
    section: Optional[str] = Field(
        None,
        description="Section to view: header, summary, experience, education, skills, projects, references"
    )

    # For ask_clarification
    question: Optional[str] = Field(
        None, description="Clarification question to ask the candidate."
    )

    # For check_reference
    reference_id: Optional[str] = Field(
        None, description="Reference to contact: ref1 or ref2."
    )

    # For submit_decision
    decision: Optional[Literal["accept", "reject"]] = Field(
        None, description="Final hiring decision."
    )
    fraud_flag: Optional[bool] = Field(
        None, description="Whether the resume is flagged as fraudulent."
    )
    confidence: Optional[float] = Field(
        None, description="Confidence in the decision (0.0 to 1.0)."
    )
    fraud_reasoning: Optional[str] = Field(
        None, description="Explanation if fraud is suspected."
    )


class ResumeState(State):
    """
    Internal state of the environment for tracking episode progress.
    """
    current_index: int = Field(0, description="Index of the current resume.")
    task_type: str = Field("easy", description="Difficulty level.")
    max_steps: int = Field(8, description="Maximum steps allowed.")
    sections_viewed: List[str] = Field(
        default_factory=list, description="Sections the agent has viewed."
    )
    clarifications_asked: int = Field(
        0, description="Number of clarification questions asked."
    )
    references_checked: int = Field(
        0, description="Number of references contacted."
    )
    verifications_done: int = Field(
        0, description="Number of credential verifications performed."
    )
    investigation_score: float = Field(
        0.0, description="Running partial reward from investigation steps."
    )


# ============================================================
# Fleet AI Models — Multi-Agent Hiring Fleet
# ============================================================

class SpecialistReport(BaseModel):
    """Report submitted by a specialist agent after completing its investigation phase."""
    specialist_role: str = Field(
        ..., description="Role of the specialist: fraud_specialist, skills_specialist, or timeline_specialist."
    )
    findings: str = Field(
        ..., description="Detailed findings from the specialist's investigation."
    )
    has_issues: bool = Field(
        False, description="Whether the specialist flagged any red flags or concerns."
    )
    confidence: float = Field(
        0.5, description="Specialist's confidence in their findings (0.0 to 1.0)."
    )


class FleetObservation(Observation):
    """
    Observation for the multi-agent Fleet environment.

    Each step is attributed to the currently active specialist/overseer.
    The observation includes prior specialist reports so each subsequent agent
    (and the final overseer) can see what others found.
    """
    task_type: Literal["easy", "medium", "hard"] = Field(
        ..., description="Difficulty level of the current task."
    )
    current_phase: Literal[
        "fraud_specialist",
        "skills_specialist",
        "timeline_specialist",
        "overseer",
        "complete"
    ] = Field(..., description="Which agent is currently active.")
    role_instructions: str = Field(
        "", description="Instructions telling the current agent their role and goal."
    )
    job_description: str = Field(
        "", description="The job description to evaluate against."
    )
    visible_sections: Dict[str, str] = Field(
        default_factory=dict,
        description="Resume sections visible to the current specialist."
    )
    specialist_reports: List[SpecialistReport] = Field(
        default_factory=list,
        description="Reports submitted by completed specialist agents."
    )
    available_actions: List[str] = Field(
        default_factory=list,
        description="Action types available to the current agent."
    )
    clarification_response: Optional[str] = Field(
        None, description="Response to the last clarification question."
    )
    reference_response: Optional[str] = Field(
        None, description="Result of the last reference check."
    )
    verification_result: Optional[str] = Field(
        None, description="Result of the last credential verification."
    )
    steps_remaining: int = Field(
        0, description="Steps remaining in the current phase budget."
    )
    total_steps_remaining: int = Field(
        0, description="Total steps remaining across all phases."
    )
    violations_count: int = Field(
        0, description="Number of out-of-role action violations this episode. Each costs 0.05 from final reward."
    )
    # Day 3 — Overseer report-reading tracking
    reports_read: List[str] = Field(
        default_factory=list,
        description=(
            "Specialist roles whose reports the overseer has explicitly read via read_reports. "
            "Reading all three before deciding earns a thoroughness bonus."
        )
    )
    read_report_details: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Enriched report text keyed by specialist role. "
            "Populated progressively as the overseer uses read_reports."
        )
    )
    feedback: Optional[str] = Field(
        None, description="Environment feedback for the current agent."
    )


class FleetAction(Action):
    """
    Action type for the multi-agent fleet environment.

    Specialists use: view_section, ask_clarification, check_reference,
                     verify_credential, submit_specialist_report.
    Overseer uses:   request_reinvestigation, submit_final_decision.
    """
    action_type: Literal[
        "view_section",
        "ask_clarification",
        "check_reference",
        "verify_credential",
        "submit_specialist_report",
        "read_reports",
        "request_reinvestigation",
        "submit_final_decision"
    ] = Field(..., description="Type of action to take.")

    # Session routing
    episode_id: Optional[str] = Field(
        None, description="Episode ID for session routing (set by client)."
    )

    # For view_section
    section: Optional[str] = Field(
        None,
        description="Section to view: header, summary, experience, education, skills, projects, references"
    )

    # For ask_clarification
    question: Optional[str] = Field(
        None, description="Clarification question to ask the candidate."
    )

    # For check_reference
    reference_id: Optional[str] = Field(
        None, description="Reference to contact: ref1 or ref2."
    )

    # For read_reports (overseer only — Day 3)
    report_target: Optional[str] = Field(
        None,
        description=(
            "Which specialist report to read in full: "
            "'fraud_specialist', 'skills_specialist', or 'timeline_specialist'."
        )
    )

    # For submit_specialist_report
    findings: Optional[str] = Field(
        None, description="Specialist's findings summary."
    )
    has_issues: Optional[bool] = Field(
        None, description="Whether the specialist found red flags."
    )
    specialist_confidence: Optional[float] = Field(
        None, description="Specialist's confidence in their findings (0.0 to 1.0)."
    )

    # For request_reinvestigation (overseer only)
    reinvestigation_target: Optional[str] = Field(
        None, description="Which specialist phase to re-examine: fraud_specialist, skills_specialist, or timeline_specialist."
    )
    reinvestigation_reason: Optional[str] = Field(
        None, description="Why the overseer wants more investigation."
    )

    # For submit_final_decision (overseer only)
    decision: Optional[Literal["accept", "reject"]] = Field(
        None, description="Final hiring decision."
    )
    fraud_flag: Optional[bool] = Field(
        None, description="Whether the resume is flagged as fraudulent."
    )
    confidence: Optional[float] = Field(
        None, description="Confidence in the final decision (0.0 to 1.0)."
    )
    fraud_reasoning: Optional[str] = Field(
        None, description="Explanation if fraud is suspected."
    )


class FleetState(State):
    """Internal state for the multi-agent fleet environment."""
    current_index: int = Field(0, description="Index of the current resume.")
    task_type: str = Field("easy", description="Difficulty level.")
    phase_idx: int = Field(0, description="Current phase index (0=fraud, 1=skills, 2=timeline, 3=overseer).")
    phase_steps_used: int = Field(0, description="Steps used in current phase.")
    total_steps_used: int = Field(0, description="Total steps used across all phases.")
    max_total_steps: int = Field(9, description="Maximum steps allowed in total.")
    sections_viewed: List[str] = Field(default_factory=list)
    specialist_reports: List[dict] = Field(default_factory=list)
    references_checked: int = Field(0)
    verifications_done: int = Field(0)
    clarifications_asked: int = Field(0)
    reinvestigation_used: bool = Field(False, description="Whether overseer requested reinvestigation.")
    violations_count: int = Field(0, description="Out-of-role action violations this episode.")
    # Day 3 — Overseer report reading
    reports_read: List[str] = Field(
        default_factory=list,
        description="Specialist roles whose reports the overseer has explicitly read."
    )
    read_report_details: Dict[str, str] = Field(
        default_factory=dict,
        description="Enriched detail text per specialist role, populated by read_reports action."
    )
    done: bool = Field(False)


# Rebuild models for Pydantic v2 compatibility
ResumeObservation.model_rebuild()
ResumeAction.model_rebuild()
ResumeState.model_rebuild()
FleetObservation.model_rebuild()
FleetAction.model_rebuild()
FleetState.model_rebuild()
