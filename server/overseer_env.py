"""
OverseerEnvironment — Deliberate Synthesis Layer
=================================================
Day 3 implementation: the Overseer receives specialist_reports[] and must
actively READ each report (read_reports action) before issuing a verdict.

Key concepts
------------
• read_reports            — explicit action to examine one specialist's report;
                            returns enriched detail + small thoroughness reward
• request_reinvestigation — already existed; formalized here with guardrails
• submit_final_decision   — unchanged API, but now penalises blind decisions
                            (decided without reading any reports)

Architecture
------------
OverseerConfig                  — dataclass for overseer constraints & reward params
OVERSEER_DECISION_MATRIX        — consensus map (fraud×skills×timeline) → hint
OVERSEER_ROLE_INSTRUCTIONS      — detailed prompt injected into FleetObservation
get_report_enrichment()         — generates env-level analysis appended on read
compute_read_reward()           — per-step reward for read_reports actions
build_overseer_available_actions() — dynamic action list for overseer steps
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# OverseerConfig
# ---------------------------------------------------------------------------

@dataclass
class OverseerConfig:
    """
    Declarative configuration for the overseer phase.

    Decoupled from SpecialistConfig so overseer behaviour can evolve
    independently without touching the specialist whitelist logic.
    """

    # Full set of action types the overseer may use.
    allowed_actions: List[str] = field(default_factory=lambda: [
        "read_reports",
        "request_reinvestigation",
        "submit_final_decision",
    ])

    # Maximum number of reinvestigations per episode.
    max_reinvestigations: int = 1

    # Reward for each distinct specialist report the overseer explicitly reads.
    reward_per_report_read: float = 0.02

    # Bonus for reading ALL specialist reports before deciding.
    all_reports_read_bonus: float = 0.03

    # Human-readable focus summary (shown in feedback when overseer acts).
    focus_description: str = (
        "Synthesise specialist reports and issue the final hiring verdict. "
        "Read each specialist report with read_reports, optionally "
        "request_reinvestigation ONCE, then submit_final_decision."
    )


# ---------------------------------------------------------------------------
# Consensus decision matrix
# ---------------------------------------------------------------------------

# Maps (fraud_raised, skills_raised, timeline_raised) → suggested decision.
# Exposed so the overseer agent can request it via environment feedback.
OVERSEER_DECISION_MATRIX: Dict[Tuple[bool, bool, bool], str] = {
    (False, False, False): "accept",   # All clear
    (False, False, True):  "reject",   # Timeline issues only
    (False, True,  False): "reject",   # Skills mismatch only
    (False, True,  True):  "reject",   # Skills + timeline
    (True,  False, False): "reject",   # Fraud only — always reject
    (True,  False, True):  "reject",   # Fraud + timeline
    (True,  True,  False): "reject",   # Fraud + skills
    (True,  True,  True):  "reject",   # Everything flagged
}


# ---------------------------------------------------------------------------
# Report enrichment
# ---------------------------------------------------------------------------

def get_report_enrichment(report, resume_sample: dict) -> str:
    """
    Generate environment-level analytical context for one specialist report.

    Called when the overseer uses read_reports for a specific target.
    Returns a formatted string that includes:
      • The specialist's own findings and confidence
      • Environment-sourced ground-truth hints (fraud indicators, skills gap, gaps)

    The ground-truth hints help the overseer agent make a calibrated final
    decision even when a specialist's findings were vague.
    """
    gt = resume_sample.get("ground_truth", {})
    role = report.specialist_role

    if role == "fraud_specialist":
        fraud_indicators = gt.get("fraud_indicators", [])
        if fraud_indicators:
            env_note = f"Known fraud signals: {', '.join(fraud_indicators)}."
        else:
            env_note = "No fraud signals on record for this resume."

        return (
            f"[FRAUD SPECIALIST — FULL READ]\n"
            f"Findings      : {report.findings}\n"
            f"Issues Flagged: {'YES' if report.has_issues else 'NO'}\n"
            f"Confidence    : {report.confidence:.2f}\n"
            f"Env Context   : {env_note}"
        )

    elif role == "skills_specialist":
        required = resume_sample.get("required_skills", [])
        req_str = ", ".join(required) if required else "Not specified"
        expected_decision = gt.get("decision", "unknown")

        return (
            f"[SKILLS SPECIALIST — FULL READ]\n"
            f"Findings      : {report.findings}\n"
            f"Issues Flagged: {'YES' if report.has_issues else 'NO'}\n"
            f"Confidence    : {report.confidence:.2f}\n"
            f"Env Context   : Required skills: {req_str}. "
            f"Ground-truth decision: {expected_decision}."
        )

    elif role == "timeline_specialist":
        gaps = gt.get("employment_gaps", [])
        gap_str = ", ".join(gaps) if gaps else "None detected"
        is_fraud = gt.get("is_fraud", False)

        return (
            f"[TIMELINE SPECIALIST — FULL READ]\n"
            f"Findings      : {report.findings}\n"
            f"Issues Flagged: {'YES' if report.has_issues else 'NO'}\n"
            f"Confidence    : {report.confidence:.2f}\n"
            f"Env Context   : Employment gaps: {gap_str}. "
            f"Fraud case: {'YES' if is_fraud else 'NO'}."
        )

    # Generic fallback
    return (
        f"[{role.upper()} — FULL READ]\n"
        f"Findings      : {report.findings}\n"
        f"Issues Flagged: {'YES' if report.has_issues else 'NO'}\n"
        f"Confidence    : {report.confidence:.2f}"
    )


# ---------------------------------------------------------------------------
# Per-step read reward
# ---------------------------------------------------------------------------

def compute_read_reward(
    reports_read: List[str],
    all_report_roles: List[str],
    config: OverseerConfig,
) -> float:
    """
    Return the incremental reward for the LATEST read_reports step.

    • Each unique report read earns reward_per_report_read (0.02).
    • Reading ALL reports earns an additional all_reports_read_bonus (0.03).
      The bonus is awarded only once — on the step that completes the set.
    """
    per_report = config.reward_per_report_read
    all_read = set(reports_read) >= set(all_report_roles) and len(all_report_roles) > 0
    bonus = config.all_reports_read_bonus if all_read else 0.0
    # Return per_report for the current read + bonus if just completed the set
    return round(per_report + bonus, 4)


# ---------------------------------------------------------------------------
# Dynamic available-actions builder
# ---------------------------------------------------------------------------

def build_overseer_available_actions(
    reports_read: List[str],
    all_report_roles: List[str],
    reinvestigation_used: bool,
    config: OverseerConfig,
) -> List[str]:
    """
    Build the overseer's executable action list given current episode state.

    • read_reports              — available while any report remains unread
    • request_reinvestigation   — available until used once
    • submit_final_decision     — always available (overseer must be able to decide)
    """
    actions: List[str] = []

    unread = set(all_report_roles) - set(reports_read)
    if unread:
        actions.append("read_reports")

    if not reinvestigation_used:
        actions.append("request_reinvestigation")

    # Always last so agent sees it as the terminal option
    actions.append("submit_final_decision")

    return actions


# ---------------------------------------------------------------------------
# Role instructions injected into FleetObservation.role_instructions
# ---------------------------------------------------------------------------

OVERSEER_ROLE_INSTRUCTIONS = (
    "You are the OVERSEER. Three specialist agents have investigated the resume.\n"
    "Your task: synthesise their reports and issue the final hiring verdict.\n\n"
    "RECOMMENDED WORKFLOW:\n"
    "  1. read_reports — Explicitly read each specialist's report for full detail.\n"
    "     Set report_target to: 'fraud_specialist', 'skills_specialist', or\n"
    "     'timeline_specialist'. Reading all three earns a thoroughness bonus.\n"
    "  2. request_reinvestigation — Use ONCE if reports conflict or are incomplete.\n"
    "     Set reinvestigation_target and reinvestigation_reason.\n"
    "  3. submit_final_decision — Issue your verdict:\n"
    "       decision        : 'accept' or 'reject'\n"
    "       fraud_flag      : true / false\n"
    "       confidence      : 0.0 – 1.0\n"
    "       fraud_reasoning : required when fraud_flag is true\n\n"
    "CONSTRAINTS:\n"
    "  • You CANNOT view resume sections directly.\n"
    "  • You CANNOT use check_reference or verify_credential.\n"
    "  • You have a limited step budget — plan reads carefully.\n"
    "  • Deciding without reading any report reduces your synthesis bonus."
)


# ---------------------------------------------------------------------------
# Consensus hint (optional utility for feedback messages)
# ---------------------------------------------------------------------------

def get_consensus_hint(specialist_reports: list) -> str:
    """
    Derive a consensus signal from submitted specialist reports and return
    a human-readable hint string.  Used in read_reports feedback.
    """
    if not specialist_reports:
        return "No specialist reports submitted yet."

    fraud_raised = any(
        r.has_issues for r in specialist_reports
        if r.specialist_role == "fraud_specialist"
    )
    skills_raised = any(
        r.has_issues for r in specialist_reports
        if r.specialist_role == "skills_specialist"
    )
    timeline_raised = any(
        r.has_issues for r in specialist_reports
        if r.specialist_role == "timeline_specialist"
    )

    key = (fraud_raised, skills_raised, timeline_raised)
    suggested = OVERSEER_DECISION_MATRIX.get(key, "reject")

    flags = []
    if fraud_raised:
        flags.append("fraud")
    if skills_raised:
        flags.append("skills mismatch")
    if timeline_raised:
        flags.append("timeline issues")

    flag_str = ", ".join(flags) if flags else "none"
    return (
        f"Consensus flags: [{flag_str}]. "
        f"Suggested decision based on specialist consensus: {suggested.upper()}."
    )
