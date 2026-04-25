"""
Fleet Resume Screening Environment
====================================
A multi-agent hiring fleet where three specialist AI agents (Fraud Specialist,
Skills Specialist, Timeline Specialist) sequentially investigate a resume and
submit reports. An Overseer agent then reads all three reports and makes the
final hiring decision.

This environment implements the "Fleet AI — Scalable Oversight" sub-theme of
the Meta OpenEnv Hackathon Round 2.

Phase structure:
  Phase 0  fraud_specialist   — checks references, verifies credentials, hunts fraud signals
  Phase 1  skills_specialist  — reads experience/education/skills, asks technical clarifications
  Phase 2  timeline_specialist— checks chronological consistency across header/summary/experience
  Phase 3  overseer           — reads all three reports, may request one reinvestigation, decides

Episode budgets (Day 3 — overseer budget expanded for read_reports):
  easy   →  8 total steps  (2 / 2 / 2 / 2 per phase)
  medium → 11 total steps  (2 / 3 / 3 / 3 per phase)
  hard   → 15 total steps  (3 / 4 / 4 / 4 per phase)
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openenv.core.env_server import Environment
from models import (
    FleetObservation, FleetAction, FleetState, SpecialistReport
)
try:
    from server.specialist_env import (
        SPECIALIST_CONFIGS, SpecialistActionValidator, compute_violation_penalty,
    )
    from server.overseer_env import (
        OverseerConfig, OVERSEER_ROLE_INSTRUCTIONS,
        get_report_enrichment, compute_read_reward,
        build_overseer_available_actions, get_consensus_hint,
    )
except ModuleNotFoundError:
    from specialist_env import (   # relative import when run inside server/
        SPECIALIST_CONFIGS, SpecialistActionValidator, compute_violation_penalty,
    )
    from overseer_env import (
        OverseerConfig, OVERSEER_ROLE_INSTRUCTIONS,
        get_report_enrichment, compute_read_reward,
        build_overseer_available_actions, get_consensus_hint,
    )


VALID_SECTIONS = ["header", "summary", "experience", "education", "skills", "projects", "references"]

# Which sections each specialist is primarily responsible for
PHASE_SECTIONS = {
    "fraud_specialist":   ["header", "references"],
    "skills_specialist":  ["experience", "education", "skills", "projects"],
    "timeline_specialist": ["header", "summary", "experience"],
}

# Phase ordering
PHASES = ["fraud_specialist", "skills_specialist", "timeline_specialist", "overseer"]

# Per-phase step budgets by difficulty
# Each specialist needs at least 2 steps (1 investigation + 1 report)
# Overseer budget expanded in Day 3: read_reports consumes 1 step per report.
#   easy   overseer budget=2  → read 1 report  + decide   (or skip reads)
#   medium overseer budget=3  → read 2 reports + decide
#   hard   overseer budget=4  → read all 3     + decide   (full synthesis)
PHASE_BUDGETS: Dict[str, Dict[str, int]] = {
    "easy":   {"fraud_specialist": 2, "skills_specialist": 2, "timeline_specialist": 2, "overseer": 2},
    "medium": {"fraud_specialist": 2, "skills_specialist": 3, "timeline_specialist": 3, "overseer": 3},
    "hard":   {"fraud_specialist": 3, "skills_specialist": 4, "timeline_specialist": 4, "overseer": 4},
}
# Total steps: easy=8, medium=11, hard=15

# Role instructions delivered to each agent in the observation
ROLE_INSTRUCTIONS: Dict[str, str] = {
    "fraud_specialist": (
        "You are the FRAUD SPECIALIST. Your job is to detect fraudulent or exaggerated claims. "
        "Focus on: checking references (check_reference), verifying credentials (verify_credential), "
        "and viewing the 'references' or 'header' sections. "
        "When done, submit your report using submit_specialist_report with your findings, "
        "has_issues (true if you found red flags), and confidence."
    ),
    "skills_specialist": (
        "You are the SKILLS SPECIALIST. Your job is to assess whether the candidate's skills match "
        "the job requirements. Focus on: viewing 'experience', 'education', 'skills', 'projects' sections, "
        "and asking technical clarification questions (ask_clarification). "
        "When done, submit your report using submit_specialist_report with your findings, "
        "has_issues (true if skills are mismatched), and confidence."
    ),
    "timeline_specialist": (
        "You are the TIMELINE SPECIALIST. Your job is to check for chronological consistency "
        "and employment gaps. Focus on: viewing 'header', 'summary', 'experience' sections, "
        "and asking clarifications about career gaps (ask_clarification). "
        "When done, submit your report using submit_specialist_report with your findings, "
        "has_issues (true if you found timeline inconsistencies or gaps), and confidence."
    ),
    "overseer": (
        "You are the OVERSEER. You have received reports from three specialist agents. "
        "Review the specialist_reports carefully. You may use request_reinvestigation ONCE "
        "if you need more information from a specific specialist. "
        "Then submit your final decision using submit_final_decision with: "
        "decision (accept/reject), fraud_flag (true/false), confidence (0.0-1.0), "
        "and fraud_reasoning (if fraud detected)."
    ),
}

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


class FleetResumeEnvironment(Environment[FleetObservation, FleetAction, FleetState]):
    """
    Multi-agent fleet environment for resume screening.

    Three specialist agents investigate in sequence, then an overseer synthesises
    their reports and makes a final hiring/fraud decision. The environment manages
    all phase transitions, step budgets, and reward calculations internally.

    NOTE: The OpenEnv HTTP server creates a new instance per request.
    Class-level _episode_store persists episode data across instances.
    """
    SUPPORTS_CONCURRENT_SESSIONS = True

    _episode_store: Dict[str, dict] = {}
    _default_session: Optional[str] = None
    _dataset_cache: Optional[Dict[str, list]] = None

    def __init__(self):
        super().__init__()
        self.data_path = "data/resumes.json"
        if FleetResumeEnvironment._dataset_cache is None:
            FleetResumeEnvironment._dataset_cache = load_dataset(self.data_path)
        self.dataset = FleetResumeEnvironment._dataset_cache

        # Per-episode state
        self._task_type: str = "easy"
        self._current_index: int = 0
        self._sample: Optional[dict] = None
        self._phase_idx: int = 0                    # 0-3
        self._phase_steps_used: int = 0
        self._total_steps_used: int = 0
        self._max_total_steps: int = 7
        self._sections_viewed: List[str] = []           # global (for visibility)
        self._sections_viewed_this_phase: List[str] = []  # per-phase (for reward/dedup)
        self._specialist_reports: List[SpecialistReport] = []
        self._references_checked: int = 0
        self._verifications_done: int = 0
        self._clarifications_asked: int = 0
        self._reinvestigation_used: bool = False
        self._violations_count: int = 0          # Day 2: out-of-role action counter
        self._done: bool = False
        # Day 3: Overseer report-reading state
        self._reports_read: List[str] = []
        self._read_report_details: Dict[str, str] = {}
        # Per-step response caches (instance, persisted in episode store)
        self._last_clarification: Optional[str] = None
        self._last_reference: Optional[str] = None
        self._last_verification: Optional[str] = None
        self._last_feedback: str = ""

    # ------------------------------------------------------------------
    # State persistence (class-level, survives across HTTP instances)
    # ------------------------------------------------------------------
    def _save_state(self) -> None:
        key = FleetResumeEnvironment._default_session
        if key is None:
            return
        FleetResumeEnvironment._episode_store[key] = {
            "task_type": self._task_type,
            "current_index": self._current_index,
            "sample": self._sample,
            "phase_idx": self._phase_idx,
            "phase_steps_used": self._phase_steps_used,
            "total_steps_used": self._total_steps_used,
            "max_total_steps": self._max_total_steps,
            "sections_viewed": list(self._sections_viewed),
            "sections_viewed_this_phase": list(self._sections_viewed_this_phase),
            "specialist_reports": [r.model_dump() for r in self._specialist_reports],
            "references_checked": self._references_checked,
            "verifications_done": self._verifications_done,
            "clarifications_asked": self._clarifications_asked,
            "reinvestigation_used": self._reinvestigation_used,
            "violations_count": self._violations_count,
            "done": self._done,
            # Day 3
            "reports_read": list(self._reports_read),
            "read_report_details": dict(self._read_report_details),
            "last_clarification": self._last_clarification,
            "last_reference": self._last_reference,
            "last_verification": self._last_verification,
            "last_feedback": self._last_feedback,
        }

    def _restore_state(self) -> bool:
        key = FleetResumeEnvironment._default_session
        if key is None or key not in FleetResumeEnvironment._episode_store:
            return False
        s = FleetResumeEnvironment._episode_store[key]
        self._task_type = s["task_type"]
        self._current_index = s["current_index"]
        self._sample = s["sample"]
        self._phase_idx = s["phase_idx"]
        self._phase_steps_used = s["phase_steps_used"]
        self._total_steps_used = s["total_steps_used"]
        self._max_total_steps = s["max_total_steps"]
        self._sections_viewed = list(s["sections_viewed"])
        self._sections_viewed_this_phase = list(s.get("sections_viewed_this_phase", []))
        self._specialist_reports = [
            SpecialistReport(**r) for r in s["specialist_reports"]
        ]
        self._references_checked = s["references_checked"]
        self._verifications_done = s["verifications_done"]
        self._clarifications_asked = s["clarifications_asked"]
        self._reinvestigation_used = s["reinvestigation_used"]
        self._violations_count = s.get("violations_count", 0)
        self._done = s["done"]
        # Day 3
        self._reports_read = list(s.get("reports_read", []))
        self._read_report_details = dict(s.get("read_report_details", {}))
        self._last_clarification = s.get("last_clarification")
        self._last_reference = s.get("last_reference")
        self._last_verification = s.get("last_verification")
        self._last_feedback = s.get("last_feedback", "")
        return True

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(
        self,
        task_type: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> FleetObservation:
        if seed is not None:
            random.seed(seed)

        episode_id = kwargs.get("episode_id") or "fleet-default"
        FleetResumeEnvironment._default_session = episode_id

        if task_type and task_type in self.dataset:
            self._task_type = task_type
        else:
            self._task_type = random.choice(list(self.dataset.keys()))

        pool = self.dataset[self._task_type]
        self._current_index = random.randint(0, len(pool) - 1)
        self._sample = pool[self._current_index]

        # Episode setup
        self._phase_idx = 0
        self._phase_steps_used = 0
        self._total_steps_used = 0
        self._max_total_steps = sum(PHASE_BUDGETS[self._task_type].values())
        self._sections_viewed = []
        self._sections_viewed_this_phase = []
        self._specialist_reports = []
        self._references_checked = 0
        self._verifications_done = 0
        self._clarifications_asked = 0
        self._reinvestigation_used = False
        self._violations_count = 0
        self._done = False
        # Day 3
        self._reports_read = []
        self._read_report_details = {}
        self._last_clarification = None
        self._last_reference = None
        self._last_verification = None
        self._last_feedback = ""

        self._save_state()

        current_phase = PHASES[0]
        phase_budget = PHASE_BUDGETS[self._task_type][current_phase]
        return FleetObservation(
            task_type=self._task_type,
            current_phase=current_phase,
            role_instructions=ROLE_INSTRUCTIONS[current_phase],
            job_description=self._sample["job_description"],
            visible_sections={},
            specialist_reports=[],
            available_actions=self._get_available_actions(),
            steps_remaining=phase_budget,
            total_steps_remaining=self._max_total_steps,
            reports_read=[],
            read_report_details={},
            feedback=(
                "Fleet episode started. The Fraud Specialist begins investigation. "
                "Read your role_instructions carefully and use your allotted steps wisely."
            ),
            done=False,
            reward=0.0,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action: FleetAction) -> FleetObservation:
        # Concurrent-session fix: route to correct session via embedded episode_id
        if action.episode_id:
            FleetResumeEnvironment._default_session = action.episode_id

        if self._sample is None:
            self._restore_state()

        if self._done or self._sample is None:
            return self._terminal_obs(reward=0.0, feedback="Episode already ended.")

        # Free re-view: viewing an already-seen section in the same phase
        # gives informative feedback but does NOT consume a step.
        _section_already_viewed = (
            action.action_type == "view_section"
            and bool(action.section)
            and action.section.lower().strip() in self._sections_viewed_this_phase
        )
        if not _section_already_viewed:
            self._phase_steps_used += 1
            self._total_steps_used += 1

        current_phase = PHASES[self._phase_idx]
        phase_budget = PHASE_BUDGETS[self._task_type][current_phase]
        phase_steps_remaining = phase_budget - self._phase_steps_used

        # Dispatch to correct handler
        if current_phase in ("fraud_specialist", "skills_specialist", "timeline_specialist"):
            obs = self._handle_specialist_action(action, current_phase, phase_steps_remaining)
        elif current_phase == "overseer":
            obs = self._handle_overseer_action(action, phase_steps_remaining)
        else:
            obs = self._terminal_obs(reward=0.0, feedback="Unknown phase.")

        self._save_state()
        return obs

    # ------------------------------------------------------------------
    # Specialist phase handler  (Day 2: hard whitelist enforcement)
    # ------------------------------------------------------------------
    def _handle_specialist_action(
        self, action: FleetAction, phase: str, phase_steps_remaining: int
    ) -> FleetObservation:
        config = SPECIALIST_CONFIGS[phase]
        validator = SpecialistActionValidator(config)

        # ── HARD VALIDATION ────────────────────────────────────────────
        valid, reason = validator.validate(action)
        if not valid:
            self._violations_count += 1
            self._last_feedback = reason
            total_remaining = self._max_total_steps - self._total_steps_used

            # Budget may have run out even on a violation step
            if phase_steps_remaining <= 0:
                return self._auto_advance_phase(phase, 0.0)

            return FleetObservation(
                task_type=self._task_type,
                current_phase=phase,
                role_instructions=validator.role_instructions(),
                job_description=self._sample["job_description"],
                visible_sections=validator.filter_sections(self._get_visible_sections()),
                specialist_reports=list(self._specialist_reports),
                available_actions=validator.available_actions(
                    self._sections_viewed_this_phase,
                    self._references_checked,
                    self._verifications_done,
                ),
                steps_remaining=phase_steps_remaining,
                total_steps_remaining=total_remaining,
                violations_count=self._violations_count,
                reports_read=list(self._reports_read),
                read_report_details=dict(self._read_report_details),
                feedback=reason,
                done=False,
                reward=0.0,
            )

        # ── VALID ACTION — dispatch ─────────────────────────────────────
        reward = 0.0
        if action.action_type == "submit_specialist_report":
            return self._handle_submit_specialist_report(action, phase)
        elif action.action_type == "view_section":
            reward = self._do_view_section(action.section or "experience")
        elif action.action_type == "ask_clarification":
            reward = self._do_ask_clarification(action.question or "")
        elif action.action_type == "check_reference":
            reward = self._do_check_reference(action.reference_id or "ref1")
        elif action.action_type == "verify_credential":
            reward = self._do_verify_credential()

        # ── AUTO-ADVANCE if phase budget exhausted ──────────────────────
        if phase_steps_remaining <= 0:
            return self._auto_advance_phase(phase, reward)

        total_remaining = self._max_total_steps - self._total_steps_used
        return FleetObservation(
            task_type=self._task_type,
            current_phase=phase,
            role_instructions=validator.role_instructions(),
            job_description=self._sample["job_description"],
            # ── Role-filtered observation (Day 2 key feature) ──────────
            visible_sections=validator.filter_sections(self._get_visible_sections()),
            specialist_reports=list(self._specialist_reports),
            available_actions=validator.available_actions(
                self._sections_viewed_this_phase,
                self._references_checked,
                self._verifications_done,
            ),
            clarification_response=self._last_clarification,
            reference_response=self._last_reference,
            verification_result=self._last_verification,
            steps_remaining=phase_steps_remaining,
            total_steps_remaining=total_remaining,
            violations_count=self._violations_count,
            reports_read=list(self._reports_read),
            read_report_details=dict(self._read_report_details),
            feedback=self._last_feedback,
            done=False,
            reward=reward,
        )

    def _handle_submit_specialist_report(
        self, action: FleetAction, phase: str
    ) -> FleetObservation:
        """Process a specialist submitting their report and advance to next phase."""
        findings = action.findings or "No specific findings."
        has_issues = action.has_issues if action.has_issues is not None else False
        confidence = float(action.specialist_confidence or 0.5)
        confidence = max(0.0, min(1.0, confidence))

        report = SpecialistReport(
            specialist_role=phase,
            findings=findings,
            has_issues=has_issues,
            confidence=confidence,
        )
        self._specialist_reports.append(report)

        # Reward: specialist accuracy
        reward = self._score_specialist_report(phase, report)

        # Advance to next phase
        self._phase_idx += 1
        self._phase_steps_used = 0
        self._sections_viewed_this_phase = []   # reset per-phase section tracking
        self._last_clarification = None
        self._last_reference = None
        self._last_verification = None

        if self._phase_idx >= len(PHASES):
            # Shouldn't happen after specialists, overseer comes next
            self._done = True
            return self._terminal_obs(reward=reward, feedback="All phases complete.")

        next_phase = PHASES[self._phase_idx]
        next_budget = PHASE_BUDGETS[self._task_type][next_phase]
        total_remaining = self._max_total_steps - self._total_steps_used

        # Build validator for the NEXT phase (so available_actions + role_instructions
        # are already correct for the incoming agent)
        if next_phase == "overseer":
            next_available = build_overseer_available_actions(
                self._reports_read,
                [r.specialist_role for r in self._specialist_reports],
                self._reinvestigation_used,
                OverseerConfig(),
            )
            next_role_instructions = OVERSEER_ROLE_INSTRUCTIONS
            next_visible = {}   # overseer never sees raw sections
        else:
            next_validator = SpecialistActionValidator(SPECIALIST_CONFIGS[next_phase])
            next_available = next_validator.available_actions(
                self._sections_viewed_this_phase,   # per-phase — always [] for a fresh phase
                self._references_checked,
                self._verifications_done,
                self._reinvestigation_used,
            )
            next_role_instructions = next_validator.role_instructions()
            next_visible = next_validator.filter_sections(self._get_visible_sections())

        return FleetObservation(
            task_type=self._task_type,
            current_phase=next_phase,
            role_instructions=next_role_instructions,
            job_description=self._sample["job_description"],
            visible_sections=next_visible,
            specialist_reports=list(self._specialist_reports),
            available_actions=next_available,
            steps_remaining=next_budget,
            total_steps_remaining=total_remaining,
            violations_count=self._violations_count,
            reports_read=list(self._reports_read),
            read_report_details=dict(self._read_report_details),
            feedback=(
                f"{phase.replace('_', ' ').title()} report submitted (reward: {reward:.3f}). "
                f"Advancing to {next_phase.replace('_', ' ').title()} phase."
            ),
            done=False,
            reward=reward,
        )

    # ------------------------------------------------------------------
    # Overseer phase handler  (Day 3: read_reports + richer validation)
    # ------------------------------------------------------------------
    def _handle_overseer_action(
        self, action: FleetAction, phase_steps_remaining: int
    ) -> FleetObservation:
        overseer_cfg = OverseerConfig()
        all_report_roles = [r.specialist_role for r in self._specialist_reports]
        total_remaining = self._max_total_steps - self._total_steps_used

        # ── Hard validation: action_type must be in OverseerConfig.allowed_actions ──
        if action.action_type not in overseer_cfg.allowed_actions:
            self._violations_count += 1
            self._last_feedback = (
                f"[VIOLATION] Overseer cannot use '{action.action_type}'. "
                f"Allowed: {overseer_cfg.allowed_actions}. "
                f"This step costs you a step and earns 0 reward."
            )
            if phase_steps_remaining <= 0:
                return self._auto_timeout_overseer()
            return FleetObservation(
                task_type=self._task_type,
                current_phase="overseer",
                role_instructions=OVERSEER_ROLE_INSTRUCTIONS,
                job_description=self._sample["job_description"],
                visible_sections={},          # overseer never sees raw sections
                specialist_reports=list(self._specialist_reports),
                available_actions=build_overseer_available_actions(
                    self._reports_read, all_report_roles,
                    self._reinvestigation_used, overseer_cfg,
                ),
                steps_remaining=phase_steps_remaining,
                total_steps_remaining=total_remaining,
                violations_count=self._violations_count,
                reports_read=list(self._reports_read),
                read_report_details=dict(self._read_report_details),
                feedback=self._last_feedback,
                done=False,
                reward=0.0,
            )

        # ── Dispatch: read_reports ─────────────────────────────────────
        if action.action_type == "read_reports":
            return self._handle_read_reports(action, all_report_roles, overseer_cfg,
                                             phase_steps_remaining, total_remaining)

        # ── Dispatch: request_reinvestigation ──────────────────────────
        elif action.action_type == "request_reinvestigation":
            return self._handle_request_reinvestigation(action, all_report_roles, overseer_cfg,
                                                        phase_steps_remaining, total_remaining)

        # ── Dispatch: submit_final_decision ────────────────────────────
        elif action.action_type == "submit_final_decision":
            return self._handle_submit_final_decision(action)

    # ------------------------------------------------------------------
    # Overseer sub-handlers  (Day 3)
    # ------------------------------------------------------------------

    def _handle_read_reports(
        self,
        action: FleetAction,
        all_report_roles: List[str],
        overseer_cfg: OverseerConfig,
        phase_steps_remaining: int,
        total_remaining: int,
    ) -> FleetObservation:
        """Process the read_reports action — reveal enriched report detail."""
        target = (action.report_target or "").lower().strip()

        # Validate target
        valid_targets = set(all_report_roles)
        if target not in valid_targets:
            # If target is missing/invalid, offer list of unread reports
            unread = list(set(all_report_roles) - set(self._reports_read))
            self._last_feedback = (
                f"Unknown report_target '{target}'. "
                f"Use one of: {list(valid_targets)}. "
                f"Unread so far: {unread}."
            )
            return FleetObservation(
                task_type=self._task_type,
                current_phase="overseer",
                role_instructions=OVERSEER_ROLE_INSTRUCTIONS,
                job_description=self._sample["job_description"],
                visible_sections={},
                specialist_reports=list(self._specialist_reports),
                available_actions=build_overseer_available_actions(
                    self._reports_read, all_report_roles,
                    self._reinvestigation_used, overseer_cfg,
                ),
                steps_remaining=phase_steps_remaining,
                total_steps_remaining=total_remaining,
                violations_count=self._violations_count,
                reports_read=list(self._reports_read),
                read_report_details=dict(self._read_report_details),
                feedback=self._last_feedback,
                done=False,
                reward=0.0,
            )

        # Already read? Warn but don't penalise
        if target in self._reports_read:
            self._last_feedback = (
                f"You already read the {target} report. "
                f"Unread reports: {list(set(all_report_roles) - set(self._reports_read))}."
            )
            if phase_steps_remaining <= 0:
                return self._auto_timeout_overseer()
            return FleetObservation(
                task_type=self._task_type,
                current_phase="overseer",
                role_instructions=OVERSEER_ROLE_INSTRUCTIONS,
                job_description=self._sample["job_description"],
                visible_sections={},
                specialist_reports=list(self._specialist_reports),
                available_actions=build_overseer_available_actions(
                    self._reports_read, all_report_roles,
                    self._reinvestigation_used, overseer_cfg,
                ),
                steps_remaining=phase_steps_remaining,
                total_steps_remaining=total_remaining,
                violations_count=self._violations_count,
                reports_read=list(self._reports_read),
                read_report_details=dict(self._read_report_details),
                feedback=self._last_feedback,
                done=False,
                reward=0.0,
            )

        # First read of this report — generate enrichment
        matching = [r for r in self._specialist_reports if r.specialist_role == target]
        if not matching:
            self._last_feedback = f"No report found for '{target}'. Reports available: {all_report_roles}."
            if phase_steps_remaining <= 0:
                return self._auto_timeout_overseer()
            return FleetObservation(
                task_type=self._task_type,
                current_phase="overseer",
                role_instructions=OVERSEER_ROLE_INSTRUCTIONS,
                job_description=self._sample["job_description"],
                visible_sections={},
                specialist_reports=list(self._specialist_reports),
                available_actions=build_overseer_available_actions(
                    self._reports_read, all_report_roles,
                    self._reinvestigation_used, overseer_cfg,
                ),
                steps_remaining=phase_steps_remaining,
                total_steps_remaining=total_remaining,
                violations_count=self._violations_count,
                reports_read=list(self._reports_read),
                read_report_details=dict(self._read_report_details),
                feedback=self._last_feedback,
                done=False,
                reward=0.0,
            )

        report = matching[0]
        enriched = get_report_enrichment(report, self._sample)
        self._reports_read.append(target)
        self._read_report_details[target] = enriched

        reward = compute_read_reward(self._reports_read, all_report_roles, overseer_cfg)

        consensus = get_consensus_hint(self._specialist_reports)
        unread_remaining = list(set(all_report_roles) - set(self._reports_read))
        self._last_feedback = (
            f"Read: {target} report. "
            + (f"Unread: {unread_remaining}. " if unread_remaining else "All reports read. ")
            + consensus
        )

        if phase_steps_remaining <= 0:
            return self._auto_timeout_overseer(accumulated_reward=reward)

        return FleetObservation(
            task_type=self._task_type,
            current_phase="overseer",
            role_instructions=OVERSEER_ROLE_INSTRUCTIONS,
            job_description=self._sample["job_description"],
            visible_sections={},
            specialist_reports=list(self._specialist_reports),
            available_actions=build_overseer_available_actions(
                self._reports_read, all_report_roles,
                self._reinvestigation_used, overseer_cfg,
            ),
            steps_remaining=phase_steps_remaining,
            total_steps_remaining=total_remaining,
            violations_count=self._violations_count,
            reports_read=list(self._reports_read),
            read_report_details=dict(self._read_report_details),
            feedback=self._last_feedback,
            done=False,
            reward=reward,
        )

    def _handle_request_reinvestigation(
        self,
        action: FleetAction,
        all_report_roles: List[str],
        overseer_cfg: OverseerConfig,
        phase_steps_remaining: int,
        total_remaining: int,
    ) -> FleetObservation:
        """Process request_reinvestigation (once per episode)."""
        if not self._reinvestigation_used:
            self._reinvestigation_used = True
            target = action.reinvestigation_target or "fraud_specialist"
            reinvest_reason = action.reinvestigation_reason or "Need more information."
            self._last_feedback = (
                f"Reinvestigation requested for {target}: {reinvest_reason}. "
                "Specialist reports are finalised — this note informs your final decision. "
                "Now submit_final_decision."
            )
            reward = 0.02
        else:
            self._last_feedback = (
                "Reinvestigation already used this episode. "
                "Submit your final decision now."
            )
            reward = 0.0

        if phase_steps_remaining <= 0:
            return self._auto_timeout_overseer(accumulated_reward=reward)

        return FleetObservation(
            task_type=self._task_type,
            current_phase="overseer",
            role_instructions=OVERSEER_ROLE_INSTRUCTIONS,
            job_description=self._sample["job_description"],
            visible_sections={},
            specialist_reports=list(self._specialist_reports),
            available_actions=build_overseer_available_actions(
                self._reports_read, all_report_roles,
                self._reinvestigation_used, overseer_cfg,
            ),
            steps_remaining=phase_steps_remaining,
            total_steps_remaining=total_remaining,
            violations_count=self._violations_count,
            reports_read=list(self._reports_read),
            read_report_details=dict(self._read_report_details),
            feedback=self._last_feedback,
            done=False,
            reward=reward,
        )

    # ------------------------------------------------------------------
    # Independent reward sub-functions (build-guide: multiple named rewards)
    # ------------------------------------------------------------------

    def _reward_decision_accuracy(self, decision: str, fraud_flag: bool,
                                   confidence: float, gt: dict) -> float:
        """Component A: Overseer decision accuracy (0.0 – 0.70)."""
        correct_decision = (decision == gt["decision"])
        correct_fraud    = (fraud_flag == gt["is_fraud"])
        both_correct     = correct_decision and correct_fraud
        r = 0.0
        if correct_decision: r += 0.35
        if correct_fraud:    r += 0.25
        if both_correct:     r += round(0.10 * confidence, 4)
        if not both_correct and confidence >= 0.7:
            r = max(0.0, r - 0.05)   # miscalibration penalty
        return r

    def _reward_specialist_quality(self, tier_mult: float) -> tuple:
        """Component B: Specialist report quality (0.0 – 0.22)."""
        per_spec = round((0.20 / 3) * tier_mult, 4)
        correct  = [r.specialist_role for r in self._specialist_reports
                    if self._specialist_was_correct(r)]
        return per_spec * len(correct), correct

    def _reward_investigation_quality(self, gt: dict) -> float:
        """Component E: Investigation depth and tool use (0.0 – 0.05)."""
        n_sections   = len(set(self._sections_viewed))
        max_sections = len(self._sample.get("resume_sections", {}) or {})
        depth_ratio  = n_sections / max_sections if max_sections > 0 else 0.0
        tool_count   = (self._references_checked
                        + self._verifications_done
                        + self._clarifications_asked)
        tool_ratio   = min(tool_count / 3.0, 1.0)
        depth_score  = depth_ratio * 0.6 + tool_ratio * 0.4
        r = round(0.05 * depth_score, 4)
        if gt["is_fraud"] and tool_count == 0:
            r = max(0.0, r - 0.05)   # anti-shortcut penalty
        return r

    def _reward_format_compliance(self, fraud_flag: bool, fraud_reasoning: str,
                                   gt: dict) -> float:
        """Component F: Fraud reasoning quality and format compliance (0.0 – 0.05)."""
        if fraud_flag and len(fraud_reasoning.strip()) < 15:
            return -0.05   # penalty: fraud flagged with no reasoning
        indicators = gt.get("fraud_indicators", [])
        if indicators and fraud_reasoning:
            low = fraud_reasoning.lower()
            matched = sum(
                1 for ind in indicators
                if any(kw in low for kw in ind.replace("_", " ").split() if len(kw) > 3)
            )
            return round(0.05 * (matched / len(indicators)), 4)
        elif not indicators and not fraud_flag:
            return 0.02   # correctly identified clean resume
        return 0.0

    def _reward_fleet_coordination(self, both_correct: bool,
                                    correct_specialists: list) -> float:
        """Component C: Fleet coordination bonus (0.0 – 0.08).

        Rewards the full fleet working in harmony — all three specialists AND
        the overseer reaching the correct conclusion together.  Raised from
        0.03 → 0.08 to meaningfully incentivise multi-agent cooperation over
        a single-agent shortcut strategy.
        """
        if both_correct and len(correct_specialists) == 3:
            return 0.08
        return 0.0

    def _reward_oversight_quality(self, gt: dict) -> float:
        """Component D: Overseer read thoroughness + reinvestigation (0.0 – 0.08)."""
        n_reports_total = len(self._specialist_reports)
        n_reports_read  = len(set(self._reports_read))
        r = 0.0
        # D1: Read-thoroughness — must read reports before deciding
        if n_reports_total > 0 and n_reports_read >= n_reports_total:
            r += 0.04   # read ALL reports
        elif n_reports_read > 0:
            r += 0.01   # read at least one
        # D2: Reinvestigation appropriateness
        if self._reinvestigation_used and gt["is_fraud"]:
            r += 0.04   # correctly escalated a fraud case
        elif not self._reinvestigation_used and not gt["is_fraud"]:
            r += 0.03   # efficient: no wasted reinvestigation on clean resume
        return r

    def _reward_step_efficiency(self, both_correct: bool) -> float:
        """Component G: Step efficiency bonus (0.0 – 0.04, only when correct).

        Rewards decisive agents that reach correct decisions without exhausting
        their full step budget.
        """
        if both_correct and self._max_total_steps > 0:
            steps_used_ratio = self._total_steps_used / self._max_total_steps
            efficiency       = max(0.0, 1.0 - steps_used_ratio)
            return round(0.04 * efficiency, 4)
        return 0.0

    def _handle_submit_final_decision(self, action: FleetAction) -> FleetObservation:
        """
        Terminal reward function for the fleet episode.

        Delegates to seven independent sub-functions — one per reward component.
        Each sub-function can be read, tested, and tuned in isolation.

        Component breakdown (maximum possible before clamping ≈ 1.20):
        ┌──────────────────────────────────────────────────────┬──────────┐
        │ Component (sub-function)                             │  Max     │
        ├──────────────────────────────────────────────────────┼──────────┤
        │ A: Decision accuracy   _reward_decision_accuracy     │  +0.70   │
        │ B: Specialist quality  _reward_specialist_quality    │  +0.22   │
        │ C: Fleet coordination  _reward_fleet_coordination    │  +0.08   │
        │ D: Oversight quality   _reward_oversight_quality     │  +0.08   │
        │ E: Investigation depth _reward_investigation_quality │  +0.05   │
        │ F: Format compliance   _reward_format_compliance     │  +0.05   │
        │ G: Step efficiency     _reward_step_efficiency       │  +0.04   │
        ├──────────────────────────────────────────────────────┼──────────┤
        │ H: Violation penalty   (−0.05 each, max −0.25)       │  −0.25   │
        ├──────────────────────────────────────────────────────┼──────────┤
        │ Final reward: clamped to [0.0, 1.0]                  │   1.0    │
        └──────────────────────────────────────────────────────┴──────────┘
        """
        self._done = True
        gt = self._sample["ground_truth"]

        decision        = (action.decision or "reject").lower()
        fraud_flag      = action.fraud_flag if action.fraud_flag is not None else False
        confidence      = max(0.0, min(1.0, float(action.confidence or 0.5)))
        fraud_reasoning = action.fraud_reasoning or ""
        tier_mult       = {"easy": 1.0, "medium": 1.1, "hard": 1.3}.get(self._task_type, 1.0)

        # Derived flag needed by C and G sub-functions
        both_correct = (decision == gt["decision"]) and (fraud_flag == gt["is_fraud"])

        # ── A: Decision accuracy (includes confidence reward + anti-exploit) ─
        reward  = self._reward_decision_accuracy(decision, fraud_flag, confidence, gt)

        # ── B: Specialist quality ─────────────────────────────────────────────
        spec_reward, correct_specialists = self._reward_specialist_quality(tier_mult)
        reward += spec_reward

        # ── C: Fleet coordination ─────────────────────────────────────────────
        reward += self._reward_fleet_coordination(both_correct, correct_specialists)

        # ── D: Oversight quality ──────────────────────────────────────────────
        reward += self._reward_oversight_quality(gt)

        # ── E: Investigation depth ────────────────────────────────────────────
        reward += self._reward_investigation_quality(gt)

        # ── F: Format & fraud reasoning ───────────────────────────────────────
        reward += self._reward_format_compliance(fraud_flag, fraud_reasoning, gt)

        # ── G: Step efficiency ────────────────────────────────────────────────
        reward += self._reward_step_efficiency(both_correct)

        # ── H: Violation penalty ─────────────────────────────────────────────
        violation_penalty = compute_violation_penalty(self._violations_count)
        reward = max(0.0, reward - violation_penalty)

        # ── Final clamp ───────────────────────────────────────────────────────
        final_reward = max(0.0, min(1.0, reward))

        # Build feedback breakdown for transparency
        n_reports_total = len(self._specialist_reports)
        n_reports_read  = len(set(self._reports_read))
        violation_note = (
            f" Violations: {self._violations_count} (−{violation_penalty:.2f})."
            if self._violations_count > 0 else ""
        )
        read_note = (
            f" Read {n_reports_read}/{n_reports_total} reports."
            if n_reports_total > 0 else ""
        )
        correct_note = (
            f" Specialists correct: {len(correct_specialists)}/3."
            if self._specialist_reports else ""
        )
        return FleetObservation(
            task_type=self._task_type,
            current_phase="complete",
            role_instructions="",
            job_description=self._sample["job_description"],
            visible_sections=self._get_visible_sections(),
            specialist_reports=list(self._specialist_reports),
            available_actions=[],
            steps_remaining=0,
            total_steps_remaining=0,
            violations_count=self._violations_count,
            reports_read=list(self._reports_read),
            read_report_details=dict(self._read_report_details),
            feedback=(
                f"Episode complete. Decision={decision} fraud_flag={fraud_flag} "
                f"confidence={confidence:.2f}. "
                f"Final reward: {final_reward:.4f}.{violation_note}{read_note}{correct_note}"
            ),
            done=True,
            reward=final_reward,
        )

    # ------------------------------------------------------------------
    # Investigation helpers (shared across specialist phases)
    # ------------------------------------------------------------------
    def _do_view_section(self, section: str) -> float:
        section = section.lower().strip()
        if section not in VALID_SECTIONS:
            self._last_feedback = f"Invalid section '{section}'."
            return 0.0
        if section in self._sections_viewed_this_phase:
            self._last_feedback = f"Section '{section}' already viewed this phase."
            return 0.0
        # Add to both: global list keeps it visible to future phases;
        # per-phase list prevents duplicate reward within this phase.
        self._sections_viewed_this_phase.append(section)
        if section not in self._sections_viewed:
            self._sections_viewed.append(section)
        tier_mult = {"easy": 1.0, "medium": 1.2, "hard": 1.5}.get(self._task_type, 1.0)
        base = 0.03 if section in HIGH_VALUE_SECTIONS else 0.01
        reward = round(base * tier_mult, 4)
        self._last_feedback = f"Revealed section: {section}"
        return reward

    def _do_ask_clarification(self, question: str) -> float:
        self._clarifications_asked += 1
        question_lower = question.lower().strip()
        answers = self._sample.get("clarification_answers", {})
        best_key, best_score = None, 0
        for key in answers:
            overlap = len(set(key.replace("_", " ").lower().split()) & set(question_lower.split()))
            if overlap > best_score:
                best_score, best_key = overlap, key
        tier_mult = {"easy": 1.0, "medium": 1.2, "hard": 1.5}.get(self._task_type, 1.0)
        if best_key and best_score > 0:
            self._last_clarification = answers[best_key]
            reward = round(0.03 * tier_mult, 4)
        else:
            self._last_clarification = "No specific information available for that question."
            reward = round(0.01 * tier_mult, 4)
        self._last_feedback = "Clarification received."
        return reward

    def _do_check_reference(self, ref_id: str) -> float:
        self._references_checked += 1
        refs = self._sample.get("reference_check_results", {})
        ref_id = ref_id.lower().strip()
        if ref_id in refs:
            rd = refs[ref_id]
            self._last_reference = f"{rd['name']}: {rd['response']}"
        else:
            available = list(refs.keys())
            if available:
                self._last_reference = (
                    f"Reference '{ref_id}' not found. "
                    f"Available references: {available}."
                )
            else:
                self._last_reference = "No references available."
            self._last_feedback = "Reference check completed."
            return 0.0
        is_fraud = self._sample["ground_truth"].get("is_fraud", False)
        reward = 0.05 if is_fraud else 0.02
        self._last_feedback = "Reference check completed."
        return reward

    def _do_verify_credential(self) -> float:
        self._verifications_done += 1
        verification = self._sample.get("verification_data", {})
        parts = [
            f"{k.replace('_', ' ').title()}: {'VERIFIED' if v else 'FAILED'}"
            for k, v in verification.items()
        ]
        self._last_verification = "; ".join(parts) if parts else "No verification data."
        has_failed = any(not v for v in verification.values())
        reward = 0.05 if has_failed else 0.02
        self._last_feedback = "Credential verification completed."
        return reward

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    def _score_specialist_report(self, phase: str, report: SpecialistReport) -> float:
        """Award reward for a useful specialist report."""
        gt = self._sample["ground_truth"]
        reward = 0.0

        if phase == "fraud_specialist":
            # Correct fraud signal: specialist flagged issues when fraud exists and vice versa
            correct_signal = (report.has_issues == gt["is_fraud"])
            reward += 0.06 if correct_signal else 0.01
            # Bonus for high confidence when correct
            if correct_signal:
                reward += round(0.04 * report.confidence, 4)

        elif phase == "skills_specialist":
            # Correct match signal: flag issues only when decision is reject
            expected_issues = (gt["decision"] == "reject")
            correct_signal = (report.has_issues == expected_issues)
            reward += 0.05 if correct_signal else 0.01
            if correct_signal:
                reward += round(0.03 * report.confidence, 4)

        elif phase == "timeline_specialist":
            # Similar to skills: flag issues if fraud or reject
            expected_issues = gt["is_fraud"] or (gt["decision"] == "reject")
            correct_signal = (report.has_issues == expected_issues)
            reward += 0.05 if correct_signal else 0.01
            if correct_signal:
                reward += round(0.03 * report.confidence, 4)

        return round(min(reward, 0.10), 4)

    def _specialist_was_correct(self, report: SpecialistReport) -> bool:
        """Check if a specialist's has_issues flag was accurate."""
        gt = self._sample["ground_truth"]
        phase = report.specialist_role
        if phase == "fraud_specialist":
            return report.has_issues == gt["is_fraud"]
        elif phase == "skills_specialist":
            return report.has_issues == (gt["decision"] == "reject")
        elif phase == "timeline_specialist":
            return report.has_issues == (gt["is_fraud"] or gt["decision"] == "reject")
        return False

    # ------------------------------------------------------------------
    # Available actions per phase  (Day 2: delegate to SpecialistActionValidator)
    # ------------------------------------------------------------------
    def _get_available_actions(self) -> List[str]:
        if self._phase_idx >= len(PHASES):
            return []
        phase = PHASES[self._phase_idx]
        config = SPECIALIST_CONFIGS.get(phase)
        if config is None:
            return []
        validator = SpecialistActionValidator(config)
        return validator.available_actions(
            self._sections_viewed_this_phase,
            self._references_checked,
            self._verifications_done,
            self._reinvestigation_used,
        )

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def _get_visible_sections(self) -> Dict[str, str]:
        if not self._sample:
            return {}
        sections = self._sample["resume_sections"]
        return {s: sections[s] for s in self._sections_viewed if s in sections}

    def _terminal_obs(self, reward: float, feedback: str) -> FleetObservation:
        return FleetObservation(
            task_type=self._task_type,
            current_phase="complete",
            role_instructions="",
            job_description=self._sample.get("job_description", "") if self._sample else "",
            visible_sections=self._get_visible_sections(),
            specialist_reports=list(self._specialist_reports),
            available_actions=[],
            steps_remaining=0,
            total_steps_remaining=0,
            violations_count=self._violations_count,
            reports_read=list(self._reports_read),
            read_report_details=dict(self._read_report_details),
            feedback=feedback,
            done=True,
            reward=reward,
        )

    def _auto_advance_phase(self, phase: str, accumulated_reward: float) -> FleetObservation:
        """Auto-submit report when phase budget runs out without explicit report."""
        report = SpecialistReport(
            specialist_role=phase,
            findings="[Auto-submitted: phase budget exhausted without explicit report]",
            has_issues=False,
            confidence=0.3,
        )
        self._specialist_reports.append(report)
        self._phase_idx += 1
        self._phase_steps_used = 0
        self._sections_viewed_this_phase = []   # reset per-phase section tracking
        self._last_clarification = None
        self._last_reference = None
        self._last_verification = None

        if self._phase_idx >= len(PHASES):
            self._done = True
            return self._terminal_obs(
                reward=accumulated_reward,
                feedback="Phase budget exhausted. Episode ended."
            )

        next_phase = PHASES[self._phase_idx]
        next_budget = PHASE_BUDGETS[self._task_type][next_phase]
        total_remaining = self._max_total_steps - self._total_steps_used

        if next_phase == "overseer":
            next_available = build_overseer_available_actions(
                self._reports_read,
                [r.specialist_role for r in self._specialist_reports],
                self._reinvestigation_used,
                OverseerConfig(),
            )
            next_role_instructions = OVERSEER_ROLE_INSTRUCTIONS
            next_visible: Dict[str, str] = {}
        else:
            next_validator = SpecialistActionValidator(SPECIALIST_CONFIGS[next_phase])
            next_available = next_validator.available_actions(
                self._sections_viewed_this_phase,   # [] for a fresh phase
                self._references_checked,
                self._verifications_done, self._reinvestigation_used,
            )
            next_role_instructions = next_validator.role_instructions()
            next_visible = next_validator.filter_sections(self._get_visible_sections())

        return FleetObservation(
            task_type=self._task_type,
            current_phase=next_phase,
            role_instructions=next_role_instructions,
            job_description=self._sample["job_description"],
            visible_sections=next_visible,
            specialist_reports=list(self._specialist_reports),
            available_actions=next_available,
            steps_remaining=next_budget,
            total_steps_remaining=total_remaining,
            violations_count=self._violations_count,
            reports_read=list(self._reports_read),
            read_report_details=dict(self._read_report_details),
            feedback=f"Phase budget for {phase} exhausted. Auto-advanced to {next_phase}.",
            done=False,
            reward=accumulated_reward,
        )

    def _auto_timeout_overseer(self, accumulated_reward: float = 0.0) -> FleetObservation:
        """
        Called when the overseer's phase budget is exhausted.
        accumulated_reward carries any reward earned on the step that triggered
        the timeout (e.g. a read_reports reward on the final overseer step).
        """
        self._done = True
        final_reward = max(0.0, min(1.0, accumulated_reward))
        return FleetObservation(
            task_type=self._task_type,
            current_phase="complete",
            role_instructions="",
            job_description=self._sample["job_description"],
            visible_sections=self._get_visible_sections(),
            specialist_reports=list(self._specialist_reports),
            available_actions=[],
            steps_remaining=0,
            total_steps_remaining=0,
            violations_count=self._violations_count,
            reports_read=list(self._reports_read),
            read_report_details=dict(self._read_report_details),
            feedback=(
                f"Overseer step budget exhausted. Episode ended. "
                f"Read {len(set(self._reports_read))}/{len(self._specialist_reports)} reports."
            ),
            done=True,
            reward=final_reward,
        )

    @property
    def state(self) -> FleetState:
        return FleetState(
            current_index=self._current_index,
            task_type=self._task_type,
            phase_idx=self._phase_idx,
            phase_steps_used=self._phase_steps_used,
            total_steps_used=self._total_steps_used,
            max_total_steps=self._max_total_steps,
            sections_viewed=list(self._sections_viewed),
            specialist_reports=[r.model_dump() for r in self._specialist_reports],
            references_checked=self._references_checked,
            verifications_done=self._verifications_done,
            clarifications_asked=self._clarifications_asked,
            reinvestigation_used=self._reinvestigation_used,
            violations_count=self._violations_count,
            reports_read=list(self._reports_read),
            read_report_details=dict(self._read_report_details),
            done=self._done,
        )
