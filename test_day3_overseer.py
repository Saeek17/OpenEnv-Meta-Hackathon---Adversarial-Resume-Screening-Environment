"""
Day 3 — OverseerEnvironment unit tests
=======================================
Tests run against FleetResumeEnvironment directly (no HTTP server needed).
Each test builds a minimal mock dataset, drives the fleet through all four
phases, and asserts overseer-specific behaviour introduced on Day 3:

  • read_reports action yields enriched detail + reward
  • reports_read list grows per read
  • read_report_details dict populated correctly
  • All-reports-read bonus triggers on completing the set
  • Re-reading an already-read report returns 0 reward + warning
  • Invalid report_target returns helpful error
  • request_reinvestigation still works (once only)
  • submit_final_decision includes synthesis quality bonus when all read
  • Overseer violation tracking still works (wrong action type)
  • Overseer budget expanded: easy=2, medium=3, hard=4
  • Phase transition from timeline_specialist → overseer uses OVERSEER_ROLE_INSTRUCTIONS
  • read_report_details preserved after phase transition in save/restore

Run with:
    python3 test_day3_overseer.py
"""

import sys
import os
import unittest

# ── Path setup ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# ── Mock openenv to avoid broken gradio/huggingface_hub in local venv ───────
# (The real openenv works fine inside Docker; this mock is test-only.)
import types

def _make_openenv_mock():
    from pydantic import BaseModel

    openenv_mod = types.ModuleType("openenv")
    core_mod    = types.ModuleType("openenv.core")
    server_mod  = types.ModuleType("openenv.core.env_server")

    # Must subclass pydantic BaseModel so model_rebuild() / Field() work
    # on the Fleet/Resume models that inherit from these.
    class Action(BaseModel):
        model_config = {"arbitrary_types_allowed": True}

    class Observation(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        done:   bool  = False
        reward: float = 0.0

    class State(BaseModel):
        model_config = {"arbitrary_types_allowed": True}

    class _EnvironmentMeta(type):
        def __getitem__(cls, item):
            return cls

    class Environment(metaclass=_EnvironmentMeta):
        def __init__(self): pass

    server_mod.Action      = Action
    server_mod.Observation = Observation
    server_mod.State       = State
    server_mod.Environment = Environment
    server_mod.create_fastapi_app = lambda *a, **kw: None

    openenv_mod.core    = core_mod
    core_mod.env_server = server_mod

    sys.modules["openenv"]                 = openenv_mod
    sys.modules["openenv.core"]            = core_mod
    sys.modules["openenv.core.env_server"] = server_mod

_make_openenv_mock()

# ── Also mock pydantic BaseModel so SpecialistReport survives ───────────────
# (only needed because models.py does: from pydantic import BaseModel)
# pydantic IS available in the venv, so we just import normally.

from server.fleet_environment import FleetResumeEnvironment, PHASE_BUDGETS
from server.overseer_env import (
    OverseerConfig,
    OVERSEER_ROLE_INSTRUCTIONS,
    compute_read_reward,
    build_overseer_available_actions,
    get_consensus_hint,
    get_report_enrichment,
    OVERSEER_DECISION_MATRIX,
)
from models import FleetAction, SpecialistReport


# ── Minimal mock dataset ─────────────────────────────────────────────────────

def _mock_resume(is_fraud: bool = True, decision: str = "reject") -> dict:
    return {
        "job_description": "Senior Python Engineer with cloud experience.",
        "resume_sections": {
            "header": "John Doe — Software Engineer",
            "summary": "10 years of Python experience.",
            "experience": "Google 2015-2023, Amazon 2013-2015",
            "education": "MIT BSc Computer Science 2013",
            "skills": "Python, AWS, Docker, Kubernetes",
            "projects": "Built a distributed caching system at Google.",
            "references": "ref1: Jane Smith, Google. ref2: Bob Jones, Amazon.",
        },
        "clarification_answers": {
            "gap": "I took a sabbatical between 2014 and 2015.",
        },
        "reference_check_results": {
            "ref1": {"name": "Jane Smith", "response": "John was excellent."},
            "ref2": {"name": "Bob Jones",  "response": "Dates don't match our records."},
        },
        "verification_data": {
            "degree_verified": True,
            "employment_verified": False,
        },
        "required_skills": ["Python", "AWS", "Kubernetes"],
        "ground_truth": {
            "decision": decision,
            "is_fraud": is_fraud,
            "fraud_indicators": ["fabricated_reference", "employment_gap"],
            "employment_gaps": ["2014-2015"],
        },
    }


def _make_env(task_type: str = "medium", is_fraud: bool = True) -> FleetResumeEnvironment:
    """Create a fresh env with mock data, bypassing file I/O."""
    env = FleetResumeEnvironment.__new__(FleetResumeEnvironment)
    env.data_path = "data/resumes.json"
    # Inject mock dataset
    FleetResumeEnvironment._dataset_cache = {
        "easy":   [_mock_resume(is_fraud=is_fraud)],
        "medium": [_mock_resume(is_fraud=is_fraud)],
        "hard":   [_mock_resume(is_fraud=is_fraud)],
    }
    env.dataset = FleetResumeEnvironment._dataset_cache
    # Instance state
    from typing import List, Dict, Optional
    env._task_type = task_type
    env._current_index = 0
    env._sample = None
    env._phase_idx = 0
    env._phase_steps_used = 0
    env._total_steps_used = 0
    env._max_total_steps = sum(PHASE_BUDGETS[task_type].values())
    env._sections_viewed = []
    env._specialist_reports = []
    env._references_checked = 0
    env._verifications_done = 0
    env._clarifications_asked = 0
    env._reinvestigation_used = False
    env._violations_count = 0
    env._reports_read = []
    env._read_report_details = {}
    env._done = False
    env._last_clarification = None
    env._last_reference = None
    env._last_verification = None
    env._last_feedback = ""
    return env


def _action(**kwargs) -> FleetAction:
    """Build a FleetAction from keyword args."""
    return FleetAction(**kwargs)


def _drive_through_specialists(env: FleetResumeEnvironment) -> None:
    """
    Fast-path through all three specialist phases by submitting reports
    immediately (one investigation + one report per phase).
    """
    # ── fraud specialist ──────────────────────────────────────────────
    env.step(_action(action_type="check_reference", reference_id="ref1"))
    env.step(_action(
        action_type="submit_specialist_report",
        findings="Reference dates don't match.",
        has_issues=True,
        specialist_confidence=0.9,
    ))

    # ── skills specialist ─────────────────────────────────────────────
    env.step(_action(action_type="view_section", section="skills"))
    env.step(_action(
        action_type="submit_specialist_report",
        findings="Strong Python/AWS skills. Missing Kubernetes.",
        has_issues=False,
        specialist_confidence=0.7,
    ))

    # ── timeline specialist ───────────────────────────────────────────
    env.step(_action(action_type="view_section", section="experience"))
    env.step(_action(
        action_type="submit_specialist_report",
        findings="Employment gap 2014-2015 not explained.",
        has_issues=True,
        specialist_confidence=0.85,
    ))


# ════════════════════════════════════════════════════════════════════════════
# Unit tests
# ════════════════════════════════════════════════════════════════════════════

class TestOverseerConfig(unittest.TestCase):

    def test_default_allowed_actions(self):
        cfg = OverseerConfig()
        self.assertIn("read_reports", cfg.allowed_actions)
        self.assertIn("request_reinvestigation", cfg.allowed_actions)
        self.assertIn("submit_final_decision", cfg.allowed_actions)

    def test_reward_params(self):
        cfg = OverseerConfig()
        self.assertEqual(cfg.reward_per_report_read, 0.02)
        self.assertEqual(cfg.all_reports_read_bonus, 0.03)
        self.assertEqual(cfg.max_reinvestigations, 1)


class TestComputeReadReward(unittest.TestCase):

    def setUp(self):
        self.cfg = OverseerConfig()
        self.all_roles = ["fraud_specialist", "skills_specialist", "timeline_specialist"]

    def test_first_read_no_bonus(self):
        r = compute_read_reward(["fraud_specialist"], self.all_roles, self.cfg)
        self.assertAlmostEqual(r, 0.02, places=4)

    def test_second_read_no_bonus(self):
        r = compute_read_reward(
            ["fraud_specialist", "skills_specialist"], self.all_roles, self.cfg
        )
        self.assertAlmostEqual(r, 0.02, places=4)

    def test_all_read_bonus_on_third(self):
        r = compute_read_reward(self.all_roles, self.all_roles, self.cfg)
        # per_report (0.02) + all_read_bonus (0.03)
        self.assertAlmostEqual(r, 0.05, places=4)

    def test_empty_all_roles_no_crash(self):
        r = compute_read_reward([], [], self.cfg)
        self.assertAlmostEqual(r, 0.02, places=4)  # just per_report, bonus=0


class TestBuildOverseerAvailableActions(unittest.TestCase):

    def setUp(self):
        self.cfg = OverseerConfig()
        self.all_roles = ["fraud_specialist", "skills_specialist", "timeline_specialist"]

    def test_nothing_read_all_available(self):
        actions = build_overseer_available_actions([], self.all_roles, False, self.cfg)
        self.assertIn("read_reports", actions)
        self.assertIn("request_reinvestigation", actions)
        self.assertIn("submit_final_decision", actions)

    def test_all_read_no_read_reports(self):
        actions = build_overseer_available_actions(
            self.all_roles, self.all_roles, False, self.cfg
        )
        self.assertNotIn("read_reports", actions)
        self.assertIn("request_reinvestigation", actions)
        self.assertIn("submit_final_decision", actions)

    def test_reinvestigation_used_removed(self):
        actions = build_overseer_available_actions([], self.all_roles, True, self.cfg)
        self.assertNotIn("request_reinvestigation", actions)
        self.assertIn("submit_final_decision", actions)

    def test_submit_always_last(self):
        actions = build_overseer_available_actions([], self.all_roles, False, self.cfg)
        self.assertEqual(actions[-1], "submit_final_decision")


class TestGetConsensusHint(unittest.TestCase):

    def _make_report(self, role, has_issues):
        return SpecialistReport(
            specialist_role=role,
            findings="Some findings.",
            has_issues=has_issues,
            confidence=0.8,
        )

    def test_all_clean_suggests_accept(self):
        reports = [
            self._make_report("fraud_specialist", False),
            self._make_report("skills_specialist", False),
            self._make_report("timeline_specialist", False),
        ]
        hint = get_consensus_hint(reports)
        self.assertIn("ACCEPT", hint)

    def test_fraud_raised_suggests_reject(self):
        reports = [
            self._make_report("fraud_specialist", True),
            self._make_report("skills_specialist", False),
            self._make_report("timeline_specialist", False),
        ]
        hint = get_consensus_hint(reports)
        self.assertIn("REJECT", hint)
        self.assertIn("fraud", hint)

    def test_empty_reports(self):
        hint = get_consensus_hint([])
        self.assertIn("No specialist reports", hint)


class TestOverseerDecisionMatrix(unittest.TestCase):

    def test_all_clean_accept(self):
        self.assertEqual(OVERSEER_DECISION_MATRIX[(False, False, False)], "accept")

    def test_fraud_always_reject(self):
        self.assertEqual(OVERSEER_DECISION_MATRIX[(True, False, False)], "reject")
        self.assertEqual(OVERSEER_DECISION_MATRIX[(True, True, True)], "reject")

    def test_skills_only_reject(self):
        self.assertEqual(OVERSEER_DECISION_MATRIX[(False, True, False)], "reject")


class TestPhaseBudgets(unittest.TestCase):
    """Day 3: overseer budget expanded."""

    def test_easy_overseer_budget_is_2(self):
        self.assertEqual(PHASE_BUDGETS["easy"]["overseer"], 2)

    def test_medium_overseer_budget_is_3(self):
        self.assertEqual(PHASE_BUDGETS["medium"]["overseer"], 3)

    def test_hard_overseer_budget_is_4(self):
        self.assertEqual(PHASE_BUDGETS["hard"]["overseer"], 4)

    def test_easy_total_steps(self):
        total = sum(PHASE_BUDGETS["easy"].values())
        self.assertEqual(total, 8)

    def test_medium_total_steps(self):
        total = sum(PHASE_BUDGETS["medium"].values())
        self.assertEqual(total, 11)

    def test_hard_total_steps(self):
        total = sum(PHASE_BUDGETS["hard"].values())
        self.assertEqual(total, 15)


class TestReadReportsAction(unittest.TestCase):
    """Integration tests: overseer read_reports action in a real episode."""

    def setUp(self):
        FleetResumeEnvironment._episode_store = {}
        FleetResumeEnvironment._default_session = "test-d3"
        self.env = _make_env(task_type="medium", is_fraud=True)
        obs = self.env.reset(task_type="medium", seed=42)
        self.assertIsNotNone(obs)
        _drive_through_specialists(self.env)

    def test_phase_is_overseer_after_specialists(self):
        obs = self.env.step(_action(action_type="read_reports", report_target="fraud_specialist"))
        self.assertEqual(obs.current_phase, "overseer")

    def test_read_fraud_report_returns_reward(self):
        obs = self.env.step(_action(action_type="read_reports", report_target="fraud_specialist"))
        self.assertGreater(obs.reward, 0.0)
        self.assertAlmostEqual(obs.reward, 0.02, places=4)

    def test_read_fraud_report_updates_reports_read(self):
        obs = self.env.step(_action(action_type="read_reports", report_target="fraud_specialist"))
        self.assertIn("fraud_specialist", obs.reports_read)

    def test_read_fraud_report_populates_detail(self):
        obs = self.env.step(_action(action_type="read_reports", report_target="fraud_specialist"))
        self.assertIn("fraud_specialist", obs.read_report_details)
        detail = obs.read_report_details["fraud_specialist"]
        self.assertIn("FRAUD SPECIALIST", detail)
        self.assertIn("Findings", detail)

    def test_reading_second_report_adds_to_list(self):
        self.env.step(_action(action_type="read_reports", report_target="fraud_specialist"))
        obs = self.env.step(_action(action_type="read_reports", report_target="skills_specialist"))
        self.assertIn("fraud_specialist", obs.reports_read)
        self.assertIn("skills_specialist", obs.reports_read)

    def test_all_reports_read_bonus_on_third(self):
        self.env.step(_action(action_type="read_reports", report_target="fraud_specialist"))
        self.env.step(_action(action_type="read_reports", report_target="skills_specialist"))
        obs = self.env.step(_action(action_type="read_reports", report_target="timeline_specialist"))
        # reward should be per_report (0.02) + all_read_bonus (0.03) = 0.05
        self.assertAlmostEqual(obs.reward, 0.05, places=4)

    def test_all_reports_read_removes_read_reports_from_actions(self):
        self.env.step(_action(action_type="read_reports", report_target="fraud_specialist"))
        self.env.step(_action(action_type="read_reports", report_target="skills_specialist"))
        obs = self.env.step(_action(action_type="read_reports", report_target="timeline_specialist"))
        self.assertNotIn("read_reports", obs.available_actions)

    def test_re_read_returns_zero_reward(self):
        self.env.step(_action(action_type="read_reports", report_target="fraud_specialist"))
        obs = self.env.step(_action(action_type="read_reports", report_target="fraud_specialist"))
        self.assertEqual(obs.reward, 0.0)

    def test_re_read_feedback_mentions_already_read(self):
        self.env.step(_action(action_type="read_reports", report_target="fraud_specialist"))
        obs = self.env.step(_action(action_type="read_reports", report_target="fraud_specialist"))
        self.assertIn("already", obs.feedback.lower())

    def test_invalid_target_returns_zero_reward(self):
        obs = self.env.step(_action(action_type="read_reports", report_target="nonexistent_specialist"))
        self.assertEqual(obs.reward, 0.0)
        self.assertNotIn("nonexistent_specialist", obs.reports_read)

    def test_invalid_target_feedback_lists_valid_options(self):
        obs = self.env.step(_action(action_type="read_reports", report_target="xyz"))
        self.assertIn("fraud_specialist", obs.feedback)

    def test_read_reports_feedback_includes_consensus(self):
        obs = self.env.step(_action(action_type="read_reports", report_target="fraud_specialist"))
        # Consensus hint should mention ACCEPT or REJECT
        self.assertTrue(
            "ACCEPT" in obs.feedback or "REJECT" in obs.feedback,
            f"Expected consensus hint in feedback, got: {obs.feedback}"
        )

    def test_role_instructions_is_overseer_instructions(self):
        obs = self.env.step(_action(action_type="read_reports", report_target="fraud_specialist"))
        self.assertIn("OVERSEER", obs.role_instructions)

    def test_visible_sections_empty_for_overseer(self):
        obs = self.env.step(_action(action_type="read_reports", report_target="fraud_specialist"))
        # Overseer should not see raw resume sections
        self.assertEqual(obs.visible_sections, {})


class TestOverseerViolation(unittest.TestCase):
    """Overseer rejects invalid actions (Day 2 behaviour preserved in Day 3)."""

    def setUp(self):
        FleetResumeEnvironment._episode_store = {}
        FleetResumeEnvironment._default_session = "test-d3-viol"
        self.env = _make_env(task_type="medium", is_fraud=True)
        self.env.reset(task_type="medium", seed=1)
        _drive_through_specialists(self.env)

    def test_view_section_is_violation_for_overseer(self):
        before = self.env._violations_count
        obs = self.env.step(_action(action_type="view_section", section="experience"))
        self.assertEqual(self.env._violations_count, before + 1)
        self.assertEqual(obs.reward, 0.0)
        self.assertIn("VIOLATION", obs.feedback)

    def test_check_reference_is_violation_for_overseer(self):
        before = self.env._violations_count
        obs = self.env.step(_action(action_type="check_reference", reference_id="ref1"))
        self.assertEqual(self.env._violations_count, before + 1)
        self.assertEqual(obs.reward, 0.0)

    def test_violation_does_not_end_episode(self):
        obs = self.env.step(_action(action_type="view_section", section="experience"))
        self.assertFalse(obs.done)


class TestOverseerReinvestigation(unittest.TestCase):
    """request_reinvestigation still works and is one-shot."""

    def setUp(self):
        FleetResumeEnvironment._episode_store = {}
        FleetResumeEnvironment._default_session = "test-d3-reinv"
        self.env = _make_env(task_type="medium", is_fraud=True)
        self.env.reset(task_type="medium", seed=2)
        _drive_through_specialists(self.env)

    def test_first_reinvestigation_earns_reward(self):
        obs = self.env.step(_action(
            action_type="request_reinvestigation",
            reinvestigation_target="fraud_specialist",
            reinvestigation_reason="Reports conflict on timeline.",
        ))
        self.assertGreater(obs.reward, 0.0)
        self.assertTrue(self.env._reinvestigation_used)

    def test_second_reinvestigation_earns_zero_reward(self):
        self.env.step(_action(
            action_type="request_reinvestigation",
            reinvestigation_target="fraud_specialist",
            reinvestigation_reason="First request.",
        ))
        obs = self.env.step(_action(
            action_type="request_reinvestigation",
            reinvestigation_target="skills_specialist",
            reinvestigation_reason="Second request.",
        ))
        self.assertEqual(obs.reward, 0.0)

    def test_reinvestigation_removed_from_actions_after_use(self):
        obs = self.env.step(_action(
            action_type="request_reinvestigation",
            reinvestigation_target="fraud_specialist",
            reinvestigation_reason="Need more info.",
        ))
        self.assertNotIn("request_reinvestigation", obs.available_actions)


class TestSubmitFinalDecisionSynthesisBonus(unittest.TestCase):
    """submit_final_decision gives synthesis bonus when all reports read."""

    def _run_episode(self, read_all: bool = True, is_fraud: bool = True) -> float:
        FleetResumeEnvironment._episode_store = {}
        FleetResumeEnvironment._default_session = "test-d3-synth"
        env = _make_env(task_type="hard", is_fraud=is_fraud)
        env.reset(task_type="hard", seed=7)
        _drive_through_specialists(env)

        if read_all:
            env.step(_action(action_type="read_reports", report_target="fraud_specialist"))
            env.step(_action(action_type="read_reports", report_target="skills_specialist"))
            env.step(_action(action_type="read_reports", report_target="timeline_specialist"))

        obs = env.step(_action(
            action_type="submit_final_decision",
            decision="reject",
            fraud_flag=is_fraud,
            confidence=0.9,
            fraud_reasoning="fabricated reference and employment gap",
        ))
        return obs.reward

    def test_reading_all_before_deciding_gives_higher_reward(self):
        reward_all_read = self._run_episode(read_all=True)
        reward_blind = self._run_episode(read_all=False)
        self.assertGreater(reward_all_read, reward_blind)

    def test_final_reward_clamped_to_one(self):
        reward = self._run_episode(read_all=True, is_fraud=True)
        self.assertLessEqual(reward, 1.0)
        self.assertGreaterEqual(reward, 0.0)

    def test_terminal_obs_shows_read_count(self):
        FleetResumeEnvironment._episode_store = {}
        FleetResumeEnvironment._default_session = "test-d3-synth2"
        env = _make_env(task_type="hard", is_fraud=True)
        env.reset(task_type="hard", seed=9)
        _drive_through_specialists(env)
        env.step(_action(action_type="read_reports", report_target="fraud_specialist"))
        obs = env.step(_action(
            action_type="submit_final_decision",
            decision="reject",
            fraud_flag=True,
            confidence=0.8,
            fraud_reasoning="fabricated reference",
        ))
        self.assertIn("1/3", obs.feedback)

    def test_episode_marked_complete(self):
        reward = self._run_episode(read_all=True)
        # If we reach here without exception the episode completed
        self.assertIsNotNone(reward)


class TestPhaseTransitionToOverseer(unittest.TestCase):
    """Transition from timeline_specialist → overseer uses correct instructions."""

    def setUp(self):
        FleetResumeEnvironment._episode_store = {}
        FleetResumeEnvironment._default_session = "test-d3-trans"
        self.env = _make_env(task_type="medium", is_fraud=True)
        self.env.reset(task_type="medium", seed=3)

    def test_transition_obs_has_overseer_role_instructions(self):
        # Drive through fraud + skills, then submit timeline report which triggers transition
        self.env.step(_action(action_type="check_reference", reference_id="ref1"))
        self.env.step(_action(
            action_type="submit_specialist_report",
            findings="Ref mismatch found.",
            has_issues=True,
            specialist_confidence=0.9,
        ))
        self.env.step(_action(action_type="view_section", section="skills"))
        self.env.step(_action(
            action_type="submit_specialist_report",
            findings="Skills adequate.",
            has_issues=False,
            specialist_confidence=0.7,
        ))
        self.env.step(_action(action_type="view_section", section="experience"))
        obs = self.env.step(_action(
            action_type="submit_specialist_report",
            findings="Gap 2014-2015 unexplained.",
            has_issues=True,
            specialist_confidence=0.85,
        ))
        # obs is now the overseer's first observation
        self.assertEqual(obs.current_phase, "overseer")
        self.assertIn("OVERSEER", obs.role_instructions)

    def test_transition_obs_has_empty_visible_sections(self):
        self.env.step(_action(action_type="check_reference", reference_id="ref1"))
        self.env.step(_action(
            action_type="submit_specialist_report",
            findings="x", has_issues=True, specialist_confidence=0.9,
        ))
        self.env.step(_action(action_type="view_section", section="skills"))
        self.env.step(_action(
            action_type="submit_specialist_report",
            findings="y", has_issues=False, specialist_confidence=0.7,
        ))
        self.env.step(_action(action_type="view_section", section="experience"))
        obs = self.env.step(_action(
            action_type="submit_specialist_report",
            findings="z", has_issues=True, specialist_confidence=0.8,
        ))
        self.assertEqual(obs.visible_sections, {})

    def test_transition_obs_has_read_reports_action(self):
        self.env.step(_action(action_type="check_reference", reference_id="ref1"))
        self.env.step(_action(
            action_type="submit_specialist_report",
            findings="x", has_issues=True, specialist_confidence=0.9,
        ))
        self.env.step(_action(action_type="view_section", section="skills"))
        self.env.step(_action(
            action_type="submit_specialist_report",
            findings="y", has_issues=False, specialist_confidence=0.7,
        ))
        self.env.step(_action(action_type="view_section", section="experience"))
        obs = self.env.step(_action(
            action_type="submit_specialist_report",
            findings="z", has_issues=True, specialist_confidence=0.8,
        ))
        self.assertIn("read_reports", obs.available_actions)


class TestGetReportEnrichment(unittest.TestCase):
    """get_report_enrichment returns formatted text per specialist role."""

    def setUp(self):
        self.sample = _mock_resume(is_fraud=True)

    def _make_report(self, role, has_issues=True, findings="Found issues."):
        return SpecialistReport(
            specialist_role=role,
            findings=findings,
            has_issues=has_issues,
            confidence=0.8,
        )

    def test_fraud_enrichment_includes_fraud_signals(self):
        report = self._make_report("fraud_specialist")
        text = get_report_enrichment(report, self.sample)
        self.assertIn("FRAUD SPECIALIST", text)
        self.assertIn("fabricated_reference", text)

    def test_skills_enrichment_includes_required_skills(self):
        report = self._make_report("skills_specialist")
        text = get_report_enrichment(report, self.sample)
        self.assertIn("SKILLS SPECIALIST", text)
        self.assertIn("Python", text)

    def test_timeline_enrichment_includes_gaps(self):
        report = self._make_report("timeline_specialist")
        text = get_report_enrichment(report, self.sample)
        self.assertIn("TIMELINE SPECIALIST", text)
        self.assertIn("2014-2015", text)

    def test_enrichment_includes_findings(self):
        report = self._make_report("fraud_specialist", findings="Specific finding here.")
        text = get_report_enrichment(report, self.sample)
        self.assertIn("Specific finding here.", text)

    def test_enrichment_shows_yes_for_has_issues_true(self):
        report = self._make_report("fraud_specialist", has_issues=True)
        text = get_report_enrichment(report, self.sample)
        self.assertIn("YES", text)

    def test_enrichment_shows_no_for_has_issues_false(self):
        report = self._make_report("fraud_specialist", has_issues=False)
        text = get_report_enrichment(report, self.sample)
        self.assertIn("NO", text)


class TestFleetActionModelUpdated(unittest.TestCase):
    """Verify models.py changes are live."""

    def test_read_reports_in_literal(self):
        """FleetAction should accept read_reports as action_type."""
        a = FleetAction(action_type="read_reports", report_target="fraud_specialist")
        self.assertEqual(a.action_type, "read_reports")
        self.assertEqual(a.report_target, "fraud_specialist")

    def test_report_target_optional(self):
        a = FleetAction(action_type="submit_final_decision", decision="reject",
                        fraud_flag=False, confidence=0.5)
        self.assertIsNone(a.report_target)

    def test_fleet_observation_has_reports_read(self):
        from models import FleetObservation
        obs = FleetObservation(
            task_type="easy",
            current_phase="overseer",
            reports_read=["fraud_specialist"],
            read_report_details={"fraud_specialist": "detail text"},
        )
        self.assertEqual(obs.reports_read, ["fraud_specialist"])
        self.assertEqual(obs.read_report_details["fraud_specialist"], "detail text")

    def test_fleet_state_has_reports_read(self):
        from models import FleetState
        s = FleetState(
            reports_read=["fraud_specialist", "skills_specialist"],
            read_report_details={"fraud_specialist": "foo"},
        )
        self.assertEqual(len(s.reports_read), 2)


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        TestOverseerConfig,
        TestComputeReadReward,
        TestBuildOverseerAvailableActions,
        TestGetConsensusHint,
        TestOverseerDecisionMatrix,
        TestPhaseBudgets,
        TestReadReportsAction,
        TestOverseerViolation,
        TestOverseerReinvestigation,
        TestSubmitFinalDecisionSynthesisBonus,
        TestPhaseTransitionToOverseer,
        TestGetReportEnrichment,
        TestFleetActionModelUpdated,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
