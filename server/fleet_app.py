from openenv.core.env_server import create_fastapi_app
from .fleet_environment import FleetResumeEnvironment
from models import FleetAction, FleetObservation

# Create the FastAPI app for the multi-agent fleet environment
app = create_fastapi_app(
    FleetResumeEnvironment,
    action_cls=FleetAction,
    observation_cls=FleetObservation,
)

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
import logging

logger = logging.getLogger(__name__)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error"},
    )


@app.get("/", response_class=HTMLResponse)
@app.get("/web", response_class=HTMLResponse)
async def home():
    """Home page for the Hugging Face Space."""
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hiring Fleet — AI Oversight System</title>
<style>
  :root {
    --bg:       #0a0f1e;
    --surface:  #111827;
    --card:     #1a2235;
    --border:   #1e3a5f;
    --accent:   #3b82f6;
    --purple:   #7c3aed;
    --green:    #10b981;
    --yellow:   #f59e0b;
    --red:      #ef4444;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --mono:     'JetBrains Mono', 'Fira Code', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, sans-serif; line-height: 1.6; }

  /* ── Nav ── */
  nav {
    position: sticky; top: 0; z-index: 100;
    background: rgba(10,15,30,0.92); backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    padding: 0 2rem; display: flex; align-items: center; justify-content: space-between; height: 56px;
  }
  .nav-brand { font-weight: 700; font-size: 1rem; display: flex; align-items: center; gap: 8px; }
  .nav-links { display: flex; gap: 1.5rem; }
  .nav-links a { color: var(--muted); text-decoration: none; font-size: 0.875rem; transition: color .2s; }
  .nav-links a:hover { color: var(--text); }
  .nav-pill { background: var(--green); color: #000; padding: 3px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 700; }

  /* ── Layout ── */
  .container { max-width: 1100px; margin: 0 auto; padding: 0 1.5rem; }
  section { padding: 4rem 0; }
  section + section { border-top: 1px solid var(--border); }

  /* ── Hero ── */
  .hero { padding: 5rem 0 4rem; text-align: center; }
  .hero-badge { display: inline-flex; align-items: center; gap: 6px; background: rgba(59,130,246,.15); border: 1px solid rgba(59,130,246,.3); color: var(--accent); padding: 4px 14px; border-radius: 20px; font-size: 0.78rem; margin-bottom: 1.5rem; }
  .hero h1 { font-size: clamp(2rem, 5vw, 3.2rem); font-weight: 800; line-height: 1.15; margin-bottom: 1rem; }
  .hero h1 span { background: linear-gradient(135deg, #3b82f6, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .hero-sub { color: var(--muted); font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem; }
  .hero-buttons { display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; }
  .btn { padding: 10px 22px; border-radius: 8px; font-size: 0.9rem; font-weight: 600; text-decoration: none; cursor: pointer; border: none; transition: all .2s; display: inline-flex; align-items: center; gap: 6px; }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-primary:hover { background: #2563eb; transform: translateY(-1px); }
  .btn-outline { background: transparent; color: var(--text); border: 1px solid var(--border); }
  .btn-outline:hover { border-color: var(--accent); color: var(--accent); }

  /* ── Stats row ── */
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 3rem; }
  .stat { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1.2rem; text-align: center; }
  .stat-val { font-size: 1.8rem; font-weight: 800; color: var(--accent); }
  .stat-label { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }

  /* ── Pipeline ── */
  .pipeline { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; align-items: stretch; margin: 2rem 0; }
  .agent-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.4rem 1rem;
    position: relative; transition: border-color .2s, transform .2s;
  }
  .agent-card:hover { border-color: var(--accent); transform: translateY(-2px); }
  .pipeline-arrow {
    display: flex; align-items: center; justify-content: center;
    color: var(--muted); font-size: 1.2rem; flex-shrink: 0;
  }
  .agent-icon { font-size: 2rem; margin-bottom: 0.5rem; }
  .agent-name { font-weight: 700; font-size: 0.9rem; margin-bottom: 0.3rem; }
  .agent-desc { font-size: 0.75rem; color: var(--muted); margin-bottom: 0.7rem; }
  .agent-tools { display: flex; flex-wrap: wrap; gap: 4px; }
  .tool-chip { background: rgba(59,130,246,.12); border: 1px solid rgba(59,130,246,.25); color: #93c5fd; padding: 2px 7px; border-radius: 4px; font-size: 0.65rem; font-family: var(--mono); }
  .agent-sections { font-size: 0.7rem; color: var(--muted); margin-top: 6px; }
  .phase-num { position: absolute; top: 10px; right: 10px; background: var(--purple); color: #fff; width: 20px; height: 20px; border-radius: 50%; font-size: 0.65rem; font-weight: 700; display: flex; align-items: center; justify-content: center; }

  /* ── Reward ── */
  .reward-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
  .reward-card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1.4rem; }
  .reward-card h4 { font-size: 0.85rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; margin-bottom: 1rem; }
  .reward-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,.05); font-size: 0.82rem; }
  .reward-row:last-child { border-bottom: none; }
  .reward-name { color: var(--text); font-family: var(--mono); font-size: 0.78rem; }
  .reward-val { font-weight: 700; color: var(--green); font-size: 0.85rem; }
  .reward-val.neg { color: var(--red); }
  .sub-fn { display: flex; align-items: center; gap: 8px; padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,.05); font-size: 0.82rem; }
  .sub-fn:last-child { border-bottom: none; }
  .fn-letter { width: 22px; height: 22px; border-radius: 4px; background: var(--purple); color: #fff; font-size: 0.7rem; font-weight: 700; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
  .fn-name { font-family: var(--mono); font-size: 0.75rem; color: #93c5fd; flex: 1; }
  .fn-max { color: var(--green); font-weight: 700; font-size: 0.8rem; }

  /* ── Demo ── */
  .demo-grid { display: grid; grid-template-columns: 340px 1fr; gap: 1.5rem; }
  .demo-controls { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1.4rem; }
  .demo-controls h4 { font-size: 0.85rem; text-transform: uppercase; color: var(--muted); letter-spacing: .05em; margin-bottom: 1rem; }
  .form-group { margin-bottom: 1rem; }
  .form-group label { font-size: 0.78rem; color: var(--muted); display: block; margin-bottom: 4px; }
  select, input { width: 100%; background: var(--bg); border: 1px solid var(--border); color: var(--text); padding: 8px 10px; border-radius: 6px; font-size: 0.85rem; }
  select:focus, input:focus { outline: none; border-color: var(--accent); }
  .action-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 0.5rem; }
  .action-btn { background: rgba(59,130,246,.1); border: 1px solid rgba(59,130,246,.2); color: #93c5fd; padding: 7px 6px; border-radius: 6px; font-size: 0.72rem; cursor: pointer; transition: all .2s; font-family: var(--mono); }
  .action-btn:hover { background: rgba(59,130,246,.2); border-color: var(--accent); }
  .action-btn:disabled { opacity: .4; cursor: not-allowed; }
  .demo-output { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1.4rem; display: flex; flex-direction: column; }
  .output-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem; }
  .output-header h4 { font-size: 0.85rem; text-transform: uppercase; color: var(--muted); letter-spacing: .05em; }
  .phase-badge { padding: 3px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 700; background: rgba(124,58,237,.2); color: #c4b5fd; border: 1px solid rgba(124,58,237,.3); }
  .reward-display { font-size: 1.4rem; font-weight: 800; color: var(--green); }
  pre { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; font-family: var(--mono); font-size: 0.75rem; overflow-y: auto; flex: 1; max-height: 420px; white-space: pre-wrap; word-break: break-all; color: #94a3b8; line-height: 1.5; }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); display: inline-block; animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  /* ── API table ── */
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th { text-align: left; padding: 10px 12px; background: var(--card); color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: .05em; border-bottom: 1px solid var(--border); }
  td { padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,.05); vertical-align: top; }
  tr:last-child td { border-bottom: none; }
  code { font-family: var(--mono); background: rgba(59,130,246,.1); color: #93c5fd; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; }
  .method { font-weight: 700; font-family: var(--mono); }
  .method.get { color: var(--green); }
  .method.post { color: var(--yellow); }

  /* ── Links ── */
  .links-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
  .link-card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1.2rem; text-decoration: none; transition: all .2s; display: flex; align-items: flex-start; gap: 12px; }
  .link-card:hover { border-color: var(--accent); transform: translateY(-2px); }
  .link-icon { font-size: 1.5rem; flex-shrink: 0; }
  .link-title { font-weight: 600; font-size: 0.9rem; color: var(--text); }
  .link-desc { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }

  /* ── Section headings ── */
  .section-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: .1em; color: var(--purple); font-weight: 700; margin-bottom: 0.5rem; }
  h2 { font-size: 1.7rem; font-weight: 800; margin-bottom: 0.5rem; }
  .section-sub { color: var(--muted); font-size: 0.95rem; margin-bottom: 2rem; }

  /* ── Responsive ── */
  @media (max-width: 768px) {
    .pipeline { grid-template-columns: 1fr; }
    .pipeline-arrow { display: none; }
    .stats { grid-template-columns: repeat(2, 1fr); }
    .reward-grid, .demo-grid, .links-grid { grid-template-columns: 1fr; }
    .action-grid { grid-template-columns: repeat(3, 1fr); }
  }
</style>
</head>
<body>

<!-- Nav -->
<nav>
  <div class="nav-brand">🛡️ Hiring Fleet <span class="nav-pill">v3.0.0</span></div>
  <div class="nav-links">
    <a href="#pipeline">Pipeline</a>
    <a href="#reward">Reward</a>
    <a href="#demo">Live Demo</a>
    <a href="#api">API</a>
    <a href="/docs">Swagger ↗</a>
  </div>
</nav>

<!-- Hero -->
<section class="hero">
  <div class="container">
    <div class="hero-badge"><span class="status-dot"></span> Environment Online</div>
    <h1>Can four constrained AI specialists<br>catch fraud that <span>fools a single agent?</span></h1>
    <p class="hero-sub">A multi-agent hiring pipeline where each specialist investigates one slice of a resume — and an Overseer synthesises their findings without ever seeing the raw evidence.</p>
    <div class="hero-buttons">
      <a href="#demo" class="btn btn-primary">▶ Try Live Demo</a>
      <a href="/docs" class="btn btn-outline">API Docs ↗</a>
      <a href="https://github.com/Ishika-eng/OpenEnv-Meta-Hackathon---Adversarial-Resume-Screening-Environment" target="_blank" class="btn btn-outline">GitHub ↗</a>
    </div>
    <div class="stats">
      <div class="stat"><div class="stat-val">4</div><div class="stat-label">Specialist Agents</div></div>
      <div class="stat"><div class="stat-val">36</div><div class="stat-label">Resume Episodes</div></div>
      <div class="stat"><div class="stat-val">7</div><div class="stat-label">Reward Sub-functions</div></div>
      <div class="stat"><div class="stat-val">+15.5%</div><div class="stat-label">GRPO Improvement</div></div>
    </div>
  </div>
</section>

<!-- Pipeline -->
<section id="pipeline">
  <div class="container">
    <div class="section-label">Architecture</div>
    <h2>Four-Phase Investigation Pipeline</h2>
    <p class="section-sub">Each agent has a hard-enforced action whitelist and sees only its authorised resume sections. Information is genuinely siloed.</p>

    <div class="pipeline">
      <div class="agent-card">
        <div class="phase-num">1</div>
        <div class="agent-icon">🔍</div>
        <div class="agent-name">Fraud Specialist</div>
        <div class="agent-desc">Detects fabricated credentials and suspicious references</div>
        <div class="agent-tools">
          <span class="tool-chip">verify_credential</span>
          <span class="tool-chip">check_reference</span>
          <span class="tool-chip">view_section</span>
        </div>
        <div class="agent-sections">Sections: header, references</div>
      </div>
      <div class="pipeline-arrow">→</div>
      <div class="agent-card">
        <div class="phase-num">2</div>
        <div class="agent-icon">💡</div>
        <div class="agent-name">Skills Specialist</div>
        <div class="agent-desc">Assesses technical fit against job requirements</div>
        <div class="agent-tools">
          <span class="tool-chip">view_section</span>
          <span class="tool-chip">ask_clarification</span>
        </div>
        <div class="agent-sections">Sections: experience, education, skills, projects</div>
      </div>
      <div class="pipeline-arrow">→</div>
      <div class="agent-card">
        <div class="phase-num">3</div>
        <div class="agent-icon">📅</div>
        <div class="agent-name">Timeline Specialist</div>
        <div class="agent-desc">Checks chronological consistency and employment gaps</div>
        <div class="agent-tools">
          <span class="tool-chip">view_section</span>
          <span class="tool-chip">ask_clarification</span>
        </div>
        <div class="agent-sections">Sections: header, summary, experience</div>
      </div>
      <div class="pipeline-arrow">→</div>
      <div class="agent-card">
        <div class="phase-num">4</div>
        <div class="agent-icon">⚖️</div>
        <div class="agent-name">Overseer</div>
        <div class="agent-desc">Reads all specialist reports and issues final verdict</div>
        <div class="agent-tools">
          <span class="tool-chip">read_reports</span>
          <span class="tool-chip">request_reinvestigation</span>
          <span class="tool-chip">submit_final_decision</span>
        </div>
        <div class="agent-sections">❌ Cannot view resume sections directly</div>
      </div>
    </div>
  </div>
</section>

<!-- Reward -->
<section id="reward">
  <div class="container">
    <div class="section-label">Reward Design</div>
    <h2>7 Independent Sub-functions</h2>
    <p class="section-sub">Dense rewards across all four phases. Terminal reward delegates to seven named functions — each testable and tunable in isolation. No LLM judge required.</p>

    <div class="reward-grid">
      <div class="reward-card">
        <h4>Terminal Reward Sub-functions</h4>
        <div class="sub-fn"><div class="fn-letter">A</div><div class="fn-name">_reward_decision_accuracy</div><div class="fn-max">+0.70</div></div>
        <div class="sub-fn"><div class="fn-letter">B</div><div class="fn-name">_reward_specialist_quality</div><div class="fn-max">+0.22</div></div>
        <div class="sub-fn"><div class="fn-letter">C</div><div class="fn-name">_reward_fleet_coordination</div><div class="fn-max">+0.08</div></div>
        <div class="sub-fn"><div class="fn-letter">D</div><div class="fn-name">_reward_oversight_quality</div><div class="fn-max">+0.08</div></div>
        <div class="sub-fn"><div class="fn-letter">E</div><div class="fn-name">_reward_investigation_quality</div><div class="fn-max">+0.05</div></div>
        <div class="sub-fn"><div class="fn-letter">F</div><div class="fn-name">_reward_format_compliance</div><div class="fn-max">+0.05</div></div>
        <div class="sub-fn"><div class="fn-letter">G</div><div class="fn-name">_reward_step_efficiency</div><div class="fn-max">+0.04</div></div>
      </div>
      <div class="reward-card">
        <h4>Per-Step Rewards &amp; Anti-Exploit Penalties</h4>
        <div class="reward-row"><span class="reward-name">verify_credential → FAILED</span><span class="reward-val">+0.05</span></div>
        <div class="reward-row"><span class="reward-name">check_reference → fraud signal</span><span class="reward-val">+0.05</span></div>
        <div class="reward-row"><span class="reward-name">view_section (high-value)</span><span class="reward-val">+0.03</span></div>
        <div class="reward-row"><span class="reward-name">ask_clarification (substantive)</span><span class="reward-val">+0.03</span></div>
        <div class="reward-row"><span class="reward-name">read_reports (all 3 read)</span><span class="reward-val">+0.03</span></div>
        <div class="reward-row"><span class="reward-name">submit_specialist_report</span><span class="reward-val">up to +0.10</span></div>
        <div class="reward-row"><span class="reward-name">Wrong + overconfident (≥0.7)</span><span class="reward-val neg">−0.05</span></div>
        <div class="reward-row"><span class="reward-name">fraud_flag=True, empty reasoning</span><span class="reward-val neg">−0.05</span></div>
        <div class="reward-row"><span class="reward-name">Fraud resume, zero tools used</span><span class="reward-val neg">−0.05</span></div>
        <div class="reward-row"><span class="reward-name">Out-of-role action violation</span><span class="reward-val neg">−0.05 each</span></div>
      </div>
    </div>
  </div>
</section>

<!-- Live Demo -->
<section id="demo">
  <div class="container">
    <div class="section-label">Interactive Demo</div>
    <h2>Try the Environment</h2>
    <p class="section-sub">Run a live episode against the deployed environment. Start with Reset, then step through each agent phase.</p>

    <div class="demo-grid">
      <div class="demo-controls">
        <h4>Controls</h4>
        <div class="form-group">
          <label>Difficulty</label>
          <select id="taskType">
            <option value="easy">Easy — Obvious fraud (8 steps)</option>
            <option value="medium" selected>Medium — Subtle fraud (11 steps)</option>
            <option value="hard">Hard — Sophisticated fraud (15 steps)</option>
          </select>
        </div>
        <div class="form-group">
          <label>Seed</label>
          <input type="number" id="seed" value="42" min="0" max="999">
        </div>
        <button class="btn btn-primary" style="width:100%;margin-bottom:1rem;" onclick="resetEpisode()">↺ Reset Episode</button>

        <h4 style="margin-bottom:0.6rem;">Actions</h4>
        <div class="action-grid">
          <button class="action-btn" onclick="step({action_type:'verify_credential'})" id="btn-verify">verify_credential</button>
          <button class="action-btn" onclick="step({action_type:'check_reference',reference_id:'ref2'})" id="btn-ref2">check_reference ref2</button>
          <button class="action-btn" onclick="step({action_type:'check_reference',reference_id:'ref1'})" id="btn-ref1">check_reference ref1</button>
          <button class="action-btn" onclick="step({action_type:'view_section',section:'experience'})" id="btn-exp">view experience</button>
          <button class="action-btn" onclick="step({action_type:'view_section',section:'education'})" id="btn-edu">view education</button>
          <button class="action-btn" onclick="step({action_type:'view_section',section:'skills'})" id="btn-skills">view skills</button>
          <button class="action-btn" onclick="step({action_type:'view_section',section:'header'})" id="btn-header">view header</button>
          <button class="action-btn" onclick="step({action_type:'view_section',section:'references'})" id="btn-refs">view references</button>
          <button class="action-btn" onclick="step({action_type:'ask_clarification',question:'Can you clarify your employment dates?'})" id="btn-clarify">ask_clarification</button>
          <button class="action-btn" onclick="submitSpecialistReport()" id="btn-submit-spec">submit_specialist_report</button>
          <button class="action-btn" onclick="step({action_type:'read_reports',report_target:'fraud_specialist'})" id="btn-read-fraud">read fraud report</button>
          <button class="action-btn" onclick="step({action_type:'read_reports',report_target:'skills_specialist'})" id="btn-read-skills">read skills report</button>
          <button class="action-btn" onclick="step({action_type:'read_reports',report_target:'timeline_specialist'})" id="btn-read-timeline">read timeline report</button>
          <button class="action-btn" onclick="submitFinalDecision('reject')" id="btn-reject">submit reject</button>
          <button class="action-btn" onclick="submitFinalDecision('accept')" id="btn-accept">submit accept</button>
        </div>
      </div>

      <div class="demo-output">
        <div class="output-header">
          <h4>Observation</h4>
          <div style="display:flex;align-items:center;gap:12px;">
            <span id="phase-badge" class="phase-badge">—</span>
            <span id="reward-display" class="reward-display">—</span>
          </div>
        </div>
        <div style="display:flex;gap:1rem;margin-bottom:0.8rem;font-size:0.8rem;color:var(--muted);">
          <span>Phase steps: <strong id="steps-left" style="color:var(--text)">—</strong></span>
          <span>Total left: <strong id="total-left" style="color:var(--text)">—</strong></span>
          <span>Violations: <strong id="violations" style="color:var(--red)">—</strong></span>
        </div>
        <pre id="obs-output">Click "↺ Reset Episode" to start a new episode.</pre>
      </div>
    </div>
  </div>
</section>

<!-- API -->
<section id="api">
  <div class="container">
    <div class="section-label">API Reference</div>
    <h2>Endpoints</h2>
    <p class="section-sub">All endpoints return JSON. The full OpenAPI spec is available at <a href="/docs" style="color:var(--accent)">/docs</a>.</p>

    <table>
      <thead>
        <tr><th>Method</th><th>Path</th><th>Description</th><th>Key fields</th></tr>
      </thead>
      <tbody>
        <tr>
          <td><span class="method post">POST</span></td>
          <td><code>/reset</code></td>
          <td>Start a new episode</td>
          <td><code>task_type</code>, <code>seed</code></td>
        </tr>
        <tr>
          <td><span class="method post">POST</span></td>
          <td><code>/step</code></td>
          <td>Submit an action, receive next observation + reward</td>
          <td><code>action_type</code> + role-specific fields</td>
        </tr>
        <tr>
          <td><span class="method get">GET</span></td>
          <td><code>/state</code></td>
          <td>Current episode state (read-only)</td>
          <td>—</td>
        </tr>
        <tr>
          <td><span class="method get">GET</span></td>
          <td><code>/health</code></td>
          <td>Health check</td>
          <td>—</td>
        </tr>
        <tr>
          <td><span class="method get">GET</span></td>
          <td><code>/docs</code></td>
          <td>Interactive Swagger UI</td>
          <td>—</td>
        </tr>
      </tbody>
    </table>
  </div>
</section>

<!-- Links -->
<section>
  <div class="container">
    <div class="section-label">Resources</div>
    <h2>Materials</h2>
    <p class="section-sub">Everything you need to run, train, and extend the Hiring Fleet environment.</p>

    <div class="links-grid">
      <a href="https://github.com/Ishika-eng/OpenEnv-Meta-Hackathon---Adversarial-Resume-Screening-Environment" target="_blank" class="link-card">
        <div class="link-icon">💻</div>
        <div><div class="link-title">GitHub Repository</div><div class="link-desc">Full source code, dataset, and evaluation scripts</div></div>
      </a>
      <a href="/docs" class="link-card">
        <div class="link-icon">📖</div>
        <div><div class="link-title">API Documentation</div><div class="link-desc">Interactive Swagger UI — try every endpoint in your browser</div></div>
      </a>
      <a href="https://huggingface.co/spaces/IshikaMahadar/resume-env" target="_blank" class="link-card">
        <div class="link-icon">🤗</div>
        <div><div class="link-title">HuggingFace Space</div><div class="link-desc">Live deployed environment — no setup required</div></div>
      </a>
    </div>
  </div>
</section>

<!-- Footer -->
<footer style="text-align:center;padding:2rem;color:var(--muted);font-size:0.8rem;border-top:1px solid var(--border);">
  Hiring Fleet v3.0.0 &nbsp;·&nbsp; OpenEnv Hackathon &nbsp;·&nbsp; Built with HuggingFace TRL + GRPO
</footer>

<script>
  const BASE = window.location.origin;

  async function resetEpisode() {
    const taskType = document.getElementById('taskType').value;
    const seed = parseInt(document.getElementById('seed').value) || 42;
    setOutput('Resetting episode...');
    try {
      const r = await fetch(`${BASE}/reset`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({task_type: taskType, seed})
      });
      const data = await r.json();
      const obs = data.observation || data;
      updateUI(obs, data.reward ?? 0);
    } catch(e) {
      setOutput(`Error: ${e.message}`);
    }
  }

  async function step(action) {
    setOutput('Sending action...');
    try {
      const r = await fetch(`${BASE}/step`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(action)
      });
      const data = await r.json();
      const obs = data.observation || data;
      updateUI(obs, data.reward ?? obs.reward ?? 0);
    } catch(e) {
      setOutput(`Error: ${e.message}`);
    }
  }

  function submitSpecialistReport() {
    step({
      action_type: 'submit_specialist_report',
      findings: 'Investigation complete. Credential verification and references checked.',
      has_issues: false,
      specialist_confidence: 0.75
    });
  }

  function submitFinalDecision(decision) {
    const isFraud = decision === 'reject';
    step({
      action_type: 'submit_final_decision',
      decision,
      fraud_flag: isFraud,
      confidence: 0.80,
      fraud_reasoning: isFraud ? 'Credential verification failed and reference denied employment.' : ''
    });
  }

  function updateUI(obs, reward) {
    const phase = obs.current_phase || '—';
    document.getElementById('phase-badge').textContent = phase.replace(/_/g, ' ');
    const rVal = typeof reward === 'number' ? reward : (obs.reward ?? 0);
    const rDisplay = document.getElementById('reward-display');
    rDisplay.textContent = `+${rVal.toFixed(4)}`;
    rDisplay.style.color = rVal > 0 ? 'var(--green)' : rVal < 0 ? 'var(--red)' : 'var(--muted)';
    document.getElementById('steps-left').textContent = obs.steps_remaining ?? '—';
    document.getElementById('total-left').textContent = obs.total_steps_remaining ?? '—';
    document.getElementById('violations').textContent = obs.violations_count ?? 0;

    // Show clean observation
    const display = {
      current_phase: obs.current_phase,
      role_instructions: obs.role_instructions ? obs.role_instructions.slice(0, 120) + '…' : undefined,
      available_actions: obs.available_actions,
      steps_remaining: obs.steps_remaining,
      total_steps_remaining: obs.total_steps_remaining,
      violations_count: obs.violations_count,
      visible_sections: obs.visible_sections && Object.keys(obs.visible_sections).length
        ? Object.fromEntries(Object.entries(obs.visible_sections).map(([k,v]) => [k, v?.slice(0,80)+'…']))
        : undefined,
      specialist_reports: obs.specialist_reports?.length ? obs.specialist_reports.map(r => ({
        role: r.specialist_role, has_issues: r.has_issues, confidence: r.confidence,
        findings: r.findings?.slice(0,80)+'…'
      })) : undefined,
      reference_response: obs.reference_response || undefined,
      verification_result: obs.verification_result || undefined,
      clarification_response: obs.clarification_response || undefined,
      read_report_details: obs.read_report_details && Object.keys(obs.read_report_details).length
        ? obs.read_report_details : undefined,
      feedback: obs.feedback,
      reward: rVal,
      done: obs.done,
    };
    // Remove undefined keys
    Object.keys(display).forEach(k => display[k] === undefined && delete display[k]);
    setOutput(JSON.stringify(display, null, 2));
  }

  function setOutput(text) {
    document.getElementById('obs-output').textContent = text;
  }
</script>
</body>
</html>"""


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


def main():
    """Entry point for the fleet environment server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
