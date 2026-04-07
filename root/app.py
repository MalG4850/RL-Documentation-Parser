"""
app.py — FastAPI wrapper for DocParserEnv
Exposes the OpenEnv-compatible step/reset/state API over HTTP.
Runs on port 7860 for HuggingFace Spaces.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rl_doc_parser import DocParserEnv, ACTIONS

app = FastAPI(
    title="RL Documentation Parser — OpenEnv API",
    description="A Gymnasium environment where an agent navigates a mock filesystem to answer technical queries.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global env instance (one session; extend with session IDs for multi-user) ──
env = DocParserEnv(max_steps=20)
_obs, _info = env.reset()
_current_obs = _obs
_current_info = _info
_episode_reward = 0.0
_done = False


# ── Pydantic models ────────────────────────────────────────────────────────────

class StepRequest(BaseModel):
    action: int  # 0=ls, 1=cd, 2=read, 3=submit_answer
    target: Optional[str] = None  # directory/file name for cd/read/submit


class ResetRequest(BaseModel):
    seed: Optional[int] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_state_response(obs, info, reward=None, done=False):
    return {
        "observation": obs,
        "info": info,
        "reward": reward,
        "done": done,
        "action_space": {"type": "Discrete", "n": 4},
        "actions": ACTIONS,
        "episode_reward": _episode_reward,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    """Simple landing page with links to the API docs."""
    return """
    <html>
      <head><title>RL Documentation Parser</title></head>
      <body style="font-family:sans-serif;max-width:700px;margin:40px auto;padding:0 20px">
        <h1>🤖 RL Documentation Parser</h1>
        <p>
          An OpenEnv-compatible Gymnasium environment where an agent learns to navigate
          a mock file system to answer user queries using technical documentation.
        </p>
        <h2>Quick Links</h2>
        <ul>
          <li><a href="/docs">📖 Interactive API Docs (Swagger UI)</a></li>
          <li><a href="/redoc">📚 ReDoc</a></li>
          <li><a href="/state">🔍 Current Environment State</a></li>
          <li><a href="/openenv.yaml">📄 openenv.yaml metadata</a></li>
        </ul>
        <h2>Actions</h2>
        <ul>
          <li><b>0 — ls</b>: List current directory contents</li>
          <li><b>1 — cd</b>: Change to a subdirectory (provide <code>target</code>)</li>
          <li><b>2 — read</b>: Read a .md file (provide <code>target</code>)</li>
          <li><b>3 — submit_answer</b>: Submit final answer (provide <code>target</code> as answer text)</li>
        </ul>
        <h2>Rewards</h2>
        <ul>
          <li>+100 Correct answer submitted</li>
          <li>-1 Per step (efficiency penalty)</li>
          <li>-5 Invalid action</li>
          <li>-20 Wrong answer submitted</li>
        </ul>
      </body>
    </html>
    """


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    """Reset the environment and return the initial observation."""
    global _current_obs, _current_info, _episode_reward, _done
    _current_obs, _current_info = env.reset(seed=req.seed)
    _episode_reward = 0.0
    _done = False
    return _build_state_response(_current_obs, _current_info, reward=None, done=False)


@app.post("/step")
def step(req: StepRequest):
    """Take a step in the environment."""
    global _current_obs, _current_info, _episode_reward, _done

    if _done:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call POST /reset to start a new episode.",
        )

    if req.action not in ACTIONS:
        raise HTTPException(status_code=400, detail=f"Invalid action {req.action}. Must be 0–3.")

    # Inject the target into the env before stepping
    if req.target is not None:
        env.target_dir = req.target
    else:
        env.target_dir = ""

    obs, reward, done, truncated, info = env.step(req.action)
    _current_obs = obs
    _current_info = info
    _episode_reward += reward
    _done = done or truncated

    return _build_state_response(obs, info, reward=reward, done=_done)


@app.get("/state")
def state():
    """Return the current environment state without taking an action."""
    return _build_state_response(_current_obs, _current_info, reward=None, done=_done)


@app.get("/openenv.yaml")
def openenv_yaml():
    """Return openenv.yaml metadata for validator compliance."""
    from fastapi.responses import PlainTextResponse
    yaml_content = """
name: rl-documentation-parser
version: "1.0.0"
description: >
  A Gymnasium environment where an agent navigates a mock filesystem
  to answer user queries using technical documentation files.
author: MalG4850
observation_space:
  type: Dict
  keys:
    current_path:
      type: Text
      max_length: 512
    dir_contents:
      type: Text
      max_length: 2048
    user_query:
      type: Text
      max_length: 512
action_space:
  type: Discrete
  n: 4
  actions:
    0: ls
    1: cd
    2: read
    3: submit_answer
reward_range: [-20, 100]
max_steps: 20
tasks:
  - name: find_ssl_config
    difficulty: easy
    description: Locate SSL configuration details
  - name: find_postgres_setup
    difficulty: medium
    description: Find PostgreSQL setup instructions
  - name: find_oauth_guide
    difficulty: hard
    description: Retrieve OAuth authentication steps
""".strip()
    return PlainTextResponse(yaml_content, media_type="text/yaml")


@app.get("/tasks")
def list_tasks():
    """List the available tasks with their difficulties."""
    return {
        "tasks": [
            {"id": "find_ssl_config", "difficulty": "easy", "description": "Locate SSL configuration details"},
            {"id": "find_postgres_setup", "difficulty": "medium", "description": "Find PostgreSQL setup instructions"},
            {"id": "find_oauth_guide", "difficulty": "hard", "description": "Retrieve OAuth authentication steps"},
        ]
    }


@app.get("/health")
def health():
    return {"status": "ok"}
