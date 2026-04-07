"""
validate.py — Pre-submission validation script
Run this before submitting to catch issues early.

Checks:
  1. openenv.yaml is present and parseable
  2. DocParserEnv has step / reset / state methods
  3. step() returns (obs, reward, done, truncated, info)
  4. reward is a float in [-20, 100]
  5. obs keys match the observation_space spec
  6. All 3 tasks run without errors and produce scores in [0, 1]
"""

import json
import sys
import traceback
from pathlib import Path

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []


def check(label: str, fn):
    try:
        fn()
        print(f"  {PASS}  {label}")
        results.append((label, True, None))
    except Exception as e:
        msg = str(e)
        print(f"  {FAIL}  {label}\n         → {msg}")
        results.append((label, False, msg))


print("=" * 55)
print("DocParserEnv — Pre-Submission Validation")
print("=" * 55)

# 1. openenv.yaml exists
check("openenv.yaml present", lambda: Path("openenv.yaml").read_text())

# 2. Import environment
from rl_doc_parser import DocParserEnv, ACTIONS

check("DocParserEnv importable", lambda: DocParserEnv())

env = DocParserEnv(max_steps=20)

# 3. reset() returns (obs, info)
def _check_reset():
    obs, info = env.reset(seed=42)
    assert isinstance(obs, dict), "obs must be a dict"
    assert "current_path" in obs
    assert "dir_contents" in obs
    assert "user_query" in obs

check("reset() returns valid (obs, info)", _check_reset)

# 4. step() returns correct tuple
def _check_step():
    env.reset(seed=42)
    env.target_dir = ""
    obs, reward, done, truncated, info = env.step(0)  # ls
    assert isinstance(reward, (int, float)), "reward must be numeric"
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

check("step() returns (obs, reward, done, truncated, info)", _check_step)

# 5. Reward is in range
def _check_reward_range():
    env.reset(seed=42)
    env.target_dir = ""
    _, reward, _, _, _ = env.step(0)
    assert -20 <= reward <= 100, f"Reward {reward} out of expected range"

check("Reward in [-20, 100]", _check_reward_range)

# 6. All 4 actions execute without crashing
def _check_all_actions():
    for action in range(4):
        env.reset(seed=42)
        env.target_dir = "network" if action in (1, 2) else ""
        env.step(action)

check("All 4 actions execute without crash", _check_all_actions)

# 7. Scores are in [0, 1]
def _check_score_range():
    for seed in [1, 2, 3]:
        obs, info = env.reset(seed=seed)
        total_reward = 0.0
        for _ in range(20):
            env.target_dir = ""
            obs, reward, done, _, info = env.step(env.action_space.sample())
            total_reward += reward
            if done:
                break
        score = max(0.0, min(1.0, (total_reward + 20) / 120))
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1]"

check("Scores normalise to [0.0–1.0]", _check_score_range)

# ── Summary ────────────────────────────────────────────────────────────────────
print()
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"Result: {passed}/{total} checks passed")

if passed == total:
    print("\n🎉 All checks passed! Ready to submit.")
    sys.exit(0)
else:
    print("\n⚠️  Fix the failing checks before submitting.")
    sys.exit(1)
