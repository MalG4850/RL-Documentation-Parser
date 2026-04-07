"""
inference.py — Baseline inference script
Uses the OpenAI API to run a model against DocParserEnv.
Reads credentials from environment variables.
Produces reproducible baseline scores on all 3 tasks.

Usage:
    export OPENAI_API_KEY=sk-...
    export MODEL_NAME=gpt-4o-mini        # optional, defaults to gpt-4o-mini
    export HF_TOKEN=hf_...              # optional, for HuggingFace logging
    python inference.py
"""

import json
import os
import random
import sys
from pathlib import Path

# ── Optional: load .env if present ────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

from rl_doc_parser import DocParserEnv, ACTIONS

# ── Config ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
SEED = 42
MAX_STEPS = 20
NUM_EPISODES_PER_TASK = 3  # run each task 3 times for reproducibility

if not OPENAI_API_KEY:
    print("[ERROR] OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)


# ── LLM Policy ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI agent navigating a mock file system to answer technical questions.
You must choose one action per turn.

Available actions (reply with ONLY the JSON object, nothing else):
  {"action": 0}                          — ls: list current directory
  {"action": 1, "target": "<dirname>"}   — cd: enter a directory
  {"action": 2, "target": "<filename>"}  — read: read a .md file
  {"action": 3, "target": "<answer>"}    — submit_answer: submit your final answer

Strategy:
1. Run ls to see what directories exist.
2. cd into the directory most likely to contain the answer.
3. ls again to see files.
4. read the relevant .md file.
5. submit_answer with the extracted answer.

Always output valid JSON and nothing else."""


def llm_action(obs: dict, history: list[dict]) -> tuple[int, str | None]:
    """Ask the LLM to pick the next action given the observation."""
    user_message = (
        f"Current path: {obs['current_path']}\n"
        f"Directory contents: {obs['dir_contents']}\n"
        f"User query: {obs['user_query']}\n\n"
        "What action do you take?"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-6:])  # keep last 3 turns for context
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=64,
    )

    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
        action = int(parsed.get("action", 0))
        target = parsed.get("target", None)
        return action, target
    except (json.JSONDecodeError, ValueError):
        print(f"  [WARN] Could not parse LLM response: {raw!r} — defaulting to ls")
        return 0, None


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(env: DocParserEnv, episode_num: int, seed: int) -> dict:
    random.seed(seed)
    obs, info = env.reset(seed=seed)

    print(f"\n  Episode {episode_num} | Query: {obs['user_query']!r}")

    history = []
    total_reward = 0.0
    success = False

    for step_i in range(MAX_STEPS):
        action, target = llm_action(obs, history)
        action_name = ACTIONS.get(action, "unknown")

        # Inject target before stepping
        env.target_dir = target or ""

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        step_summary = (
            f"  Step {step_i+1}: {action_name}"
            + (f"({target})" if target else "")
            + f" → reward={reward}"
        )
        print(step_summary)

        # Update conversation history for LLM context
        history.append({"role": "assistant", "content": json.dumps({"action": action, "target": target})})
        history.append({"role": "user", "content": f"Result: {info.get('last_action_result', '')} | Reward: {reward}"})

        if reward == 100:
            success = True

        if done or truncated:
            break

    status = "✅ SUCCESS" if success else "❌ FAIL"
    score = max(0.0, min(1.0, (total_reward + 20) / 120))  # normalise to [0, 1]
    print(f"  {status} | Total reward: {total_reward:.1f} | Score: {score:.3f}")

    return {
        "episode": episode_num,
        "query": obs["user_query"],
        "total_reward": total_reward,
        "score": score,
        "success": success,
        "steps": step_i + 1,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"RL Documentation Parser — Baseline Inference")
    print(f"Model: {MODEL_NAME}")
    print(f"Seed:  {SEED}")
    print("=" * 60)

    env = DocParserEnv(max_steps=MAX_STEPS)
    results = []

    for ep in range(1, NUM_EPISODES_PER_TASK + 1):
        result = run_episode(env, ep, seed=SEED + ep)
        results.append(result)

    env.close()

    # ── Summary ────────────────────────────────────────────────────────────────
    total_episodes = len(results)
    successes = sum(1 for r in results if r["success"])
    avg_reward = sum(r["total_reward"] for r in results) / total_episodes
    avg_score = sum(r["score"] for r in results) / total_episodes

    print("\n" + "=" * 60)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 60)
    print(f"Episodes:        {total_episodes}")
    print(f"Successes:       {successes}/{total_episodes} ({successes/total_episodes*100:.0f}%)")
    print(f"Avg reward:      {avg_reward:.2f}")
    print(f"Avg score:       {avg_score:.3f}  (range 0.0–1.0)")
    print("=" * 60)

    # Save results to JSON for reproducibility
    output_path = Path("baseline_results.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "seed": SEED,
                "episodes": results,
                "summary": {
                    "total_episodes": total_episodes,
                    "successes": successes,
                    "success_rate": successes / total_episodes,
                    "avg_reward": avg_reward,
                    "avg_score": avg_score,
                },
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")
    return avg_score


if __name__ == "__main__":
    main()
