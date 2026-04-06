import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR / "root"
QA_PAIRS_PATH = BASE_DIR / "qa_pairs.json"

with open(QA_PAIRS_PATH) as f:
    QA_PAIRS = json.load(f)

ACTIONS = {0: "ls", 1: "cd", 2: "read", 3: "submit_answer"}


def load_text_file(path: Path) -> str:
    with open(path) as f:
        return f.read()


def call_policy_model(observation: Dict[str, Any], qa_pairs: List[Dict]) -> int:
    prompt = f"""You are an AI agent navigating a file system to answer technical questions.

Current state:
- Current path: {observation['current_path']}
- Directory contents: {observation['dir_contents']}
- User query: {observation['user_query']}

Available actions:
- 0: ls (list directory contents)
- 1: cd <target> (change directory)
- 2: read  (read a file)
- 3: submit_answer <text> (submit your final answer)

Based on the user query, which directory likely contains the answer? Think step by step.
Pick an action: """

    response = f"[LLM would reason here based on user_query] Action: ls"
    return random.randint(0, 3)


def call_tool_model(file_content: str, user_query: str) -> str:
    prompt = f"""Given the following file content and user query, provide the answer.

File content:
{file_content}

User query: {user_query}

If the answer is found in the file content, return it. Otherwise, return "Not Found"."""

    return f"[LLM would extract answer from file content based on query]"


class DocParserEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps: int = 20):
        super().__init__()
        self.max_steps = max_steps
        self.current_path = ROOT_DIR
        self.user_query = ""
        self.ground_truth_answer = ""
        self.ground_truth_file = ""
        self.step_count = 0
        self.last_action = None
        self.last_action_result = ""
        self.done = False
        self.target_dir = ""

        self.observation_space = spaces.Dict({
            "current_path": spaces.Text(max_length=512),
            "dir_contents": spaces.Text(max_length=2048),
            "user_query": spaces.Text(max_length=512),
        })
        self.action_space = spaces.Discrete(4)

    def _get_obs(self) -> Dict[str, Any]:
        try:
            contents = list(self.current_path.iterdir())
            dir_names = [c.name for c in contents]
        except Exception:
            dir_names = []
        
        return {
            "current_path": str(self.current_path),
            "dir_contents": ", ".join(dir_names) if dir_names else "empty",
            "user_query": self.user_query,
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step_count": self.step_count,
            "last_action": self.last_action,
            "last_action_result": self.last_action_result,
            "ground_truth": self.ground_truth_answer,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        
        qa_pair = random.choice(QA_PAIRS)
        self.user_query = qa_pair["query"]
        self.ground_truth_answer = qa_pair["answer"]
        self.ground_truth_file = qa_pair["file_path"]
        
        self.current_path = ROOT_DIR
        self.step_count = 0
        self.done = False
        self.last_action = None
        self.last_action_result = ""
        self.target_dir = ""
        
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        self.step_count += 1
        self.last_action = ACTIONS[action]
        reward = -1
        
        if action == 0:
            try:
                contents = list(self.current_path.iterdir())
                self.last_action_result = f"Contents: {[c.name for c in contents]}"
            except Exception as e:
                self.last_action_result = f"Error: {str(e)}"
                reward = -5
        
        elif action == 1:
            target = self.target_dir
            if not target:
                self.last_action_result = "No target specified"
                reward = -5
            else:
                new_path = self.current_path / target
                if new_path.exists() and new_path.is_dir():
                    self.current_path = new_path
                    self.last_action_result = f"Changed to: {new_path}"
                else:
                    self.last_action_result = f"Invalid directory: {target}"
                    reward = -5
        
        elif action == 2:
            target = self.target_dir
            if not target:
                self.last_action_result = "No file specified"
                reward = -5
            else:
                file_path = self.current_path / target
                if file_path.exists() and file_path.is_file():
                    content = load_text_file(file_path)
                    extracted = call_tool_model(content, self.user_query)
                    self.last_action_result = f"Read file: {target}"
                else:
                    self.last_action_result = f"Cannot read: {target}"
                    reward = -5
        
        elif action == 3:
            answer = self.last_action_result
            if answer.lower().strip() == self.ground_truth_answer.lower().strip():
                reward = 100
                self.done = True
            else:
                reward = -20
                self.done = True
        
        if self.step_count >= self.max_steps:
            self.done = True
        
        return self._get_obs(), reward, self.done, False, self._get_info()

    def render(self):
        print(f"Path: {self.current_path}")
        print(f"Query: {self.user_query}")
        print(f"Step: {self.step_count}")

    def close(self):
        pass


def run_training_loop(env: gym.Env, num_steps: int = 100):
    print(f"Starting training for {num_steps} steps...")
    print("=" * 50)
    
    total_reward = 0
    episode_count = 0
    successful_episodes = 0
    
    obs, info = env.reset()
    print(f"\nEpisode {episode_count + 1}")
    print(f"User Query: {obs['user_query']}")
    print(f"Ground Truth: {info['ground_truth']}")
    print(f"Start Path: {obs['current_path']}")
    
    for step in range(num_steps):
        action = env.action_space.sample()
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:
            print(f"  Step {step + 1}: Action={ACTIONS[action]}, Reward={reward}, Done={done}")
        
        if done:
            episode_count += 1
            
            if reward == 100:
                successful_episodes += 1
                print(f"  Episode {episode_count + 1} SUCCESS! Reward: {reward}")
            else:
                print(f"  Episode {episode_count + 1} ended. Reward: {reward}")
            
            obs, info = env.reset()
            print(f"\nEpisode {episode_count + 2}")
            print(f"User Query: {obs['user_query']}")
            print(f"Ground Truth: {info['ground_truth']}")
    
    print("=" * 50)
    print(f"Training complete!")
    print(f"Total steps: {num_steps}")
    print(f"Total reward: {total_reward}")
    print(f"Episodes: {episode_count}")
    print(f"Successful: {successful_episodes}")
    print(f"Success rate: {successful_episodes / max(1, episode_count) * 100:.1f}%")


def main():
    env = DocParserEnv(max_steps=20)
    run_training_loop(env, num_steps=100)
    env.close()


if __name__ == "__main__":
    main()