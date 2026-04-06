# RL-Documentation-Parser

A Reinforcement Learning environment built with Gymnasium where an agent learns to navigate a mock file system to answer user queries using technical documentation.

## What the Environment Does

The `DocParserEnv` simulates an agent navigating a file system to answer technical questions - simulating a developer searching through documentation:

1. **Task**: Given a user query about technical documentation, the agent must explore directories, read relevant .md files, and submit the correct answer
2. **Actions**: The agent can perform 4 actions:
   - `ls` (0) - List current directory contents
   - `cd` (1) - Change to a subdirectory
   - `read` (2) - Open and extract text from a file
   - `submit_answer` (3) - Submit the final answer to end the episode
3. **Rewards**:
   - +100: Correct answer submitted (matches ground truth)
   - -1: Per-step penalty (encourages shortest path to solution)
   - -5: Invalid action (e.g., cd into a file, read a folder)
   - -20: Wrong answer submitted

## Project Structure

```
RL-Documentation-Parser/
├── rl_doc_parser.py       # Main script with DocParserEnv and training loop
├── qa_pairs.json          # Ground truth QA pairs (5 examples)
├── root/                  # Mock file system with documentation
│   ├── network/          # network/ssl_config.md, protocols.md
│   ├── db/               # db/postgres_setup.md, optimization.md
│   └── api/              # api/oauth_guide.md
├── requirements.txt      # Python dependencies
└── LICENSE               # MIT License
```

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

```bash
# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## How to Run the Demo

```bash
python rl_doc_parser.py
```

This runs a 100-step training loop demonstrating the environment:
- Shows user queries being posed
- Displays actions taken and rewards received
- Outputs episode statistics at completion

## Important Setup Steps

1. **Virtual Environment**: Always activate `venv` before running
2. **Working Directory**: Run from the project root (`RL-Documentation-Parser/`)
3. **File System**: The `root/` directory contains mock documentation - do not modify during execution
4. **Offline Execution**: The demo runs entirely offline with random actions

## Environment Details

- **Observation Space**: Dict containing current_path, dir_contents, user_query
- **Action Space**: Discrete(4)
- **Max Steps per Episode**: 20
- **Ground Truth**: 5 QA pairs stored in qa_pairs.json

