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
├── app.py                 # FastAPI wrapper (HuggingFace Spaces entry point)
├── rl_doc_parser.py       # DocParserEnv + training loop
├── inference.py           # Baseline inference script (OpenAI API)
├── validate.py            # Pre-submission validation script
├── openenv.yaml           # OpenEnv spec metadata
├── qa_pairs.json          # Ground truth QA pairs
├── Dockerfile             # Docker config for HuggingFace Spaces
├── requirements.txt       # Python dependencies
└── root/                  # Mock filesystem
    ├── network/
    │   ├── ssl_config.md
    │   └── protocols.md
    ├── db/
    │   ├── postgres_setup.md
    │   └── optimization.md
    └── api/
        └── oauth_guide.md
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

## API Endpoints
 
The environment is served via FastAPI on port 7860:
 
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Landing page |
| `POST` | `/reset` | Reset the environment |
| `POST` | `/step` | Take a step |
| `GET` | `/state` | Current state |
| `GET` | `/openenv.yaml` | OpenEnv metadata |
| `GET` | `/tasks` | List tasks |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |
 
### Example: Run an Episode via curl
 
```bash
# Reset
curl -X POST https://<your-space>.hf.space/reset
 
# Take an action (ls = 0)
curl -X POST https://<your-space>.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": 0}'
 
# cd into network/
curl -X POST https://<your-space>.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": 1, "target": "network"}'
 
# Read a file
curl -X POST https://<your-space>.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": 2, "target": "ssl_config.md"}'
 
# Submit answer
curl -X POST https://<your-space>.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": 3, "target": "TLS 1.3"}'
```
## Local Development
 
```bash
# 1. Clone
git clone https://github.com/MalG4850/RL-Documentation-Parser.git
cd RL-Documentation-Parser
 
# 2. Install deps
pip install -r requirements.txt
 
# 3. Run the API server
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
 
# 4. Or run the original demo
python rl_doc_parser.py
```
 
## Docker
 
```bash
# Build
docker build -t rl-doc-parser .
 
# Run
docker run -p 7860:7860 rl-doc-parser
```
 
## Baseline Inference
 
```bash
export OPENAI_API_KEY=sk-...
python inference.py
```
 
Results are saved to `baseline_results.json`.
 
## Validation
 
```bash
python validate.py
```
 
## Tasks
 
| ID | Difficulty | Description |
|----|-----------|-------------|
| `find_ssl_config` | Easy | Locate SSL/TLS configuration details |
| `find_postgres_setup` | Medium | Find PostgreSQL setup instructions |
| `find_oauth_guide` | Hard | Retrieve OAuth 2.0 authentication steps |

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

