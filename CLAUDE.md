# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Behavioral study investigating Theory of Mind (ToM) in LLMs. Tests whether models track protagonist beliefs vs reality using explicit belief statements.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run behavioral study
python -m src.run_study --behavioral-only

# Run with specific model
python -m src.run_study --model "Qwen/Qwen2.5-3B-Instruct"

# Start dashboard (port 8016)
nohup python server_watchdog.py > /tmp/MechInt_watchdog.log 2>&1 &

# Stop server
pkill -f "server_watchdog.py" && pkill -f "web_dashboard.py"

# Run tests
python -m pytest tests/ -v

# Generate tasks (80 total: 10 families × 8 rows, outputs tasks.json + tasks.md)
python -m src.task_generator
```

## Architecture

### Core Pipeline
1. **Task Generation** (`src/task_generator.py`) - Creates false/true belief tasks with explicit belief statements
2. **Model Execution** (`src/model_runner.py`) - Runs HuggingFace model for behavioral analysis
3. **Dashboard** (`web_dashboard.py`) - Visualizes results on port 8016

### Data Flow
```
tasks.json → model_runner → web_dashboard
                 ↓
         results/behavioral/
```

### Task Format
```
False belief: "A ball is in the blue box. Anna believes the ball is in the red box."
True belief:  "A ball is in the blue box. Anna believes the ball is in the blue box."
Question: "Where will Anna look for the ball?"
```

## Key Files

- `src/task_generator.py` - ToMTask dataclass, 8-row truth table (tb1-tb4, fb1-fb4), 50/50 TB/FB
- `src/model_runner.py` - HF model loading for inference
- `src/behavioral_analysis.py` - Response parsing and accuracy metrics
- `web_dashboard.py` - HTTP server with visualizations

## Configuration

- `.env` - Server port (8016), model name, study parameters
- `tasks.json` - Generated task definitions
- `models.json` - Available models configuration

## Critical Rules

**Commit after refactoring**: Always commit and push after major refactoring or feature additions.

**Port**: This project uses port **8016** (configured in `.env`).

## Post-Run Checklist

**After each behavioral analysis run:**

1. Review errors at `http://127.0.0.1:8016/tasks` (filter by "Wrong")
2. Check if parsing can be improved in `parse_response()` in `src/behavioral_analysis.py`
3. Re-run behavioral analysis after parsing improvements to get accurate metrics
