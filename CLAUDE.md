# CLAUDE.md
# Project Overview

Behavioral study investigating Theory of Mind (ToM) in LLMs. Tests whether models track protagonist beliefs vs reality using explicit belief statements.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run behavioral analysis (main command)
python -m src.behavioral_analysis --model mistralai/Mistral-7B-v0.3 --tasks 100

# Generate tasks
python -m src.task_generator --families 500 --output tasks.json

# Test model loading
python -m src.model_runner --model mistralai/Mistral-7B-v0.3

# Test inference caching (RUN AFTER ANY CHANGES TO model_runner.py)
python tests/test_cache.py

# Start dashboard (port 8016)
nohup python server_watchdog.py > /tmp/MechInt_watchdog.log 2>&1 &

# Stop server
pkill -f "server_watchdog.py" && pkill -f "web_dashboard.py"
```

## Architecture

### Core Pipeline

1. **Task Generation** (`src/task_generator.py`) - Creates false/true belief tasks
2. **Behavioral Analysis** (`src/behavioral_analysis.py`) - Runs inference and scores responses
3. **Dashboard** (`web_dashboard.py`) - Visualizes results on port 8016

### Data Flow
```
tasks.json → behavioral_analysis → results/behavioral/ → web_dashboard
```

### Task Format
```
False belief: "A ball is in the blue box. Anna believes the ball is in the red box."
True belief:  "A ball is in the blue box. Anna believes the ball is in the blue box."
Question: "Where will Anna look for the ball?"
```

## Key Files

- `src/task_generator.py` - ToMTask dataclass, 4 variants per family (tb1-tb2, fb1-fb2), 50/50 TB/FB
- `src/model_runner.py` - HF model loading for inference
- `src/behavioral_analysis.py` - Response parsing and accuracy metrics
- `web_dashboard.py` - HTTP server with visualizations

## Configuration

- `.env` - Server port (8016), model name, study parameters
- `tasks.json` - Generated task definitions
- `models.json` - Available models configuration
- This project uses port **8016** (configured in `.env`).


## Critical Rules
## Workflow Orchestration
### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update 'tasks/lessons.md"
with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management
1. **Plan First**: Write plan to "tasks/todo.md" with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to "tasks/todo.md"

**Capture Lessons**: Update "tasks/lessons.md' after corrections

## Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary Avoid introducing bugs.
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
- **Commit after refactoring**: Always commit and push after major refactoring or feature additions.

## Post-Run Checklist
**After each behavioral analysis run:**

1. Review errors at `http://127.0.0.1:8016/tasks` (filter by "Wrong")
2. Check if parsing can be improved in `parse_response()` in `src/behavioral_analysis.py`
3. Re-run behavioral analysis after parsing improvements to get accurate metrics