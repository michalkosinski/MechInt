# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mechanistic Interpretability study investigating Theory of Mind (ToM) in LLMs through attention head analysis. Tests whether models track protagonist beliefs vs reality using explicit belief statements.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full study
python -m src.run_study

# Run behavioral verification only
python -m src.run_study --behavioral-only

# Run with specific model
python -m src.run_study --model "Qwen/Qwen2.5-3B-Instruct"

# Start dashboard (port 8016)
nohup python server_watchdog.py > /tmp/MechInt_watchdog.log 2>&1 &

# Stop server
pkill -f "server_watchdog.py" && pkill -f "web_dashboard.py"

# Run tests
python -m pytest tests/ -v
```

## Architecture

### Core Pipeline
1. **Task Generation** (`src/task_generator.py`) - Creates false/true belief tasks with explicit belief statements
2. **Model Execution** (`src/model_runner.py`) - Runs HuggingFace model with attention extraction
3. **Attention Analysis** (`src/attention_analyzer.py`) - Identifies ToM-relevant heads by belief-reality ratio
4. **Intervention** (`src/intervention.py`) - Ablates/boosts heads to test causal role
5. **Dashboard** (`web_dashboard.py`) - Visualizes results on port 8016

### Data Flow
```
tasks.json → model_runner → attention_analyzer → intervention → web_dashboard
                 ↓                  ↓                 ↓
         results/behavioral/  results/attention/  results/interventions/
```

### Task Format
```
False belief: "A ball is in the blue box. Anna believes the ball is in the red box."
True belief:  "A ball is in the blue box. Anna believes the ball is in the blue box."
Question: "Where will Anna look for the ball?"
```

## Key Files

- `src/task_generator.py` - ToMTask dataclass, template-based generation
- `src/model_runner.py` - HF model loading with `output_attentions=True`
- `src/attention_analyzer.py` - Head identification (belief-reality ratio > 1.5)
- `src/intervention.py` - PyTorch hooks for attention modification
- `web_dashboard.py` - HTTP server with visualizations

## Configuration

- `.env` - Server port (8016), model name, study parameters
- `tasks.json` - Generated task definitions

## Critical Rules

**Port**: This project uses port **8016** (configured in `.env`).

**Model loading**: Always use `attn_implementation="eager"` to get attention outputs.

**Attention shape**: For Qwen models: `(batch, num_heads, seq_len, seq_len)`

## ToM Head Identification

A head is "ToM-relevant" if:
1. In false belief condition: `attention_to_belief / attention_to_reality > 1.5`
2. Significant difference between false and true belief conditions
