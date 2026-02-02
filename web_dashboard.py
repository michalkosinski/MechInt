#!/usr/bin/env python3
"""
MechInt Web Dashboard

Web interface for viewing Theory of Mind behavioral analysis results.
Features:
- Behavioral accuracy overview
- Per-task results with filtering and sorting
- Model selection

Usage:
    python web_dashboard.py [--port PORT] [--host HOST]
"""

import argparse
import csv
import io
import json
import os
import re
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse
import html as html_module

# Project paths
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
TASKS_FILE = PROJECT_ROOT / "tasks.json"
MODELS_FILE = PROJECT_ROOT / "models.json"


def load_models() -> Dict[str, Dict]:
    """Load available models from models.json."""
    if MODELS_FILE.exists():
        data = json.loads(MODELS_FILE.read_text())
        return data.get("models", {})
    return {}


def get_model_list() -> List[str]:
    """Get list of available model names."""
    return list(load_models().keys())


def load_env_port() -> int:
    """Load SERVER_PORT from .env file."""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("SERVER_PORT="):
                    return int(line.split("=", 1)[1].strip())
    return 8016


DEFAULT_PORT = load_env_port()


class Dashboard:
    """Load and process MechInt study results."""

    def __init__(self, results_dir: Path = RESULTS_DIR, model: Optional[str] = None):
        self.base_results_dir = Path(results_dir)
        self.model = model
        # If a model is specified, try model-specific subdirectory first
        if model:
            # Convert model name to safe directory name (e.g., "Qwen/Qwen2.5-3B" -> "Qwen--Qwen2.5-3B")
            safe_model_name = model.replace("/", "--")
            model_dir = self.base_results_dir / safe_model_name
            # Fall back to base directory if model-specific directory doesn't exist
            if model_dir.exists():
                self.results_dir = model_dir
            else:
                self.results_dir = self.base_results_dir
        else:
            self.results_dir = self.base_results_dir

    def get_study_summary(self) -> Optional[Dict]:
        """Load study summary."""
        path = self.results_dir / "study_summary.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def get_behavioral_results(self) -> List[Dict]:
        """Load behavioral result files, filtered by model if specified."""
        results = []
        behavioral_dir = self.results_dir / "behavioral"
        if behavioral_dir.exists():
            for f in behavioral_dir.glob("*.json"):
                # If a model is specified, only load matching files
                if self.model:
                    # Model name in filename: "Qwen/Qwen2.5-3B" -> "Qwen_Qwen2.5-3B"
                    expected_filename = self.model.replace("/", "_") + ".json"
                    if f.name != expected_filename:
                        continue
                data = json.loads(f.read_text())
                data["filename"] = f.name
                results.append(data)
        return results

    def get_behavioral_results_by_task(self) -> tuple:
        """Load all behavioral files and group by task_id.

        Returns:
            (results_by_task, summaries_by_question_type)

        results_by_task = {
            "f001_fb1": {
                "reality": {...detail...},
                "protagonist": {...detail...},
                "observer": {...detail...}
            }
        }
        """
        behavioral_dir = self.results_dir / "behavioral"
        results_by_task: Dict[str, Dict] = {}
        summaries: Dict[str, Dict] = {}

        if not behavioral_dir.exists():
            return results_by_task, summaries

        # Get model slug for filename matching
        model_slug = self.model.replace("/", "_") if self.model else None

        for question_type in ["reality", "protagonist", "observer"]:
            # Find matching file for this question type
            if model_slug:
                pattern = f"{model_slug}_{question_type}.json"
            else:
                pattern = f"*_{question_type}.json"

            for f in behavioral_dir.glob(pattern):
                try:
                    data = json.loads(f.read_text())
                    details = data.get("details", [])
                    stored_summary = data.get("summary", {})

                    # Compute summary from details instead of using stored summary
                    total_tasks = len(details)
                    correct = sum(1 for d in details if d.get("is_correct", False))
                    uncertain = sum(1 for d in details if d.get("is_uncertain", False))

                    # Compute FB/TB accuracy
                    fb_tasks = [d for d in details if d.get("task_type") == "false_belief"]
                    tb_tasks = [d for d in details if d.get("task_type") == "true_belief"]
                    fb_correct = sum(1 for d in fb_tasks if d.get("is_correct", False))
                    tb_correct = sum(1 for d in tb_tasks if d.get("is_correct", False))

                    summaries[question_type] = {
                        "model_name": stored_summary.get("model_name", ""),
                        "question_type": question_type,
                        "timestamp": stored_summary.get("timestamp", ""),
                        "total_tasks": total_tasks,
                        "correct": correct,
                        "accuracy": correct / total_tasks if total_tasks > 0 else 0,
                        "uncertain": uncertain,
                        "false_belief_accuracy": fb_correct / len(fb_tasks) if fb_tasks else 0,
                        "true_belief_accuracy": tb_correct / len(tb_tasks) if tb_tasks else 0,
                    }

                    for detail in details:
                        task_id = detail.get("task_id")
                        if task_id:
                            if task_id not in results_by_task:
                                results_by_task[task_id] = {}
                            results_by_task[task_id][question_type] = detail
                except (json.JSONDecodeError, KeyError):
                    continue
                # Only process first matching file per question type
                break

        return results_by_task, summaries

    def get_tasks(self) -> List[Dict]:
        """Load tasks."""
        if TASKS_FILE.exists():
            data = json.loads(TASKS_FILE.read_text())
            return data.get("tasks", [])
        return []

    def get_probing_results(self) -> Optional[Dict]:
        """Load probing results for the selected model (if available)."""
        probing_dir = self.results_dir / "probing"
        if self.model:
            model_dir = probing_dir / self.model.replace("/", "_")
            metrics_path = model_dir / "probe_metrics.json"
            if metrics_path.exists():
                return json.loads(metrics_path.read_text())
            return None

        if probing_dir.exists():
            for metrics_path in probing_dir.glob("*/probe_metrics.json"):
                return json.loads(metrics_path.read_text())
        return None

    def get_probing_predictions(self) -> Optional[Dict]:
        """Load per-task probing predictions for the selected model."""
        probing_dir = self.results_dir / "probing"
        if self.model:
            model_dir = probing_dir / self.model.replace("/", "_")
            predictions_path = model_dir / "probe_predictions.json"
            if predictions_path.exists():
                return json.loads(predictions_path.read_text())
            return None

        if probing_dir.exists():
            for predictions_path in probing_dir.glob("*/probe_predictions.json"):
                return json.loads(predictions_path.read_text())
        return None

    def get_attention_results(self) -> Optional[Dict]:
        """Load attention results for the selected model (if available)."""
        attention_dir = self.results_dir / "attention"
        if self.model:
            model_dir = attention_dir / self.model.replace("/", "_")
            metrics_path = model_dir / "attention_metrics.json"
            if metrics_path.exists():
                return json.loads(metrics_path.read_text())
            return None

        if attention_dir.exists():
            for metrics_path in attention_dir.glob("*/attention_metrics.json"):
                return json.loads(metrics_path.read_text())
        return None


# Global dashboard instance (will be recreated per-request with model)
dashboard = Dashboard()


def get_dashboard(model: Optional[str] = None) -> Dashboard:
    """Get a Dashboard instance for the specified model."""
    return Dashboard(model=model)


def render_page(title: str, content: str, nav_active: str = "", selected_model: Optional[str] = None) -> str:
    """Render a full HTML page with navigation and model selector."""
    # Build model query string for preserving selection across navigation
    model_qs = f"?model={selected_model}" if selected_model else ""

    nav_items = [
        ("Overview", "/"),
        ("Tasks", "/tasks"),
        ("Probing", "/probing"),
        ("Attention", "/attention"),
    ]

    nav_html = "\n".join([
        f'<a href="{url}{model_qs}" class="{"active" if nav_active == name else ""}">{name}</a>'
        for name, url in nav_items
    ])

    # Build model selector dropdown
    models = get_model_list()
    model_selector = ""
    if models:
        # If no model selected, default to first model
        if not selected_model:
            selected_model = models[0]

        model_options = ""
        for model in models:
            # Get short display name (last part after /)
            display_name = model.split("/")[-1] if "/" in model else model
            selected = "selected" if model == selected_model else ""
            model_options += f'<option value="{html_module.escape(model)}" {selected}>{html_module.escape(display_name)}</option>'

        model_selector = f'''
            <div class="model-selector">
                <label for="model-select">Model:</label>
                <select id="model-select" onchange="switchModel(this.value)">
                    {model_options}
                </select>
            </div>
        '''

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - MechInt Dashboard</title>
    <style>
        :root {{
            --bg-primary: #0f1419;
            --bg-secondary: #1a1f26;
            --bg-tertiary: #242b33;
            --text-primary: #e7e9ea;
            --text-secondary: #8b98a5;
            --accent: #1d9bf0;
            --success: #00ba7c;
            --warning: #ffad1f;
            --error: #f4212e;
            --border: #2f3336;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.5;
        }}

        nav {{
            background: var(--bg-secondary);
            padding: 1rem 2rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            gap: 2rem;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }}

        nav .logo {{
            font-weight: 700;
            font-size: 1.25rem;
            color: var(--accent);
        }}

        nav a {{
            color: var(--text-secondary);
            text-decoration: none;
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            transition: all 0.2s;
        }}

        nav a:hover {{
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }}

        nav a.active {{
            background: var(--accent);
            color: white;
        }}

        main {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}

        h1 {{
            font-size: 2rem;
            margin-bottom: 1.5rem;
        }}

        h2 {{
            font-size: 1.5rem;
            margin: 2rem 0 1rem;
            color: var(--text-secondary);
        }}

        .card-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .card {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--border);
        }}

        .card h3 {{
            color: var(--text-secondary);
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}

        .card .value {{
            font-size: 2.5rem;
            font-weight: 700;
        }}

        .card .value.success {{ color: var(--success); }}
        .card .value.warning {{ color: var(--warning); }}
        .card .value.error {{ color: var(--error); }}

        .card .subtext {{
            color: var(--text-secondary);
            font-size: 0.875rem;
            margin-top: 0.25rem;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-secondary);
            border-radius: 12px;
            overflow: hidden;
        }}

        th, td {{
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        th {{
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        tr:hover {{
            background: var(--bg-tertiary);
        }}

        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        .badge.success {{ background: rgba(0, 186, 124, 0.2); color: var(--success); }}
        .badge.error {{ background: rgba(244, 33, 46, 0.2); color: var(--error); }}
        .badge.info {{ background: rgba(29, 155, 240, 0.2); color: var(--accent); }}

        .heatmap-container {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--border);
            overflow-x: auto;
        }}

        .heatmap {{
            display: grid;
            gap: 2px;
            font-size: 0.75rem;
        }}

        .heatmap-cell {{
            width: 24px;
            height: 24px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: transform 0.1s;
        }}

        .heatmap-cell:hover {{
            transform: scale(1.2);
            z-index: 10;
        }}

        .heatmap-label {{
            color: var(--text-secondary);
            font-size: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .legend {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-top: 1rem;
            color: var(--text-secondary);
            font-size: 0.875rem;
        }}

        .legend-gradient {{
            width: 200px;
            height: 16px;
            border-radius: 4px;
            background: linear-gradient(to right, hsl(0, 70%, 25%), hsl(60, 70%, 35%), hsl(120, 70%, 45%));
        }}

        .empty-state {{
            text-align: center;
            padding: 4rem;
            color: var(--text-secondary);
        }}

        .empty-state .debug-info {{
            margin-top: 1.5rem;
            text-align: left;
            display: inline-block;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            padding: 1rem 1.25rem;
            border-radius: 10px;
            font-size: 0.875rem;
        }}

        .empty-state svg {{
            width: 64px;
            height: 64px;
            margin-bottom: 1rem;
            opacity: 0.5;
        }}

        .tabs {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
        }}

        .tab {{
            padding: 0.5rem 1rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s;
        }}

        .tab:hover, .tab.active {{
            background: var(--accent);
            border-color: var(--accent);
            color: white;
        }}

        .prompt-box {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 1rem;
            font-family: monospace;
            font-size: 0.875rem;
            white-space: pre-wrap;
            word-break: break-word;
        }}

        .delta-positive {{ color: var(--success); }}
        .delta-negative {{ color: var(--error); }}

        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.875rem;
            border-top: 1px solid var(--border);
            margin-top: 2rem;
        }}

        .model-selector {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-left: auto;
            padding-left: 1rem;
            border-left: 1px solid var(--border);
        }}

        .model-selector label {{
            color: var(--text-secondary);
            font-size: 0.875rem;
        }}

        .model-selector select {{
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.375rem 0.75rem;
            font-size: 0.875rem;
            cursor: pointer;
            min-width: 180px;
        }}

        .model-selector select:hover {{
            border-color: var(--accent);
        }}

        .model-selector select:focus {{
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(29, 155, 240, 0.2);
        }}
    </style>
    <script>
        function switchModel(modelName) {{
            const url = new URL(window.location);
            if (modelName) {{
                url.searchParams.set('model', modelName);
            }} else {{
                url.searchParams.delete('model');
            }}
            window.location.href = url.toString();
        }}
    </script>
</head>
<body>
    <nav>
        <span class="logo">MechInt</span>
        {nav_html}
        {model_selector}
    </nav>
    <main>
        {content}
    </main>
    <footer>
        MechInt Dashboard &middot; Port {DEFAULT_PORT}
    </footer>
</body>
</html>"""


def render_overview(selected_model: Optional[str] = None) -> str:
    """Render the overview page with metrics from all question types."""
    dash = get_dashboard(selected_model)
    results_by_task, summaries = dash.get_behavioral_results_by_task()

    if not summaries:
        return render_page("Overview", """
            <h1>Overview</h1>
            <div class="empty-state">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p>No behavioral results found.</p>
                <p>Run <code>python -m src.behavioral_analysis</code> to generate results.</p>
            </div>
        """, "Overview", selected_model=selected_model)

    # Get model info from any available summary
    first_summary = next(iter(summaries.values()), {})
    model_name = first_summary.get("model_name", selected_model or "Unknown")
    timestamp = first_summary.get("timestamp", "N/A")
    total_tasks = first_summary.get("total_tasks", len(results_by_task))

    # Calculate overall accuracy across all question types
    total_correct = 0
    total_count = 0
    total_uncertain = 0

    for qt, summary in summaries.items():
        total_correct += summary.get("correct", 0)
        total_count += summary.get("total_tasks", 0)
        total_uncertain += summary.get("uncertain", 0)

    overall_acc = (total_correct / total_count * 100) if total_count > 0 else 0
    acc_class = "success" if overall_acc >= 70 else "warning" if overall_acc >= 50 else "error"

    # Build Row 1: Overall metrics cards
    row1_cards = []

    row1_cards.append(f"""
        <div class="card">
            <h3>Overall Accuracy</h3>
            <div class="value {acc_class}">{overall_acc:.1f}%</div>
            <div class="subtext">Across all question types</div>
        </div>
    """)

    # Get protagonist summary for FB/TB breakdown (most relevant for ToM)
    prot_summary = summaries.get("protagonist", {})
    fb_acc = prot_summary.get("false_belief_accuracy", 0) * 100
    fb_class = "success" if fb_acc >= 70 else "warning" if fb_acc >= 50 else "error"
    row1_cards.append(f"""
        <div class="card">
            <h3>False Belief (Protagonist)</h3>
            <div class="value {fb_class}">{fb_acc:.1f}%</div>
            <div class="subtext">Key ToM metric</div>
        </div>
    """)

    tb_acc = prot_summary.get("true_belief_accuracy", 0) * 100
    tb_class = "success" if tb_acc >= 70 else "warning" if tb_acc >= 50 else "error"
    row1_cards.append(f"""
        <div class="card">
            <h3>True Belief (Protagonist)</h3>
            <div class="value {tb_class}">{tb_acc:.1f}%</div>
            <div class="subtext">Control condition</div>
        </div>
    """)

    row1_cards.append(f"""
        <div class="card">
            <h3>Tasks Analyzed</h3>
            <div class="value">{total_tasks}</div>
            <div class="subtext">{len(summaries)} question types</div>
        </div>
    """)

    # Build Row 2: Question type breakdown
    row2_cards = []
    question_labels = {
        "reality": ("Reality", "Where is the object?"),
        "protagonist": ("Protagonist", "Where will they look?"),
        "observer": ("Observer", "Where will observer look?")
    }

    for qt in ["reality", "protagonist", "observer"]:
        summary = summaries.get(qt, {})
        if summary:
            acc = summary.get("accuracy", 0) * 100
            acc_class = "success" if acc >= 70 else "warning" if acc >= 50 else "error"
            label, desc = question_labels.get(qt, (qt.title(), ""))
            row2_cards.append(f"""
                <div class="card">
                    <h3>{label} Question</h3>
                    <div class="value {acc_class}">{acc:.1f}%</div>
                    <div class="subtext">{desc}</div>
                </div>
            """)

    if total_uncertain > 0:
        row2_cards.append(f"""
            <div class="card">
                <h3>Uncertain Responses</h3>
                <div class="value warning">{total_uncertain}</div>
                <div class="subtext">Could not parse answer</div>
            </div>
        """)

    # Build comparison chart (FB vs TB for each question type)
    chart_rows = []
    for qt in ["reality", "protagonist", "observer"]:
        summary = summaries.get(qt, {})
        if summary:
            fb = summary.get("false_belief_accuracy", 0) * 100
            tb = summary.get("true_belief_accuracy", 0) * 100
            label = question_labels.get(qt, (qt.title(), ""))[0]
            chart_rows.append(f"""
                <div class="chart-row">
                    <span class="chart-label">{label}</span>
                    <div class="chart-bars">
                        <div class="bar-group">
                            <span class="bar-type">FB</span>
                            <div class="bar fb" style="width: {fb}%"><span>{fb:.0f}%</span></div>
                        </div>
                        <div class="bar-group">
                            <span class="bar-type">TB</span>
                            <div class="bar tb" style="width: {tb}%"><span>{tb:.0f}%</span></div>
                        </div>
                    </div>
                </div>
            """)

    comparison_chart = f"""
        <div class="comparison-section">
            <h2>Performance by Question Type</h2>
            <div class="comparison-chart">
                {''.join(chart_rows)}
            </div>
            <div class="chart-legend">
                <span class="legend-item"><span class="legend-dot fb"></span> False Belief</span>
                <span class="legend-item"><span class="legend-dot tb"></span> True Belief</span>
            </div>
        </div>
    """ if chart_rows else ""

    # Model info table
    model_info = f"""
        <h2>Study Configuration</h2>
        <table>
            <tr><th>Model</th><td>{html_module.escape(str(model_name))}</td></tr>
            <tr><th>Timestamp</th><td>{html_module.escape(str(timestamp))}</td></tr>
            <tr><th>Total Tasks</th><td>{total_tasks}</td></tr>
            <tr><th>Question Types</th><td>{', '.join(summaries.keys())}</td></tr>
        </table>
    """

    # Custom CSS for overview
    overview_css = """
        <style>
            .comparison-section {
                background: var(--bg-secondary);
                border-radius: 12px;
                padding: 1.5rem;
                border: 1px solid var(--border);
                margin-bottom: 2rem;
            }
            .comparison-section h2 {
                margin: 0 0 1rem 0;
                font-size: 1rem;
                color: var(--text-secondary);
            }
            .comparison-chart {
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }
            .chart-row {
                display: grid;
                grid-template-columns: 100px 1fr;
                gap: 1rem;
                align-items: center;
            }
            .chart-label {
                color: var(--text-secondary);
                font-weight: 500;
                font-size: 0.875rem;
            }
            .chart-bars {
                display: flex;
                flex-direction: column;
                gap: 0.25rem;
            }
            .bar-group {
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            .bar-type {
                width: 24px;
                font-size: 0.7rem;
                color: var(--text-secondary);
            }
            .bar {
                height: 24px;
                border-radius: 4px;
                display: flex;
                align-items: center;
                justify-content: flex-end;
                padding-right: 0.5rem;
                color: white;
                font-size: 0.75rem;
                font-weight: 600;
                min-width: 40px;
                transition: width 0.3s ease;
            }
            .bar.fb { background: var(--accent); }
            .bar.tb { background: var(--success); }
            .chart-legend {
                display: flex;
                gap: 1.5rem;
                margin-top: 1rem;
                padding-top: 1rem;
                border-top: 1px solid var(--border);
            }
            .legend-item {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-size: 0.875rem;
                color: var(--text-secondary);
            }
            .legend-dot {
                width: 12px;
                height: 12px;
                border-radius: 3px;
            }
            .legend-dot.fb { background: var(--accent); }
            .legend-dot.tb { background: var(--success); }
        </style>
    """

    content = f"""
        {overview_css}
        <h1>Study Overview</h1>
        <div class="card-grid">
            {''.join(row1_cards)}
        </div>
        <div class="card-grid">
            {''.join(row2_cards)}
        </div>
        {comparison_chart}
        {model_info}
    """

    return render_page("Overview", content, "Overview", selected_model=selected_model)

def render_tasks(selected_model: Optional[str] = None) -> str:
    """Render the tasks page with card layout showing all 3 question types per task."""
    dash = get_dashboard(selected_model)
    tasks = dash.get_tasks()
    results_by_task, _ = dash.get_behavioral_results_by_task()

    # Filter to only show tasks with results for the selected model
    if results_by_task:
        tasks = [t for t in tasks if t.get("task_id") in results_by_task]

    if not tasks:
        return render_page("Tasks", """
            <h1>Tasks</h1>
            <div class="empty-state">
                <p>No tasks found.</p>
                <p>Run <code>python -m src.task_generator</code> to generate tasks.</p>
            </div>
        """, "Tasks", selected_model=selected_model)

    # Sort tasks by ID (natural sort for f001_fb1 format)
    def sort_key(t):
        task_id = t.get("task_id", "")
        # Extract family number and variant (e.g., "f001_fb1" -> ("f", 1, "fb1"))
        match = re.match(r"f(\d+)_([a-z]+)(\d+)", task_id)
        if match:
            return (int(match.group(1)), match.group(2), int(match.group(3)))
        return (0, task_id, 0)

    tasks = sorted(tasks, key=sort_key)

    # Question type info
    question_types = ["reality", "protagonist", "observer"]
    question_labels = {
        "reality": "Reality",
        "protagonist": "Protagonist",
        "observer": "Observer"
    }

    # Build cards
    cards = []
    total_with_errors = 0

    for idx, task in enumerate(tasks):
        task_id = task.get("task_id", "")
        task_type = task.get("task_type", "")
        family_id = task.get("family_id", "")
        narrative = task.get("narrative", "")

        # Get task details for generating questions
        protagonist_name = task.get("protagonist", "")
        observer_name = task.get("observer", "")
        obj = task.get("obj", "")
        protagonist_belief = task.get("protagonist_belief", "")
        final_location = task.get("final_location", "")

        # Generate question text from task data (last sentence of prompt)
        questions = {
            "reality": f"The {obj} is in the",
            "protagonist": f"{protagonist_name} will look for the {obj} in the",
            "observer": f"{observer_name} will look for the {obj} in the"
        }

        expected_answers = {
            "reality": final_location,
            "protagonist": protagonist_belief,
            "observer": final_location
        }

        # Get results for this task
        task_results = results_by_task.get(task_id, {})

        # Determine card status
        has_any_result = bool(task_results)
        error_count = 0
        status_dots = []
        result_rows = []

        for qt in question_types:
            result = task_results.get(qt)
            expected = expected_answers.get(qt, "")

            if result:
                is_correct = result.get("is_correct", False)
                is_uncertain = result.get("is_uncertain", False)
                parsed = result.get("parsed_response", "")
                raw_response = result.get("response", "")
                full_prompt = result.get("prompt", "")

                # Extract question (last sentence after "returns.")
                if "returns." in full_prompt:
                    question_text = full_prompt.split("returns.")[-1].strip()
                else:
                    question_text = full_prompt.split(".")[-1].strip() if "." in full_prompt else full_prompt

                if is_uncertain:
                    status_class = "uncertain"
                    status_symbol = "?"
                    error_count += 1
                elif is_correct:
                    status_class = "correct"
                    status_symbol = "✓"
                else:
                    status_class = "incorrect"
                    status_symbol = "✗"
                    error_count += 1
            else:
                # No result for this question type - use generated question
                status_class = "missing"
                status_symbol = "—"
                parsed = "—"
                raw_response = ""
                full_prompt = ""
                question_text = questions.get(qt, "")

            status_dots.append(f'<span class="status-dot {status_class}" title="{question_labels[qt]}: {status_class}"></span>')

            # Truncate raw response for display
            raw_display = raw_response[:40] + "..." if len(raw_response) > 40 else raw_response

            row_id = f"{task_id}_{qt}"
            result_rows.append(f"""
                <div class="result-row" data-row-id="{row_id}">
                    <div class="result-main">
                        <span class="result-label">{question_labels[qt]}</span>
                        <span class="result-prompt">{html_module.escape(question_text)}</span>
                        <span class="result-expected">{html_module.escape(expected)}</span>
                        <span class="result-parsed {status_class}">{html_module.escape(parsed)}</span>
                        <span class="result-raw" title="{html_module.escape(raw_response)}">{html_module.escape(raw_display) if raw_response else '—'}</span>
                        <span class="result-status {status_class}">{status_symbol}</span>
                        <button class="btn-row-toggle" onclick="toggleRowDetails('{row_id}')" {'disabled' if not result else ''}>▶</button>
                    </div>
                    <div class="result-details" id="row-details-{row_id}" style="display: none;">
                        <div class="detail-item">
                            <span class="detail-label">Full Prompt:</span>
                            <code class="detail-value">{html_module.escape(full_prompt) if full_prompt else '—'}</code>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Raw Response:</span>
                            <code class="detail-value">{html_module.escape(raw_response) if raw_response else '—'}</code>
                        </div>
                    </div>
                </div>
            """)

        if error_count > 0:
            total_with_errors += 1

        # Card status class
        if not has_any_result:
            card_status = "no-results"
        elif error_count == 0:
            card_status = "all-correct"
        else:
            card_status = "has-error"

        # Type badge
        type_class = "tb" if task_type == "true_belief" else "fb"
        type_label = "True Belief" if task_type == "true_belief" else "False Belief"

        # Build expanded details
        details_sections = []
        for qt in question_types:
            result = task_results.get(qt)
            if result:
                prompt = result.get("prompt", "")
                raw_response = result.get("response", "")
                parsed = result.get("parsed_response", "")
                expected = expected_answers.get(qt, "")
                is_correct = result.get("is_correct", False)

                details_sections.append(f"""
                    <div class="detail-question">
                        <h4>{question_labels[qt]} Question</h4>
                        <div class="detail-prompt">{html_module.escape(prompt)}</div>
                        <div class="detail-response">
                            <span class="response-label">Response:</span>
                            <code>{html_module.escape(raw_response)}</code>
                        </div>
                        <div class="detail-comparison">
                            <span>Parsed: <strong>{html_module.escape(parsed)}</strong></span>
                            <span>Expected: <strong>{html_module.escape(expected)}</strong></span>
                            <span class="{'correct' if is_correct else 'incorrect'}">{'✓ Match' if is_correct else '✗ Mismatch'}</span>
                        </div>
                    </div>
                """)

        # Data attributes for filtering
        protagonist_result = task_results.get("protagonist", {})
        reality_result = task_results.get("reality", {})

        cards.append(f"""
            <div class="task-card {card_status}"
                 data-task-id="{task_id}"
                 data-family-id="{family_id}"
                 data-task-type="{task_type}"
                 data-status="{card_status}"
                 data-error-count="{error_count}"
                 data-protagonist-correct="{'true' if protagonist_result.get('is_correct') else 'false'}"
                 data-reality-correct="{'true' if reality_result.get('is_correct') else 'false'}">
                <div class="card-header">
                    <div class="card-meta">
                        <span class="task-id">{task_id}</span>
                        <span class="type-badge {type_class}">{type_label}</span>
                    </div>
                    <div class="card-status">
                        {''.join(status_dots)}
                        <button class="btn-toggle" onclick="toggleCard('{task_id}')">▶</button>
                    </div>
                </div>
                <div class="card-narrative">{html_module.escape(narrative)}</div>
                <div class="results-grid">
                    <div class="results-header">
                        <span></span>
                        <span>Question</span>
                        <span>Expected</span>
                        <span>Parsed</span>
                        <span>Raw Response</span>
                        <span></span>
                        <span></span>
                    </div>
                    {''.join(result_rows)}
                </div>
                <div class="card-details" id="details-{task_id}" style="display: none;">
                    <div class="detail-narrative">
                        <h4>Full Narrative</h4>
                        <p>{html_module.escape(narrative)}</p>
                    </div>
                    {''.join(details_sections)}
                </div>
            </div>
        """)

    total = len(tasks)

    # CSS for task cards
    tasks_css = """
        <style>
            .tasks-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1.5rem;
                flex-wrap: wrap;
                gap: 1rem;
            }
            .tasks-header h1 {
                margin: 0;
            }
            .tasks-stats {
                display: flex;
                gap: 1rem;
                font-size: 0.875rem;
                color: var(--text-secondary);
            }
            .tasks-stats span {
                padding: 0.25rem 0.75rem;
                background: var(--bg-secondary);
                border-radius: 9999px;
            }
            .filters {
                display: flex;
                gap: 1rem;
                flex-wrap: wrap;
                margin-bottom: 1.5rem;
                padding: 1rem;
                background: var(--bg-secondary);
                border-radius: 12px;
                border: 1px solid var(--border);
            }
            .filter-group {
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            .filter-group label {
                color: var(--text-secondary);
                font-size: 0.875rem;
            }
            .filter-group select, .filter-group input {
                padding: 0.5rem 0.75rem;
                border: 1px solid var(--border);
                border-radius: 6px;
                background: var(--bg-tertiary);
                color: var(--text-primary);
                font-size: 0.875rem;
            }
            .filter-group select:focus, .filter-group input:focus {
                outline: none;
                border-color: var(--accent);
            }
            .filter-group input {
                width: 200px;
            }
            .btn-reset {
                padding: 0.5rem 1rem;
                background: var(--bg-tertiary);
                border: 1px solid var(--border);
                border-radius: 6px;
                color: var(--text-secondary);
                cursor: pointer;
                font-size: 0.875rem;
            }
            .btn-reset:hover {
                border-color: var(--accent);
                color: var(--text-primary);
            }
            .task-cards-container {
                display: flex;
                flex-direction: column;
                gap: 1rem;
                max-width: 1100px;
            }
            .task-card {
                background: var(--bg-secondary);
                border-radius: 12px;
                border: 1px solid var(--border);
                overflow: hidden;
            }
            .task-card.all-correct {
                border-left: 4px solid var(--success);
            }
            .task-card.has-error {
                border-left: 4px solid var(--error);
            }
            .task-card.no-results {
                border-left: 4px solid var(--text-secondary);
                opacity: 0.7;
            }
            .card-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.75rem 1rem;
                background: var(--bg-tertiary);
                border-bottom: 1px solid var(--border);
            }
            .card-meta {
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }
            .task-id {
                font-weight: 600;
                font-family: monospace;
                color: var(--accent);
            }
            .type-badge {
                padding: 0.2rem 0.6rem;
                border-radius: 4px;
                font-size: 0.75rem;
                font-weight: 600;
            }
            .type-badge.fb {
                background: rgba(244, 33, 46, 0.2);
                color: var(--error);
            }
            .type-badge.tb {
                background: rgba(0, 186, 124, 0.2);
                color: var(--success);
            }
            .card-status {
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            .status-dot {
                width: 10px;
                height: 10px;
                border-radius: 50%;
            }
            .status-dot.correct { background: var(--success); }
            .status-dot.incorrect { background: var(--error); }
            .status-dot.uncertain { background: var(--warning); }
            .status-dot.missing {
                background: transparent;
                border: 2px dashed var(--text-secondary);
            }
            .btn-toggle {
                background: none;
                border: none;
                color: var(--text-secondary);
                cursor: pointer;
                font-size: 0.875rem;
                padding: 0.25rem 0.5rem;
                border-radius: 4px;
            }
            .btn-toggle:hover {
                background: var(--bg-secondary);
                color: var(--text-primary);
            }
            .card-narrative {
                padding: 0.75rem 1rem;
                font-size: 0.875rem;
                color: var(--text-secondary);
                border-bottom: 1px solid var(--border);
                line-height: 1.5;
            }
            .results-grid {
                padding: 0.5rem 1rem;
            }
            .results-header {
                display: grid;
                grid-template-columns: 80px 1fr 70px 70px 120px 30px 30px;
                gap: 0.5rem;
                padding: 0.5rem 0;
                font-size: 0.7rem;
                color: var(--text-secondary);
                text-transform: uppercase;
                border-bottom: 1px solid var(--border);
            }
            .result-row {
                border-bottom: 1px solid var(--border);
            }
            .result-row:last-child {
                border-bottom: none;
            }
            .result-main {
                display: grid;
                grid-template-columns: 80px 1fr 70px 70px 120px 30px 30px;
                gap: 0.5rem;
                align-items: center;
                padding: 0.5rem 0;
                font-size: 0.8rem;
            }
            .result-label {
                color: var(--text-secondary);
                font-weight: 500;
                font-size: 0.75rem;
            }
            .result-prompt {
                font-family: monospace;
                font-size: 0.75rem;
                color: var(--text-primary);
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .result-expected {
                font-family: monospace;
                color: var(--text-secondary);
            }
            .result-parsed {
                font-family: monospace;
                font-weight: 600;
            }
            .result-parsed.correct { color: var(--success); }
            .result-parsed.incorrect { color: var(--error); }
            .result-parsed.uncertain { color: var(--warning); }
            .result-parsed.missing { color: var(--text-secondary); opacity: 0.5; }
            .result-raw {
                font-family: monospace;
                font-size: 0.75rem;
                color: var(--text-secondary);
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .result-status {
                font-weight: bold;
                text-align: center;
            }
            .result-status.correct { color: var(--success); }
            .result-status.incorrect { color: var(--error); }
            .result-status.uncertain { color: var(--warning); }
            .result-status.missing { color: var(--text-secondary); opacity: 0.5; }
            .btn-row-toggle {
                background: none;
                border: none;
                color: var(--text-secondary);
                cursor: pointer;
                font-size: 0.7rem;
                padding: 0.2rem;
                border-radius: 3px;
            }
            .btn-row-toggle:hover:not(:disabled) {
                background: var(--bg-tertiary);
                color: var(--text-primary);
            }
            .btn-row-toggle:disabled {
                opacity: 0.3;
                cursor: default;
            }
            .result-details {
                background: var(--bg-tertiary);
                padding: 0.75rem;
                margin: 0.25rem 0 0.5rem 0;
                border-radius: 6px;
                font-size: 0.8rem;
            }
            .detail-item {
                margin-bottom: 0.5rem;
            }
            .detail-item:last-child {
                margin-bottom: 0;
            }
            .detail-label {
                color: var(--text-secondary);
                font-size: 0.7rem;
                text-transform: uppercase;
                display: block;
                margin-bottom: 0.25rem;
            }
            .detail-value {
                display: block;
                background: var(--bg-primary);
                padding: 0.5rem;
                border-radius: 4px;
                font-family: monospace;
                font-size: 0.75rem;
                white-space: pre-wrap;
                word-break: break-word;
            }
            .card-details {
                background: var(--bg-primary);
                padding: 1rem;
                border-top: 1px solid var(--border);
            }
            .detail-narrative {
                margin-bottom: 1rem;
            }
            .detail-narrative h4 {
                font-size: 0.75rem;
                text-transform: uppercase;
                color: var(--text-secondary);
                margin-bottom: 0.5rem;
            }
            .detail-narrative p {
                font-size: 0.875rem;
                line-height: 1.6;
            }
            .detail-question {
                background: var(--bg-secondary);
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 0.75rem;
            }
            .detail-question h4 {
                font-size: 0.875rem;
                color: var(--accent);
                margin-bottom: 0.75rem;
            }
            .detail-prompt {
                background: var(--bg-tertiary);
                padding: 0.75rem;
                border-radius: 6px;
                font-family: monospace;
                font-size: 0.8rem;
                margin-bottom: 0.75rem;
                white-space: pre-wrap;
                word-break: break-word;
            }
            .detail-response {
                margin-bottom: 0.5rem;
                font-size: 0.875rem;
            }
            .detail-response .response-label {
                color: var(--text-secondary);
            }
            .detail-response code {
                background: var(--bg-tertiary);
                padding: 0.2rem 0.5rem;
                border-radius: 4px;
            }
            .detail-comparison {
                display: flex;
                gap: 1.5rem;
                font-size: 0.875rem;
                color: var(--text-secondary);
            }
            .detail-comparison .correct { color: var(--success); }
            .detail-comparison .incorrect { color: var(--error); }
        </style>
    """

    # JavaScript for filtering and card expansion
    tasks_js = """
        <script>
            function toggleCard(taskId) {
                const details = document.getElementById('details-' + taskId);
                const card = details.closest('.task-card');
                const btn = card.querySelector('.btn-toggle');

                if (details.style.display === 'none') {
                    details.style.display = 'block';
                    btn.textContent = '▼';
                } else {
                    details.style.display = 'none';
                    btn.textContent = '▶';
                }
            }

            function toggleRowDetails(rowId) {
                const details = document.getElementById('row-details-' + rowId);
                const row = details.closest('.result-row');
                const btn = row.querySelector('.btn-row-toggle');

                if (details.style.display === 'none') {
                    details.style.display = 'block';
                    btn.textContent = '▼';
                } else {
                    details.style.display = 'none';
                    btn.textContent = '▶';
                }
            }

            function filterTasks() {
                const typeFilter = document.getElementById('filter-type').value;
                const statusFilter = document.getElementById('filter-status').value;
                const searchFilter = document.getElementById('filter-search').value.toLowerCase();

                const cards = document.querySelectorAll('.task-card');
                let visibleCount = 0;
                let errorCount = 0;

                cards.forEach(card => {
                    const taskType = card.dataset.taskType;
                    const status = card.dataset.status;
                    const taskId = card.dataset.taskId;
                    const familyId = card.dataset.familyId;
                    const protagonistCorrect = card.dataset.protagonistCorrect;
                    const realityCorrect = card.dataset.realityCorrect;
                    const narrative = card.querySelector('.card-narrative').textContent.toLowerCase();

                    let show = true;

                    // Type filter
                    if (typeFilter && taskType !== typeFilter) {
                        show = false;
                    }

                    // Status filter
                    if (statusFilter) {
                        if (statusFilter === 'all-correct' && status !== 'all-correct') show = false;
                        if (statusFilter === 'has-error' && status === 'all-correct') show = false;
                        if (statusFilter === 'protagonist-wrong' && protagonistCorrect !== 'false') show = false;
                        if (statusFilter === 'reality-wrong' && realityCorrect !== 'false') show = false;
                    }

                    // Search filter (task ID, family ID, or narrative)
                    if (searchFilter) {
                        const searchable = taskId.toLowerCase() + ' ' + familyId.toLowerCase() + ' ' + narrative;
                        if (!searchable.includes(searchFilter)) {
                            show = false;
                        }
                    }

                    card.style.display = show ? 'block' : 'none';

                    if (show) {
                        visibleCount++;
                        if (status !== 'all-correct') errorCount++;
                    }

                    // Collapse expanded cards when filtering
                    const details = card.querySelector('.card-details');
                    const btn = card.querySelector('.btn-toggle');
                    if (details) {
                        details.style.display = 'none';
                        btn.textContent = '▶';
                    }
                });

                // Update stats
                document.getElementById('visible-count').textContent = visibleCount;
                document.getElementById('error-count').textContent = errorCount;
            }

            function resetFilters() {
                document.getElementById('filter-type').value = '';
                document.getElementById('filter-status').value = '';
                document.getElementById('filter-search').value = '';
                filterTasks();
            }
        </script>
    """

    content = f"""
        {tasks_css}
        <div class="tasks-header">
            <h1>Tasks</h1>
            <div class="tasks-stats">
                <span>Showing <strong id="visible-count">{total}</strong> of {total}</span>
                <span><strong id="error-count">{total_with_errors}</strong> with errors</span>
            </div>
        </div>

        <div class="filters">
            <div class="filter-group">
                <label>Type:</label>
                <select id="filter-type" onchange="filterTasks()">
                    <option value="">All Types</option>
                    <option value="true_belief">True Belief</option>
                    <option value="false_belief">False Belief</option>
                </select>
            </div>
            <div class="filter-group">
                <label>Status:</label>
                <select id="filter-status" onchange="filterTasks()">
                    <option value="">All Results</option>
                    <option value="all-correct">All Correct</option>
                    <option value="has-error">Has Error</option>
                    <option value="protagonist-wrong">Protagonist Wrong</option>
                    <option value="reality-wrong">Reality Wrong</option>
                </select>
            </div>
            <div class="filter-group">
                <label>Search:</label>
                <input type="text" id="filter-search" placeholder="Task ID, family, or narrative..." onkeyup="filterTasks()">
            </div>
            <button class="btn-reset" onclick="resetFilters()">Reset</button>
        </div>

        <div class="task-cards-container">
            {''.join(cards)}
        </div>

        {tasks_js}
    """

    return render_page("Tasks", content, "Tasks", selected_model=selected_model)


def _accuracy_color(acc: float) -> str:
    """Generate color for accuracy (0.5=chance → 1.0=perfect)."""
    # Normalize: 0.5 (chance) → 0, 1.0 (perfect) → 1
    normalized = max(0.0, min(1.0, (acc - 0.5) * 2))
    # Red (bad) → Yellow (ok) → Green (good)
    hue = 120 * normalized  # 0=red, 60=yellow, 120=green
    light = 25 + (20 * normalized)
    return f"hsl({hue:.0f}, 70%, {light:.0f}%)"


def _delta_color(delta: float) -> str:
    """Generate color class for delta value."""
    if delta > 5:
        return "success"
    elif delta < -5:
        return "error"
    return "warning"


def _render_layer_accuracy_svg(metrics: list, width: int = 700, height: int = 220) -> str:
    """Generate SVG line chart of accuracy by layer."""
    if not metrics:
        return ""

    layers = [m["layer"] for m in metrics]
    accs = [m["accuracy"] for m in metrics]

    # SVG coordinates
    margin_left, margin_right, margin_top, margin_bottom = 50, 20, 20, 35
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    # Scale: x = layers, y = 0.5 to 1.0
    x_scale = plot_width / max(1, len(layers) - 1)
    y_min, y_max = 0.5, 1.0
    y_scale = plot_height / (y_max - y_min)

    # Build line path
    points = []
    for i, (layer, acc) in enumerate(zip(layers, accs)):
        x = margin_left + i * x_scale
        y = margin_top + plot_height - (acc - y_min) * y_scale
        points.append(f"{x:.1f},{y:.1f}")
    path_d = "M" + " L".join(points)

    # Y-axis ticks
    y_ticks = ""
    for val in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        y_pos = margin_top + plot_height - (val - y_min) * y_scale
        y_ticks += f'<line x1="{margin_left - 5}" y1="{y_pos:.1f}" x2="{margin_left}" y2="{y_pos:.1f}" stroke="#666"/>'
        y_ticks += f'<text x="{margin_left - 8}" y="{y_pos + 4:.1f}" text-anchor="end" fill="#888" font-size="11">{val*100:.0f}%</text>'

    # X-axis ticks (every 5 layers)
    x_ticks = ""
    for i, layer in enumerate(layers):
        if layer % 5 == 0 or layer == layers[-1]:
            x_pos = margin_left + i * x_scale
            x_ticks += f'<line x1="{x_pos:.1f}" y1="{margin_top + plot_height}" x2="{x_pos:.1f}" y2="{margin_top + plot_height + 5}" stroke="#666"/>'
            x_ticks += f'<text x="{x_pos:.1f}" y="{margin_top + plot_height + 18}" text-anchor="middle" fill="#888" font-size="11">{layer}</text>'

    # 50% baseline
    baseline_y = margin_top + plot_height - (0.5 - y_min) * y_scale

    return f'''
    <svg viewBox="0 0 {width} {height}" class="layer-accuracy-chart" style="width: 100%; max-width: {width}px; height: auto;">
        <!-- Grid -->
        <line x1="{margin_left}" y1="{baseline_y:.1f}" x2="{margin_left + plot_width}" y2="{baseline_y:.1f}" stroke="#444" stroke-dasharray="4"/>
        <!-- Axes -->
        <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#666"/>
        <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#666"/>
        {y_ticks}
        {x_ticks}
        <!-- Data line -->
        <path d="{path_d}" fill="none" stroke="#1d9bf0" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
        <!-- Axis labels -->
        <text x="{margin_left + plot_width/2}" y="{height - 5}" text-anchor="middle" fill="#888" font-size="12">Layer</text>
        <text x="15" y="{margin_top + plot_height/2}" text-anchor="middle" fill="#888" font-size="12" transform="rotate(-90, 15, {margin_top + plot_height/2})">Accuracy</text>
    </svg>
    '''


# Probe label explanations
PROBE_EXPLANATIONS = {
    "false_belief": {
        "title": "False Belief Detection",
        "question": "Can we decode WHETHER the protagonist has a false belief?",
        "description": "Tests if hidden states encode that the protagonist's belief differs from reality.",
        "labels": "0 = true belief (belief = reality), 1 = false belief (belief ≠ reality)",
    },
    "world_location": {
        "title": "World Location (Reality)",
        "question": "Can we decode WHERE the object actually is?",
        "description": "Tests if hidden states encode the true location of the object in the world.",
        "labels": "0 = object in C2, 1 = object in C1 (canonical first container)",
    },
    "belief_location": {
        "title": "Belief Location",
        "question": "Can we decode WHERE the protagonist believes the object is?",
        "description": "Tests if hidden states encode the protagonist's belief about object location.",
        "labels": "0 = believes in C2, 1 = believes in C1 (canonical first container)",
    },
}


def render_probing(selected_model: Optional[str] = None) -> str:
    """Render probing analysis page with detailed explanations."""
    probing_dashboard = get_dashboard(selected_model)
    probing = probing_dashboard.get_probing_results()

    if not probing:
        probing_dir = RESULTS_DIR / "probing"
        available_probe_models = []
        if probing_dir.exists():
            for metrics_path in probing_dir.glob("*/probe_metrics.json"):
                available_probe_models.append(metrics_path.parent.name)
        available_probe_models.sort()
        expected_metrics_path = "N/A"
        if selected_model:
            expected_metrics_path = str(Path("results") / "probing" / selected_model.replace("/", "_") / "probe_metrics.json")
        available_models_display = ", ".join(available_probe_models) if available_probe_models else "None found"
        expected_metrics_path_html = html_module.escape(expected_metrics_path)
        available_models_html = html_module.escape(available_models_display)

        return render_page("Probing", f"""
            <h1>Probing Analysis</h1>
            <div class="empty-state">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                <p>No probing results found.</p>
                <p>Run <code>python -m src.probing_analysis</code> to generate probing results.</p>
                <div class="debug-info">
                    <p><strong>Expected metrics path for selected model:</strong> <code>{expected_metrics_path_html}</code></p>
                    <p><strong>Available probe models:</strong> {available_models_html}</p>
                </div>
            </div>
        """, "Probing", selected_model=selected_model)

    model_name = probing.get("model_name", selected_model or "Unknown")
    timestamp = probing.get("timestamp", "N/A")
    config = probing.get("config", {})
    results = probing.get("results", {})
    timing = probing.get("timing", {})
    feature_position = probing.get("feature_position", "prompt_end")

    # Build configuration display
    config_html = f"""
        <div class="config-panel">
            <h3>Analysis Configuration</h3>
            <div class="config-grid">
                <div class="config-item">
                    <span class="config-label">Families</span>
                    <span class="config-value">{config.get('num_families', 'N/A')}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Tasks</span>
                    <span class="config-value">{config.get('num_families', 0) * 8}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">CV Folds</span>
                    <span class="config-value">{config.get('n_folds', 'N/A')}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Seed</span>
                    <span class="config-value">{config.get('seed', 'N/A')}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Feature</span>
                    <span class="config-value">{feature_position}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Normalization</span>
                    <span class="config-value">{'Yes' if config.get('normalize', True) else 'No'}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Filter Correct</span>
                    <span class="config-value">{'Yes' if config.get('filter_correct', False) else 'No'}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Cache</span>
                    <span class="config-value">{timing.get('cache_status', 'N/A')}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Hidden States (ms)</span>
                    <span class="config-value">{timing.get('hidden_state_time_ms', 0.0):.0f}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Cache Load (ms)</span>
                    <span class="config-value">{timing.get('cache_load_time_ms', 0.0):.0f}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Probe Train (ms)</span>
                    <span class="config-value">{timing.get('probe_train_time_ms', 0.0):.0f}</span>
                </div>
            </div>
            <div class="download-section" style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border);">
                <a href="/probing/download/csv{'?model=' + selected_model if selected_model else ''}" class="download-btn" download>
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16" style="margin-right: 0.5rem;">
                        <path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/>
                        <path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"/>
                    </svg>
                    Download Predictions (CSV)
                </a>
            </div>
        </div>
    """

    # Build sections for each probe type
    sections = []

    for label_type, label_data in results.items():
        explanation = PROBE_EXPLANATIONS.get(label_type, {})
        metrics = label_data.get("metrics", [])
        if not metrics:
            continue

        # Calculate stats
        best_layer = label_data.get("best_layer", 0)
        best_acc = label_data.get("best_accuracy", 0) * 100

        best_random = 50.0
        best_auc = 0.5
        for m in metrics:
            if m.get("layer") == best_layer:
                best_auc = m.get("auc", 0.5)
                break

        delta = best_acc - best_random
        delta_class = _delta_color(delta)

        # Build accuracy chart (CSS bars)
        chart_bars = []
        max_acc = max(m.get("accuracy", 0) for m in metrics) if metrics else 1.0
        max_acc = max(max_acc, 0.6)  # Ensure reasonable scale

        for m in metrics:
            acc = m.get("accuracy", 0)
            rand = 0.5
            layer = m.get("layer", 0)
            height_acc = (acc / max_acc) * 100
            height_rand = (rand / max_acc) * 100

            chart_bars.append(f"""
                <div class="chart-bar-group" title="Layer {layer}: {acc*100:.1f}% (random: {rand*100:.1f}%)">
                    <div class="chart-bar chart-bar-acc" style="height: {height_acc}%;"></div>
                    <div class="chart-bar chart-bar-rand" style="height: {height_rand}%;"></div>
                    <span class="chart-label">{layer}</span>
                </div>
            """)

        # Build metrics table rows
        table_rows = []
        for m in metrics:
            acc = m.get("accuracy", 0) * 100
            rand = 50.0
            row_delta = acc - rand
            row_class = _delta_color(row_delta)

            table_rows.append(f"""
                <tr class="{'highlight-row' if m.get('layer') == best_layer else ''}">
                    <td>{m.get('layer')}</td>
                    <td><strong>{acc:.1f}%</strong></td>
                    <td>{m.get('auc', 0):.3f}</td>
                    <td>{m.get('f1', 0):.3f}</td>
                    <td>{m.get('brier', 0):.3f}</td>
                    <td>{rand:.1f}%</td>
                    <td class="{row_class}">{row_delta:+.1f}%</td>
                </tr>
            """)

        # Heatmap cells
        heatmap_cells = []
        for m in metrics:
            acc = m.get("accuracy", 0)
            heatmap_cells.append(
                f"<div class='heatmap-cell' title='Layer {m.get('layer')}: {acc*100:.1f}%' "
                f"style='background: {_accuracy_color(acc)}'>{m.get('layer')}</div>"
            )

        sections.append(f"""
            <div class="probe-section">
                <div class="probe-header">
                    <h2>{explanation.get('title', label_type)}</h2>
                    <p class="probe-question">{explanation.get('question', '')}</p>
                </div>

                <div class="probe-explanation">
                    <p>{explanation.get('description', '')}</p>
                    <p class="probe-labels"><strong>Labels:</strong> {explanation.get('labels', '')}</p>
                </div>

                <div class="position-description">
                    <strong>Feature:</strong> {feature_position}
                </div>

                <div class="card-grid" style="grid-template-columns: repeat(4, 1fr);">
                    <div class="card">
                        <h3>Best Layer</h3>
                        <div class="value">{best_layer}</div>
                        <div class="subtext">of {len(metrics) - 1} layers</div>
                    </div>
                    <div class="card">
                        <h3>Best Accuracy</h3>
                        <div class="value">{best_acc:.1f}%</div>
                        <div class="subtext">50% = chance</div>
                    </div>
                    <div class="card">
                        <h3>vs Random</h3>
                        <div class="value {delta_class}">{delta:+.1f}%</div>
                        <div class="subtext">baseline: {best_random:.1f}%</div>
                    </div>
                    <div class="card">
                        <h3>Best AUC</h3>
                        <div class="value">{best_auc:.3f}</div>
                        <div class="subtext">0.5 = random</div>
                    </div>
                </div>

                <div class="chart-container">
                    <h4>Accuracy by Layer</h4>
                    <div class="chart-legend">
                        <span class="legend-item"><span class="legend-color acc"></span> Probe Accuracy</span>
                        <span class="legend-item"><span class="legend-color rand"></span> Random Baseline</span>
                    </div>
                    <div class="bar-chart">
                        {''.join(chart_bars)}
                    </div>
                </div>

                <div class="heatmap-container">
                    <h4>Layer Heatmap</h4>
                    <div class="heatmap" style="grid-template-columns: repeat({len(metrics)}, 24px);">
                        {''.join(heatmap_cells)}
                    </div>
                    <div class="legend">
                        <span>50%</span>
                        <div class="legend-gradient"></div>
                        <span>100%</span>
                    </div>
                </div>

                <div class="chart-container">
                    <h4>Accuracy Curve</h4>
                    {_render_layer_accuracy_svg(metrics)}
                </div>

                <details class="metrics-details">
                    <summary>Full Metrics Table</summary>
                    <table>
                        <thead>
                            <tr>
                                <th>Layer</th>
                                <th title="Percentage of test samples correctly classified">Accuracy</th>
                                <th title="Area Under ROC Curve - ranking quality (0.5=random, 1.0=perfect)">AUC</th>
                                <th title="F1 Score - harmonic mean of precision and recall">F1</th>
                                <th title="Brier Score - mean squared error of probabilities (lower=better)">Brier</th>
                                <th title="Accuracy with random labels">Random</th>
                                <th title="Accuracy minus random baseline">Delta</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(table_rows)}
                        </tbody>
                    </table>
                </details>
            </div>
        """)

    # Probing-specific styles
    probing_styles = """
        <style>
            .intro-box {
                background: var(--bg-secondary);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 2rem;
            }

            .intro-box h3 {
                color: var(--accent);
                margin-bottom: 0.5rem;
            }

            .config-panel {
                background: var(--bg-tertiary);
                border-radius: 8px;
                padding: 1rem 1.5rem;
                margin-bottom: 2rem;
            }

            .config-panel h3 {
                font-size: 0.875rem;
                color: var(--text-secondary);
                margin-bottom: 0.75rem;
            }

            .config-grid {
                display: flex;
                flex-wrap: wrap;
                gap: 1.5rem;
            }

            .config-item {
                display: flex;
                flex-direction: column;
            }

            .config-label {
                font-size: 0.75rem;
                color: var(--text-secondary);
                text-transform: uppercase;
            }

            .config-value {
                font-weight: 600;
                color: var(--text-primary);
            }

            .download-btn {
                display: inline-flex;
                align-items: center;
                padding: 0.5rem 1rem;
                background: var(--accent);
                color: white;
                border-radius: 6px;
                text-decoration: none;
                font-size: 0.875rem;
                font-weight: 500;
                transition: background 0.2s;
            }

            .download-btn:hover {
                background: #1a8cd8;
            }

            .probe-section {
                background: var(--bg-secondary);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 2rem;
            }

            .probe-header h2 {
                margin-bottom: 0.25rem;
            }

            .probe-question {
                color: var(--accent);
                font-size: 1.1rem;
                font-style: italic;
            }

            .probe-explanation {
                background: var(--bg-tertiary);
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
            }

            .probe-labels {
                font-size: 0.875rem;
                color: var(--text-secondary);
                margin-top: 0.5rem;
            }

            .position-tabs {
                display: flex;
                gap: 0.5rem;
                margin-bottom: 1rem;
                border-bottom: 1px solid var(--border);
                padding-bottom: 0.5rem;
            }

            .position-tab {
                background: transparent;
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 0.5rem 1rem;
                color: var(--text-secondary);
                cursor: pointer;
                transition: all 0.2s;
            }

            .position-tab:hover {
                border-color: var(--accent);
                color: var(--text-primary);
            }

            .position-tab.active {
                background: var(--accent);
                border-color: var(--accent);
                color: white;
            }

            .position-content {
                display: none;
            }

            .position-content.active {
                display: block;
            }

            .position-description {
                background: var(--bg-primary);
                padding: 0.75rem 1rem;
                border-radius: 6px;
                margin-bottom: 1rem;
                font-size: 0.875rem;
                color: var(--text-secondary);
            }

            .chart-container {
                background: var(--bg-tertiary);
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
            }

            .chart-container h4 {
                margin-bottom: 0.5rem;
                font-size: 0.875rem;
                color: var(--text-secondary);
            }

            .chart-legend {
                display: flex;
                gap: 1.5rem;
                margin-bottom: 0.75rem;
                font-size: 0.75rem;
            }

            .legend-item {
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            .legend-color {
                width: 12px;
                height: 12px;
                border-radius: 2px;
            }

            .legend-color.acc {
                background: var(--accent);
            }

            .legend-color.rand {
                background: var(--text-secondary);
                opacity: 0.5;
            }

            .bar-chart {
                display: flex;
                align-items: flex-end;
                height: 120px;
                gap: 2px;
                padding: 0.5rem 0;
                border-bottom: 1px solid var(--border);
            }

            .chart-bar-group {
                flex: 1;
                display: flex;
                flex-direction: column;
                align-items: center;
                height: 100%;
                position: relative;
            }

            .chart-bar {
                width: 100%;
                max-width: 20px;
                position: absolute;
                bottom: 16px;
                border-radius: 2px 2px 0 0;
            }

            .chart-bar-acc {
                background: var(--accent);
                z-index: 2;
            }

            .chart-bar-rand {
                background: var(--text-secondary);
                opacity: 0.3;
                z-index: 1;
            }

            .chart-label {
                position: absolute;
                bottom: 0;
                font-size: 0.6rem;
                color: var(--text-secondary);
            }

            .metrics-details {
                margin-top: 1rem;
            }

            .metrics-details summary {
                cursor: pointer;
                color: var(--accent);
                font-size: 0.875rem;
                padding: 0.5rem 0;
            }

            .highlight-row {
                background: rgba(29, 155, 240, 0.1);
            }

            .highlight-row td {
                font-weight: 600;
            }

            th[title] {
                cursor: help;
                text-decoration: underline dotted;
            }

            .success { color: var(--success); }
            .warning { color: var(--warning); }
            .error { color: var(--error); }
        </style>
    """

    probing_script = ""

    content = f"""
        {probing_styles}

        <h1>Probing Analysis</h1>

        <div class="intro-box">
            <h3>What is Linear Probing?</h3>
            <p>Linear probing tests whether a concept is encoded in a model's hidden states in a <strong>linearly accessible</strong> way.
            A simple linear classifier is trained to predict labels from hidden states at specific token positions and layers.
            If the probe achieves accuracy significantly above chance (50%), the model encodes that information.</p>
        </div>

        <p style="margin-bottom: 1rem;">
            <strong>Model:</strong> {html_module.escape(model_name)} &nbsp;|&nbsp;
            <strong>Timestamp:</strong> {html_module.escape(timestamp)}
        </p>

        {config_html}

        {''.join(sections) if sections else '<p>No probing metrics found.</p>'}

        {probing_script}
    """

    return render_page("Probing", content, "Probing", selected_model=selected_model)


# Attention analysis explanations
ATTENTION_EXPLANATIONS = {
    "belief_tracking_score": {
        "title": "Belief Tracking Score",
        "question": "Does this head attend to what the protagonist BELIEVES vs what is TRUE?",
        "description": "Measures log(attention_to_belief / attention_to_world) for false-belief tasks. "
                       "Positive scores indicate the head preferentially attends to the protagonist's "
                       "believed location rather than the actual location.",
        "interpretation": "> 0.5: Strong belief tracking, 0-0.5: Moderate, < 0: Reality tracking",
    },
    "differential_score": {
        "title": "FB vs TB Differential",
        "question": "Does this head behave differently on false-belief vs true-belief tasks?",
        "description": "Difference in attention to belief token between FB and TB tasks. "
                       "Large positive values indicate the head specifically activates for FB scenarios.",
    },
}


def generate_probing_csv(selected_model: Optional[str] = None) -> str:
    """Generate CSV with probing predictions and task metadata."""
    dash = get_dashboard(selected_model)
    probing = dash.get_probing_results()
    predictions = dash.get_probing_predictions()
    tasks = dash.get_tasks()

    if not probing or not predictions or not tasks:
        return ""

    # Build task lookup
    task_lookup = {t["task_id"]: t for t in tasks}
    results = probing.get("results", {})

    # Collect all rows with predictions from all probe types
    rows: List[Dict[str, Any]] = []
    task_data: Dict[str, Dict[str, Any]] = {}

    for label_type, label_results in results.items():
        best_layer = label_results.get("best_layer")
        if best_layer is None:
            continue

        layer_preds = predictions.get(label_type, {}).get(str(best_layer), {})

        for task_id, pred in layer_preds.items():
            if task_id not in task_data:
                task = task_lookup.get(task_id, {})
                task_data[task_id] = {
                    "task_id": task_id,
                    "family_id": task.get("family_id", ""),
                    "task_type": task.get("task_type", ""),
                    "world": task.get("world", ""),
                    "belief": task.get("belief", ""),
                    "c1": task.get("c1", ""),
                    "c2": task.get("c2", ""),
                }
            task_data[task_id][f"{label_type}_prob"] = pred.get("prob", "")
            task_data[task_id][f"{label_type}_label"] = pred.get("label", "")
            task_data[task_id][f"{label_type}_layer"] = best_layer

    rows = list(task_data.values())
    if not rows:
        return ""

    # Write CSV
    output = io.StringIO()
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    return output.getvalue()


def render_attention(selected_model: Optional[str] = None) -> str:
    """Render attention analysis page with detailed explanations."""
    attention_dashboard = get_dashboard(selected_model)
    attention = attention_dashboard.get_attention_results()

    if not attention:
        attention_dir = RESULTS_DIR / "attention"
        available_attention_models = []
        if attention_dir.exists():
            for metrics_path in attention_dir.glob("*/attention_metrics.json"):
                available_attention_models.append(metrics_path.parent.name)
        available_attention_models.sort()
        expected_metrics_path = "N/A"
        if selected_model:
            expected_metrics_path = str(Path("results") / "attention" / selected_model.replace("/", "_") / "attention_metrics.json")
        available_models_display = ", ".join(available_attention_models) if available_attention_models else "None found"

        return render_page("Attention", f"""
            <h1>Attention Analysis</h1>
            <div class="empty-state">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
                <p>No attention results found.</p>
                <p>Run <code>python -m src.attention_analysis</code> to generate attention results.</p>
                <div class="debug-info">
                    <p><strong>Expected metrics path:</strong> <code>{html_module.escape(expected_metrics_path)}</code></p>
                    <p><strong>Available attention models:</strong> {html_module.escape(available_models_display)}</p>
                </div>
            </div>
        """, "Attention", selected_model=selected_model)

    model_name = attention.get("model_name", selected_model or "Unknown")
    timestamp = attention.get("timestamp", "N/A")
    config = attention.get("config", {})
    results = attention.get("results", {})
    summary = attention.get("summary", {})
    sanity = attention.get("sanity_checks", {})
    model_info = attention.get("model_info", {})
    timing = attention.get("timing", {})

    num_layers = model_info.get("num_layers", 0)
    num_heads = model_info.get("num_heads", 0)

    # Configuration panel
    config_html = f"""
        <div class="config-panel">
            <h3>Analysis Configuration</h3>
            <div class="config-grid">
                <div class="config-item">
                    <span class="config-label">Tasks Analyzed</span>
                    <span class="config-value">{summary.get('tasks_analyzed', 0)}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">FB Tasks</span>
                    <span class="config-value">{summary.get('fb_tasks', 0)}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">TB Tasks</span>
                    <span class="config-value">{summary.get('tb_tasks', 0)}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Layers</span>
                    <span class="config-value">{num_layers}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Heads/Layer</span>
                    <span class="config-value">{num_heads}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Time (ms)</span>
                    <span class="config-value">{timing.get('total_time_ms', 0):.0f}</span>
                </div>
            </div>
        </div>
    """

    # Sanity check panel
    sanity_warnings = sanity.get("token_position_warnings", 0) + sanity.get("attention_sum_warnings", 0)
    sanity_class = "success" if sanity_warnings == 0 else "warning" if sanity_warnings < 5 else "error"
    sanity_html = f"""
        <div class="sanity-panel">
            <h3>Sanity Checks</h3>
            <div class="config-grid">
                <div class="config-item">
                    <span class="config-label">Token Position Warnings</span>
                    <span class="config-value {sanity_class}">{sanity.get('token_position_warnings', 0)}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Attention Sum Warnings</span>
                    <span class="config-value {sanity_class}">{sanity.get('attention_sum_warnings', 0)}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Tasks Skipped</span>
                    <span class="config-value">{sanity.get('tasks_skipped', 0)}</span>
                </div>
            </div>
        </div>
    """

    # Top belief-tracking heads table
    top_belief_tracking = results.get("top_belief_tracking_heads", [])[:10]
    head_metrics = {(m["layer"], m["head"]): m for m in results.get("head_metrics", [])}

    tracking_rows = []
    for i, h in enumerate(top_belief_tracking, 1):
        layer, head, score = h["layer"], h["head"], h["score"]
        full_metrics = head_metrics.get((layer, head), {})
        belief_fb = full_metrics.get("belief_attention_fb", 0) * 100
        world_fb = full_metrics.get("world_attention_fb", 0) * 100
        ratio = full_metrics.get("belief_over_world_ratio", 1.0)

        score_class = "success" if score > 0.5 else "warning" if score > 0 else "error"
        tracking_rows.append(f"""
            <tr>
                <td>{i}</td>
                <td>Layer {layer}</td>
                <td>Head {head}</td>
                <td class="{score_class}"><strong>{score:.3f}</strong></td>
                <td>{belief_fb:.2f}%</td>
                <td>{world_fb:.2f}%</td>
                <td>{ratio:.2f}</td>
            </tr>
        """)

    # Top differentiating heads table
    top_differentiating = results.get("top_differentiating_heads", [])[:10]
    diff_rows = []
    for i, h in enumerate(top_differentiating, 1):
        layer, head = h["layer"], h["head"]
        diff = h["diff"]
        p_value = h.get("p_value", 1.0)
        full_metrics = head_metrics.get((layer, head), {})
        belief_fb = full_metrics.get("belief_attention_fb", 0) * 100
        belief_tb = full_metrics.get("belief_attention_tb", 0) * 100

        sig_badge = '<span class="badge success">sig</span>' if p_value < 0.05 else '<span class="badge info">ns</span>'
        diff_class = "success" if diff > 0.01 else "warning" if diff > 0 else "error"
        diff_rows.append(f"""
            <tr>
                <td>{i}</td>
                <td>Layer {layer}</td>
                <td>Head {head}</td>
                <td class="{diff_class}"><strong>{diff*100:.2f}%</strong></td>
                <td>{belief_fb:.2f}%</td>
                <td>{belief_tb:.2f}%</td>
                <td>{p_value:.4f} {sig_badge}</td>
            </tr>
        """)

    # Layer heatmap for belief tracking score
    layer_aggregates = results.get("layer_aggregates", {})
    heatmap_cells = []
    for layer in range(num_layers):
        layer_data = layer_aggregates.get(str(layer), {})
        max_score = layer_data.get("max_belief_tracking", 0)
        # Color: negative=red, 0=gray, positive=green
        if max_score > 0:
            hue = 120  # Green
            sat = min(70, max_score * 100)
            light = 25 + min(25, max_score * 30)
        else:
            hue = 0  # Red
            sat = min(70, abs(max_score) * 100)
            light = 25 + min(25, abs(max_score) * 30)

        color = f"hsl({hue}, {sat}%, {light}%)"
        heatmap_cells.append(
            f"<div class='heatmap-cell' title='Layer {layer}: max={max_score:.3f}' "
            f"style='background: {color}'>{layer}</div>"
        )

    # Summary cards
    best_head = summary.get("best_belief_tracking_head", {})
    best_score = summary.get("best_belief_tracking_score", 0)

    cards = f"""
        <div class="card-grid">
            <div class="card">
                <h3>Best Belief Tracking Head</h3>
                <div class="value">L{best_head.get('layer', '?')}H{best_head.get('head', '?')}</div>
                <div class="subtext">Layer {best_head.get('layer', '?')}, Head {best_head.get('head', '?')}</div>
            </div>
            <div class="card">
                <h3>Best Tracking Score</h3>
                <div class="value {'success' if best_score > 0.5 else 'warning' if best_score > 0 else 'error'}">{best_score:.3f}</div>
                <div class="subtext">> 0.5 = strong tracking</div>
            </div>
            <div class="card">
                <h3>Significant Heads</h3>
                <div class="value">{len([h for h in top_differentiating if h.get('p_value', 1) < 0.05])}</div>
                <div class="subtext">p < 0.05 for FB vs TB</div>
            </div>
            <div class="card">
                <h3>Total Heads</h3>
                <div class="value">{num_layers * num_heads}</div>
                <div class="subtext">{num_layers} layers x {num_heads} heads</div>
            </div>
        </div>
    """

    # Attention-specific styles
    attention_styles = """
        <style>
            .intro-box {
                background: var(--bg-secondary);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 2rem;
            }

            .intro-box h3 {
                color: var(--accent);
                margin-bottom: 0.5rem;
            }

            .config-panel, .sanity-panel {
                background: var(--bg-tertiary);
                border-radius: 8px;
                padding: 1rem 1.5rem;
                margin-bottom: 1rem;
            }

            .config-panel h3, .sanity-panel h3 {
                font-size: 0.875rem;
                color: var(--text-secondary);
                margin-bottom: 0.75rem;
            }

            .config-grid {
                display: flex;
                flex-wrap: wrap;
                gap: 1.5rem;
            }

            .config-item {
                display: flex;
                flex-direction: column;
            }

            .config-label {
                font-size: 0.75rem;
                color: var(--text-secondary);
                text-transform: uppercase;
            }

            .config-value {
                font-weight: 600;
                color: var(--text-primary);
            }

            .attention-section {
                background: var(--bg-secondary);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 2rem;
            }

            .attention-section h2 {
                margin-bottom: 0.5rem;
            }

            .section-description {
                color: var(--text-secondary);
                margin-bottom: 1rem;
                font-size: 0.9rem;
            }

            .success { color: var(--success); }
            .warning { color: var(--warning); }
            .error { color: var(--error); }

            .badge.success { background: rgba(0, 186, 124, 0.2); color: var(--success); }
            .badge.info { background: rgba(29, 155, 240, 0.2); color: var(--accent); }
        </style>
    """

    content = f"""
        {attention_styles}

        <h1>Attention Analysis</h1>

        <div class="intro-box">
            <h3>What is Attention Analysis?</h3>
            <p>Attention analysis examines which tokens the model attends to when answering Theory of Mind questions.
            We measure attention from the <strong>question tokens</strong> (where the model forms its answer) to
            <strong>belief tokens</strong> (where the protagonist's belief is stated) vs <strong>world tokens</strong>
            (where the actual reality is stated).</p>
            <p style="margin-top: 0.5rem;">A head with high <strong>belief tracking score</strong> attends more to what
            the protagonist <em>believes</em> than to what is <em>actually true</em> - suggesting it may be involved
            in perspective-taking.</p>
        </div>

        <p style="margin-bottom: 1rem;">
            <strong>Model:</strong> {html_module.escape(model_name)} &nbsp;|&nbsp;
            <strong>Timestamp:</strong> {html_module.escape(timestamp)}
        </p>

        {config_html}
        {sanity_html}

        {cards}

        <div class="attention-section">
            <h2>Top Belief-Tracking Heads</h2>
            <p class="section-description">
                {ATTENTION_EXPLANATIONS['belief_tracking_score']['description']}
                <br><strong>Interpretation:</strong> {ATTENTION_EXPLANATIONS['belief_tracking_score']['interpretation']}
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Layer</th>
                        <th>Head</th>
                        <th title="log(belief_attn/world_attn) on FB tasks">Tracking Score</th>
                        <th title="Mean attention to belief token on FB tasks">Belief Attn (FB)</th>
                        <th title="Mean attention to world token on FB tasks">World Attn (FB)</th>
                        <th title="Belief/World ratio">Ratio</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(tracking_rows) if tracking_rows else '<tr><td colspan="7">No data</td></tr>'}
                </tbody>
            </table>
        </div>

        <div class="attention-section">
            <h2>Top Differentiating Heads (FB vs TB)</h2>
            <p class="section-description">
                {ATTENTION_EXPLANATIONS['differential_score']['description']}
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Layer</th>
                        <th>Head</th>
                        <th title="Belief attention (FB - TB)">Differential</th>
                        <th>Belief Attn (FB)</th>
                        <th>Belief Attn (TB)</th>
                        <th title="t-test p-value">p-value</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(diff_rows) if diff_rows else '<tr><td colspan="7">No data</td></tr>'}
                </tbody>
            </table>
        </div>

        <div class="heatmap-container">
            <h4>Layer Heatmap (Max Belief Tracking Score per Layer)</h4>
            <p class="section-description">Green = positive (belief tracking), Red = negative (reality tracking)</p>
            <div class="heatmap" style="grid-template-columns: repeat({num_layers}, 24px);">
                {''.join(heatmap_cells)}
            </div>
            <div class="legend">
                <span>Reality</span>
                <div class="legend-gradient" style="background: linear-gradient(to right, hsl(0, 50%, 30%), hsl(0, 0%, 25%), hsl(120, 50%, 30%));"></div>
                <span>Belief</span>
            </div>
        </div>
    """

    return render_page("Attention", content, "Attention", selected_model=selected_model)


def render_api_stats(selected_model: Optional[str] = None) -> str:
    """Return JSON stats for API."""
    summary = dashboard.get_study_summary()
    if not summary:
        return json.dumps({"error": "No data"})
    return json.dumps(summary)


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard."""

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def do_GET(self):
        """Handle GET requests."""
        global dashboard

        parsed = urlparse(self.path)
        path = parsed.path
        query_params = parse_qs(parsed.query)

        # Extract model from query params
        selected_model = query_params.get("model", [None])[0]

        # Update global dashboard with selected model
        dashboard = get_dashboard(selected_model)

        # Route requests
        if path == "/" or path == "":
            content = render_overview(selected_model)
            content_type = "text/html"
        elif path == "/tasks":
            content = render_tasks(selected_model)
            content_type = "text/html"
        elif path == "/probing":
            content = render_probing(selected_model)
            content_type = "text/html"
        elif path == "/probing/download/csv":
            content = generate_probing_csv(selected_model)
            if not content:
                self.send_error(404, "No probing data available")
                return
            content_type = "text/csv"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Disposition", "attachment; filename=probing_predictions.csv")
            self.send_header("Content-Length", len(content.encode()))
            self.end_headers()
            self.wfile.write(content.encode())
            return
        elif path == "/attention":
            content = render_attention(selected_model)
            content_type = "text/html"
        elif path == "/api/stats":
            content = render_api_stats(selected_model)
            content_type = "application/json"
        else:
            self.send_error(404, "Not Found")
            return

        # Send response
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(content.encode()))
        self.end_headers()
        self.wfile.write(content.encode())


def main():
    """Run the dashboard server."""
    parser = argparse.ArgumentParser(description="MechInt Dashboard Server")
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT, help="Port to run on")
    parser.add_argument("--host", "-H", type=str, default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), DashboardHandler)
    print(f"MechInt Dashboard running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
