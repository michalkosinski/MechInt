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
    """Render the overview page."""
    behavioral = dashboard.get_behavioral_results()

    if not behavioral:
        return render_page("Overview", """
            <h1>Overview</h1>
            <div class="empty-state">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p>No study results found.</p>
                <p>Run <code>python -m src.run_study</code> to generate results.</p>
            </div>
        """, "Overview", selected_model=selected_model)

    # Get stats from behavioral results (first file if multiple)
    behavioral_data = behavioral[0] if behavioral else {}
    summary = behavioral_data.get("summary", {})
    model_name = behavioral_data.get("model", selected_model or "Unknown")
    timestamp = behavioral_data.get("timestamp", "N/A")

    # Build cards
    cards = []

    acc = summary.get("overall_accuracy", 0) * 100
    acc_class = "success" if acc >= 70 else "warning" if acc >= 50 else "error"
    cards.append(f"""
        <div class="card">
            <h3>Overall Accuracy</h3>
            <div class="value {acc_class}">{acc:.1f}%</div>
            <div class="subtext">{summary.get('num_tasks', 0)} tasks</div>
        </div>
    """)

    fb_acc = summary.get("false_belief_accuracy", 0) * 100
    cards.append(f"""
        <div class="card">
            <h3>False Belief Accuracy</h3>
            <div class="value">{fb_acc:.1f}%</div>
            <div class="subtext">Tests belief vs reality</div>
        </div>
    """)

    tb_acc = summary.get("true_belief_accuracy", 0) * 100
    cards.append(f"""
        <div class="card">
            <h3>True Belief Accuracy</h3>
            <div class="value">{tb_acc:.1f}%</div>
            <div class="subtext">Control condition</div>
        </div>
    """)

    num_uncertain = summary.get("num_uncertain", 0)
    if num_uncertain > 0:
        cards.append(f"""
            <div class="card">
                <h3>Uncertain Responses</h3>
                <div class="value warning">{num_uncertain}</div>
                <div class="subtext">Could not parse answer</div>
            </div>
        """)

    # Model info
    model_info = f"""
        <h2>Study Configuration</h2>
        <table>
            <tr><th>Model</th><td>{model_name}</td></tr>
            <tr><th>Timestamp</th><td>{timestamp}</td></tr>
            <tr><th>Tasks</th><td>{summary.get('num_tasks', 0)}</td></tr>
        </table>
    """

    content = f"""
        <h1>Study Overview</h1>
        <div class="card-grid">
            {''.join(cards)}
        </div>
        {model_info}
    """

    return render_page("Overview", content, "Overview", selected_model=selected_model)

def render_tasks(selected_model: Optional[str] = None) -> str:
    """Render the tasks page with sorting, filtering, and parsed responses."""
    tasks = dashboard.get_tasks()
    behavioral = dashboard.get_behavioral_results()

    if not tasks:
        return render_page("Tasks", """
            <h1>Tasks</h1>
            <div class="empty-state">
                <p>No tasks found.</p>
                <p>Run the study to generate tasks.</p>
            </div>
        """, "Tasks", selected_model=selected_model)

    # Build results lookup
    results_by_id = {}
    for b in behavioral:
        for detail in b.get("details", []):
            results_by_id[detail["task_id"]] = detail

    # Sort tasks by ID (natural sort)
    def sort_key(t):
        task_id = t.get("task_id", "")
        # Extract prefix, number, and suffix (e.g., "tb_1r" -> ("tb", 1, "r"))
        match = re.match(r"([a-z]+)_(\d+)(r?)", task_id)
        if match:
            return (match.group(1), int(match.group(2)), match.group(3))
        return (task_id, 0, "")

    tasks = sorted(tasks, key=sort_key)

    rows = []
    for idx, task in enumerate(tasks):
        task_id = task.get("task_id", "")
        task_type = task.get("task_type", "")
        prompt = task.get("full_prompt", "")
        expected = task.get("expected_answer", "")
        result = results_by_id.get(task_id, {})

        if result:
            raw = result.get("response", "N/A")
            parsed = result.get("parsed_response", raw)
            is_correct = result.get("is_correct", False)
            is_uncertain = result.get("is_uncertain", False)
            if is_uncertain:
                badge = '<span class="badge warning">?</span>'
            else:
                badge = f'<span class="badge {"success" if is_correct else "error"}">{"✓" if is_correct else "✗"}</span>'
        else:
            raw = "-"
            parsed = "-"
            is_correct = False
            is_uncertain = False
            badge = '<span class="badge info">-</span>'

        # Type badge with color coding
        type_colors = {
            "true_belief": "success",
            "false_belief": "error",
            "mismatch": "warning",
            "negation_trap": "info",
            "belief_overwrite": "purple"
        }
        base_type = task_type.replace("_reversed", "")
        color = type_colors.get(base_type, "info")
        type_short = task_type.replace("_reversed", " ↔").replace("_", " ").title()
        type_badge = f'<span class="badge {color}">{type_short}</span>'

        # Escape for HTML
        raw_escaped = html_module.escape(raw)
        parsed_escaped = html_module.escape(parsed)
        prompt_escaped = html_module.escape(prompt)
        expected_escaped = html_module.escape(expected)

        rows.append(f"""
            <tr data-type="{base_type}" data-correct="{'true' if is_correct else 'false'}" data-uncertain="{'true' if is_uncertain else 'false'}" data-id="{task_id}">
                <td class="col-id">{task_id}</td>
                <td class="col-type">{type_badge}</td>
                <td class="col-prompt"><code title="{prompt_escaped}">{prompt_escaped[:80]}{'...' if len(prompt) > 80 else ''}</code></td>
                <td class="col-expected">{expected_escaped}</td>
                <td class="col-raw"><code>{raw_escaped}</code></td>
                <td class="col-parsed"><strong>{parsed_escaped}</strong></td>
                <td class="col-result">{badge}</td>
                <td class="col-expand">
                    <button class="btn-expand" onclick="toggleDetails({idx})">▶</button>
                </td>
            </tr>
            <tr class="details-row" id="details-{idx}" style="display:none;">
                <td colspan="8">
                    <div class="details-content">
                        <div class="detail-section">
                            <strong>Full Prompt:</strong>
                            <pre>{prompt_escaped}</pre>
                        </div>
                        <div class="detail-section">
                            <strong>Full Response (raw):</strong>
                            <pre>{raw_escaped}</pre>
                        </div>
                        <div class="detail-section">
                            <strong>Expected:</strong> {expected_escaped} |
                            <strong>Parsed:</strong> {parsed_escaped} |
                            <strong>Match:</strong> {parsed.lower() == expected.lower()}
                        </div>
                    </div>
                </td>
            </tr>
        """)

    # Count stats
    total = len(tasks)

    content = f"""
        <h1>Tasks ({total})</h1>

        <div class="filters" style="margin-bottom: 1rem; display: flex; gap: 1rem; flex-wrap: wrap;">
            <div>
                <label>Type: </label>
                <select id="filter-type" onchange="filterTasks()">
                    <option value="">All</option>
                    <option value="true_belief">True Belief</option>
                    <option value="false_belief">False Belief</option>
                    <option value="mismatch">Mismatch</option>
                    <option value="negation_trap">Negation Trap</option>
                    <option value="belief_overwrite">Belief Overwrite</option>
                </select>
            </div>
            <div>
                <label>Result: </label>
                <select id="filter-result" onchange="filterTasks()">
                    <option value="">All</option>
                    <option value="true">Correct</option>
                    <option value="false">Wrong</option>
                    <option value="uncertain">Uncertain</option>
                </select>
            </div>
            <div>
                <label>Search: </label>
                <input type="text" id="filter-search" placeholder="Search prompts..." onkeyup="filterTasks()" style="width: 200px;">
            </div>
        </div>

        <table id="tasks-table">
            <thead>
                <tr>
                    <th class="sortable" onclick="sortTable(0)">ID ▼</th>
                    <th class="sortable" onclick="sortTable(1)">Type</th>
                    <th>Prompt</th>
                    <th class="sortable" onclick="sortTable(3)">Expected</th>
                    <th>Raw</th>
                    <th>Parsed</th>
                    <th class="sortable" onclick="sortTable(6)">Result</th>
                    <th></th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>

        <style>
            .filters select, .filters input {{
                padding: 0.3rem 0.5rem;
                border: 1px solid var(--border);
                border-radius: 4px;
                background: var(--card);
            }}
            .sortable {{ cursor: pointer; }}
            .sortable:hover {{ background: var(--hover); }}
            .btn-expand {{
                background: none;
                border: none;
                cursor: pointer;
                font-size: 0.8rem;
                padding: 0.2rem 0.5rem;
            }}
            .details-row td {{
                background: var(--hover);
                padding: 1rem;
            }}
            .details-content {{
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }}
            .detail-section pre {{
                background: var(--card);
                padding: 0.5rem;
                border-radius: 4px;
                white-space: pre-wrap;
                word-break: break-word;
                margin: 0.25rem 0;
            }}
            .col-id {{ width: 60px; }}
            .col-type {{ width: 120px; }}
            .col-expected, .col-parsed {{ width: 80px; }}
            .col-raw {{ max-width: 150px; overflow: hidden; text-overflow: ellipsis; }}
            .col-result {{ width: 50px; text-align: center; }}
            .col-expand {{ width: 30px; }}
            .badge.warning {{ background: #f59e0b; color: white; }}
            .badge.purple {{ background: #8b5cf6; color: white; }}
        </style>

        <script>
            let sortCol = 0;
            let sortAsc = true;

            function toggleDetails(idx) {{
                const row = document.getElementById('details-' + idx);
                const btn = row.previousElementSibling.querySelector('.btn-expand');
                if (row.style.display === 'none') {{
                    row.style.display = 'table-row';
                    btn.textContent = '▼';
                }} else {{
                    row.style.display = 'none';
                    btn.textContent = '▶';
                }}
            }}

            function filterTasks() {{
                const typeFilter = document.getElementById('filter-type').value;
                const resultFilter = document.getElementById('filter-result').value;
                const searchFilter = document.getElementById('filter-search').value.toLowerCase();

                const rows = document.querySelectorAll('#tasks-table tbody tr:not(.details-row)');
                rows.forEach(row => {{
                    const type = row.dataset.type;
                    const correct = row.dataset.correct;
                    const uncertain = row.dataset.uncertain;
                    const prompt = row.querySelector('.col-prompt code').textContent.toLowerCase();

                    let show = true;
                    if (typeFilter && type !== typeFilter) show = false;
                    if (resultFilter === 'uncertain' && uncertain !== 'true') show = false;
                    else if (resultFilter && resultFilter !== 'uncertain' && correct !== resultFilter) show = false;
                    if (searchFilter && !prompt.includes(searchFilter)) show = false;

                    row.style.display = show ? '' : 'none';
                    // Also hide details row
                    const detailsRow = row.nextElementSibling;
                    if (detailsRow && detailsRow.classList.contains('details-row')) {{
                        detailsRow.style.display = 'none';
                        row.querySelector('.btn-expand').textContent = '▶';
                    }}
                }});
            }}

            function sortTable(col) {{
                const table = document.getElementById('tasks-table');
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr:not(.details-row)'));

                if (col === sortCol) {{
                    sortAsc = !sortAsc;
                }} else {{
                    sortCol = col;
                    sortAsc = true;
                }}

                rows.sort((a, b) => {{
                    let aVal = a.cells[col].textContent.trim();
                    let bVal = b.cells[col].textContent.trim();

                    // Natural sort for IDs
                    if (col === 0) {{
                        const parseId = (id) => {{
                            const match = id.match(/([a-z]+)_(\\d+)(r?)/);
                            if (match) return [match[1], parseInt(match[2]), match[3]];
                            return [id, 0, ''];
                        }};
                        const [aPrefix, aNum, aSuffix] = parseId(aVal);
                        const [bPrefix, bNum, bSuffix] = parseId(bVal);
                        if (aPrefix !== bPrefix) return aPrefix.localeCompare(bPrefix);
                        if (aNum !== bNum) return aNum - bNum;
                        return aSuffix.localeCompare(bSuffix);
                    }}

                    return aVal.localeCompare(bVal);
                }});

                if (!sortAsc) rows.reverse();

                // Re-append rows with their detail rows
                rows.forEach((row, idx) => {{
                    // Find the details row by data-id attribute
                    const taskId = row.dataset.id;
                    const detailsRow = row.nextElementSibling;
                    tbody.appendChild(row);
                    if (detailsRow && detailsRow.classList.contains('details-row')) {{
                        tbody.appendChild(detailsRow);
                    }}
                }});
            }}
        </script>
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
