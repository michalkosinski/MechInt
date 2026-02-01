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
            # Convert model name to safe directory name (e.g., "Qwen/Qwen2.5-3B-Instruct" -> "Qwen--Qwen2.5-3B-Instruct")
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
                    # Model name in filename: "Qwen/Qwen2.5-3B-Instruct" -> "Qwen_Qwen2.5-3B-Instruct"
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
    ]

    nav_html = "\n".join([
        f'<a href="{url}{model_qs}" class="{"active" if nav_active == name else ""}">{name}</a>'
        for name, url in nav_items
    ])

    # Build model selector dropdown
    models = get_model_list()
    model_options = ""
    for model in models:
        # Get short display name (last part after /)
        display_name = model.split("/")[-1] if "/" in model else model
        selected = "selected" if model == selected_model else ""
        model_options += f'<option value="{html_module.escape(model)}" {selected}>{html_module.escape(display_name)}</option>'

    # Add "None" option if no model is selected and models exist
    none_selected = "selected" if not selected_model else ""

    model_selector = ""
    if models:
        model_selector = f'''
            <div class="model-selector">
                <label for="model-select">Model:</label>
                <select id="model-select" onchange="switchModel(this.value)">
                    <option value="" {none_selected}>Default</option>
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
            background: linear-gradient(to right, #1a1f26, #1d9bf0, #00ba7c);
        }}

        .empty-state {{
            text-align: center;
            padding: 4rem;
            color: var(--text-secondary);
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
