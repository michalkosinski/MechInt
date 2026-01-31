#!/usr/bin/env python3
"""
MechInt Web Dashboard

Elegant web interface for viewing Theory of Mind attention analysis results.
Features:
- Behavioral accuracy overview
- Attention heatmaps (layers x heads)
- ToM head identification
- Intervention experiment results
- Per-task attention visualization

Usage:
    python web_dashboard.py [--port PORT] [--host HOST]
"""

import argparse
import json
import os
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

    def __init__(self, results_dir: Path = RESULTS_DIR):
        self.results_dir = Path(results_dir)

    def get_study_summary(self) -> Optional[Dict]:
        """Load study summary."""
        path = self.results_dir / "study_summary.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def get_behavioral_results(self) -> List[Dict]:
        """Load all behavioral result files."""
        results = []
        behavioral_dir = self.results_dir / "behavioral"
        if behavioral_dir.exists():
            for f in behavioral_dir.glob("*.json"):
                data = json.loads(f.read_text())
                data["filename"] = f.name
                results.append(data)
        return results

    def get_attention_heatmap(self) -> Optional[Dict]:
        """Load attention heatmap data."""
        path = self.results_dir / "attention" / "heatmap_data.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def get_head_scores(self) -> List[Dict]:
        """Load head scores."""
        path = self.results_dir / "attention" / "head_scores.json"
        if path.exists():
            return json.loads(path.read_text())
        return []

    def get_intervention_results(self, intervention_type: str) -> Optional[Dict]:
        """Load intervention results (ablation or boost)."""
        path = self.results_dir / "interventions" / f"{intervention_type}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def get_multihead_intervention_results(self) -> Dict[int, Dict]:
        """Load intervention results for different numbers of heads."""
        results = {}
        interventions_dir = self.results_dir / "interventions"
        if not interventions_dir.exists():
            return results

        # Look for files like ablation_5heads.json, boost_10heads.json, etc.
        for f in interventions_dir.glob("*_*heads.json"):
            name = f.stem  # e.g., "ablation_5heads"
            parts = name.rsplit("_", 1)
            if len(parts) == 2:
                intervention_type = parts[0]  # "ablation" or "boost"
                num_heads = int(parts[1].replace("heads", ""))
                if num_heads not in results:
                    results[num_heads] = {}
                data = json.loads(f.read_text())
                results[num_heads][intervention_type] = data
        return results

    def get_tasks(self) -> List[Dict]:
        """Load tasks."""
        if TASKS_FILE.exists():
            data = json.loads(TASKS_FILE.read_text())
            return data.get("tasks", [])
        return []

    def get_layer_selectivity(self) -> List[Dict]:
        """Load layer selectivity data."""
        path = self.results_dir / "attention" / "layer_selectivity.json"
        if path.exists():
            return json.loads(path.read_text())
        return []

    def get_token_attention(self) -> List[Dict]:
        """Load token attention data."""
        path = self.results_dir / "attention" / "token_attention.json"
        if path.exists():
            return json.loads(path.read_text())
        return []

    def get_attention_matrices(self) -> List[Dict]:
        """Load attention matrix data."""
        path = self.results_dir / "attention" / "attention_matrices.json"
        if path.exists():
            return json.loads(path.read_text())
        return []


# Global dashboard instance
dashboard = Dashboard()


def render_page(title: str, content: str, nav_active: str = "") -> str:
    """Render a full HTML page with navigation."""
    nav_items = [
        ("Overview", "/"),
        ("Attention", "/attention"),
        ("ToM Heads", "/heads"),
        ("Interventions", "/interventions"),
        ("Tasks", "/tasks"),
    ]

    # Sub-navigation for Attention section
    attention_subnav = ""
    if nav_active == "Attention":
        attention_subnav = '''
            <div class="subnav">
                <a href="/attention">Heatmaps</a>
                <a href="/attention/layers">Layers</a>
                <a href="/attention/token">Tokens</a>
                <a href="/attention/matrix">Matrix</a>
                <a href="/attention/flow">Flow</a>
                <a href="/attention/clustering">Clustering</a>
            </div>
        '''

    nav_html = "\n".join([
        f'<a href="{url}" class="{"active" if nav_active == name else ""}">{name}</a>'
        for name, url in nav_items
    ])
    nav_html += attention_subnav

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

        .subnav {{
            display: flex;
            gap: 0.5rem;
            margin-left: auto;
            padding-left: 1rem;
            border-left: 1px solid var(--border);
        }}

        .subnav a {{
            font-size: 0.875rem;
            padding: 0.375rem 0.75rem;
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
    </style>
</head>
<body>
    <nav>
        <span class="logo">MechInt</span>
        {nav_html}
    </nav>
    <main>
        {content}
    </main>
    <footer>
        MechInt Dashboard &middot; Port {DEFAULT_PORT}
    </footer>
</body>
</html>"""


def render_overview() -> str:
    """Render the overview page."""
    summary = dashboard.get_study_summary()
    behavioral = dashboard.get_behavioral_results()

    if not summary and not behavioral:
        return render_page("Overview", """
            <h1>Overview</h1>
            <div class="empty-state">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p>No study results found.</p>
                <p>Run <code>python -m src.run_study</code> to generate results.</p>
            </div>
        """, "Overview")

    # Build cards
    cards = []

    if summary:
        acc = summary.get("behavioral_accuracy", 0) * 100
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

        if summary.get("top_tom_heads"):
            num_heads = len(summary["top_tom_heads"])
            cards.append(f"""
                <div class="card">
                    <h3>ToM Heads Identified</h3>
                    <div class="value">{num_heads}</div>
                    <div class="subtext">See Attention tab</div>
                </div>
            """)

    # Model info
    model_info = ""
    if summary and summary.get("config"):
        config = summary["config"]
        model_info = f"""
            <h2>Study Configuration</h2>
            <table>
                <tr><th>Model</th><td>{config.get('model_name', 'N/A')}</td></tr>
                <tr><th>Timestamp</th><td>{summary.get('timestamp', 'N/A')}</td></tr>
                <tr><th>Tasks</th><td>{config.get('num_false_belief_tasks', 0)} FB + {config.get('num_true_belief_tasks', 0)} TB</td></tr>
                <tr><th>Attention Analyzed</th><td>{'Yes' if summary.get('attention_analyzed') else 'No'}</td></tr>
                <tr><th>Interventions Run</th><td>{'Yes' if summary.get('interventions_run') else 'No'}</td></tr>
            </table>
        """

    # Intervention summary
    intervention_info = ""
    if summary and summary.get("interventions_run"):
        abl_delta = summary.get("ablation_accuracy_delta")
        boost_delta = summary.get("boost_accuracy_delta")

        if abl_delta is not None or boost_delta is not None:
            intervention_info = "<h2>Intervention Effects</h2><div class='card-grid'>"

            if abl_delta is not None:
                delta_class = "delta-negative" if abl_delta < 0 else "delta-positive"
                intervention_info += f"""
                    <div class="card">
                        <h3>Ablation Effect</h3>
                        <div class="value {delta_class}">{abl_delta:+.1%}</div>
                        <div class="subtext">Accuracy change when ToM heads ablated</div>
                    </div>
                """

            if boost_delta is not None:
                delta_class = "delta-positive" if boost_delta > 0 else "delta-negative"
                intervention_info += f"""
                    <div class="card">
                        <h3>Boost Effect</h3>
                        <div class="value {delta_class}">{boost_delta:+.1%}</div>
                        <div class="subtext">Accuracy change when ToM heads boosted</div>
                    </div>
                """

            intervention_info += "</div>"

    content = f"""
        <h1>Study Overview</h1>
        <div class="card-grid">
            {''.join(cards)}
        </div>
        {model_info}
        {intervention_info}
    """

    return render_page("Overview", content, "Overview")


def render_attention() -> str:
    """Render the attention analysis page."""
    heatmap_data = dashboard.get_attention_heatmap()
    head_scores = dashboard.get_head_scores()

    if not heatmap_data:
        return render_page("Attention", """
            <h1>Attention Analysis</h1>
            <div class="empty-state">
                <p>No attention data available.</p>
                <p>Run the study with attention analysis enabled.</p>
            </div>
        """, "Attention")

    num_layers = heatmap_data.get("num_layers", 0)
    num_heads = heatmap_data.get("num_heads", 0)

    # Get top 10 heads for highlighting
    top_heads = set()
    if head_scores:
        sorted_heads = sorted(head_scores, key=lambda x: -x.get("tom_score", 0))[:10]
        top_heads = {(h["layer"], h["head"]) for h in sorted_heads}

    def diverging_color(value: float) -> str:
        """Generate color using diverging colormap centered at 1.0.

        Blue (< 1.0) -> White (= 1.0) -> Red (> 1.0)
        """
        # Clamp value to [0.5, 2.0] range for color scaling
        clamped = max(0.5, min(2.0, value))

        if clamped < 1.0:
            # Blue gradient: interpolate from dark blue to neutral
            t = (clamped - 0.5) / 0.5  # 0 at 0.5, 1 at 1.0
            r = int(30 + t * (70 - 30))
            g = int(60 + t * (70 - 60))
            b = int(180 + t * (100 - 180))
        else:
            # Red gradient: interpolate from neutral to red
            t = (clamped - 1.0) / 1.0  # 0 at 1.0, 1 at 2.0
            r = int(70 + t * (220 - 70))
            g = int(70 - t * 50)
            b = int(100 - t * 70)

        return f"rgb({r},{g},{b})"

    def render_heatmap(data: List[List[float]], title: str) -> str:
        """Render a single heatmap with diverging colormap and top head highlighting."""
        cells = []

        # Header row with head numbers
        cells.append('<div class="heatmap-label"></div>')
        for h in range(num_heads):
            cells.append(f'<div class="heatmap-label">H{h}</div>')

        # Data rows
        for layer_idx, row in enumerate(data):
            cells.append(f'<div class="heatmap-label">L{layer_idx}</div>')
            for head_idx, value in enumerate(row):
                color = diverging_color(value)
                is_top = (layer_idx, head_idx) in top_heads
                border_style = "border: 2px solid #ffd700;" if is_top else ""
                cells.append(
                    f'<div class="heatmap-cell" style="background:{color}; {border_style}" '
                    f'title="L{layer_idx} H{head_idx}: {value:.2f}{"  ★ Top ToM" if is_top else ""}"></div>'
                )

        grid_cols = num_heads + 1
        return f"""
            <h3>{title}</h3>
            <div class="heatmap" style="grid-template-columns: repeat({grid_cols}, auto)">
                {''.join(cells)}
            </div>
        """

    fb_heatmap = render_heatmap(heatmap_data.get("false_belief", []), "False Belief Condition")
    tb_heatmap = render_heatmap(heatmap_data.get("true_belief", []), "True Belief Condition")

    # Improved legend with diverging colormap
    legend_svg = '''
        <svg width="280" height="40" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="divergingGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:rgb(30,60,180)"/>
                    <stop offset="50%" style="stop-color:rgb(70,70,100)"/>
                    <stop offset="100%" style="stop-color:rgb(220,20,30)"/>
                </linearGradient>
            </defs>
            <rect x="40" y="5" width="200" height="16" fill="url(#divergingGrad)" rx="3"/>
            <text x="40" y="35" fill="#8b98a5" font-size="11">0.5</text>
            <text x="140" y="35" fill="#8b98a5" font-size="11" text-anchor="middle">1.0</text>
            <text x="240" y="35" fill="#8b98a5" font-size="11" text-anchor="end">2.0+</text>
        </svg>
    '''

    content = f"""
        <h1>Attention Analysis</h1>
        <p style="color: var(--text-secondary); margin-bottom: 1rem;">
            Belief-Reality Attention Ratio by Layer and Head.
            Blue indicates attention to reality location, red indicates attention to believed location.
            Gold border marks top 10 ToM heads.
        </p>

        <div class="card-grid" style="margin-bottom: 1.5rem;">
            <a href="/attention/layers" class="card" style="text-decoration: none;">
                <h3>📈 Layer Selectivity</h3>
                <p style="color: var(--text-secondary);">View how ToM selectivity changes across layers</p>
            </a>
            <a href="/attention/token" class="card" style="text-decoration: none;">
                <h3>📊 Token Attention</h3>
                <p style="color: var(--text-secondary);">View per-token attention weights for top heads</p>
            </a>
        </div>

        <div class="heatmap-container">
            {fb_heatmap}
            <div class="legend">
                <span>Reality (ratio &lt; 1)</span>
                {legend_svg}
                <span>Belief (ratio &gt; 1)</span>
            </div>
        </div>

        <div class="heatmap-container" style="margin-top: 1.5rem;">
            {tb_heatmap}
        </div>
    """

    return render_page("Attention", content, "Attention")


def render_layer_selectivity() -> str:
    """Render the layer selectivity visualization page."""
    selectivity_data = dashboard.get_layer_selectivity()

    if not selectivity_data:
        return render_page("Layer Selectivity", """
            <h1>Selectivity Across Layers</h1>
            <div class="empty-state">
                <p>No layer selectivity data available.</p>
                <p>Re-run the study to generate this data.</p>
            </div>
        """, "Attention")

    # Organize by task type and layer
    fb_data = {}
    tb_data = {}
    for item in selectivity_data:
        layer = item["layer"]
        if item["task_type"] == "false_belief":
            fb_data[layer] = item["mean_belief_ratio"]
        else:
            tb_data[layer] = item["mean_belief_ratio"]

    if not fb_data and not tb_data:
        return render_page("Layer Selectivity", """
            <h1>Selectivity Across Layers</h1>
            <div class="empty-state">
                <p>No layer data found.</p>
            </div>
        """, "Attention")

    # Get sorted layers
    all_layers = sorted(set(fb_data.keys()) | set(tb_data.keys()))
    num_layers = len(all_layers)

    # SVG dimensions
    width = 800
    height = 400
    margin_left = 70
    margin_right = 30
    margin_top = 40
    margin_bottom = 60
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    # Find y-axis range
    all_values = list(fb_data.values()) + list(tb_data.values())
    y_min = min(0.5, min(all_values) - 0.1)
    y_max = max(2.0, max(all_values) + 0.1)
    y_range = y_max - y_min

    def x_coord(layer_idx):
        return margin_left + (layer_idx / max(num_layers - 1, 1)) * plot_width

    def y_coord(value):
        return margin_top + (1 - (value - y_min) / y_range) * plot_height

    # Build path for false belief line
    fb_points = []
    for i, layer in enumerate(all_layers):
        if layer in fb_data:
            x = x_coord(i)
            y = y_coord(fb_data[layer])
            fb_points.append(f"{x},{y}")
    fb_path = f'<polyline points="{" ".join(fb_points)}" fill="none" stroke="#f4212e" stroke-width="3"/>'

    # Build path for true belief line
    tb_points = []
    for i, layer in enumerate(all_layers):
        if layer in tb_data:
            x = x_coord(i)
            y = y_coord(tb_data[layer])
            tb_points.append(f"{x},{y}")
    tb_path = f'<polyline points="{" ".join(tb_points)}" fill="none" stroke="#1d9bf0" stroke-width="3"/>'

    # Data points
    fb_dots = []
    for i, layer in enumerate(all_layers):
        if layer in fb_data:
            x = x_coord(i)
            y = y_coord(fb_data[layer])
            fb_dots.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#f4212e"/>')

    tb_dots = []
    for i, layer in enumerate(all_layers):
        if layer in tb_data:
            x = x_coord(i)
            y = y_coord(tb_data[layer])
            tb_dots.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#1d9bf0"/>')

    # X-axis (layers)
    x_axis = f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#2f3336" stroke-width="1"/>'
    x_labels = []
    step = max(1, num_layers // 10)
    for i, layer in enumerate(all_layers):
        if i % step == 0 or i == num_layers - 1:
            x = x_coord(i)
            x_labels.append(f'<text x="{x}" y="{height - margin_bottom + 20}" text-anchor="middle" fill="#8b98a5" font-size="11">L{layer}</text>')

    # Y-axis
    y_axis = f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#2f3336" stroke-width="1"/>'
    y_labels = []
    y_grid = []
    for val in [0.5, 1.0, 1.5, 2.0]:
        if y_min <= val <= y_max:
            y = y_coord(val)
            y_labels.append(f'<text x="{margin_left - 10}" y="{y + 4}" text-anchor="end" fill="#8b98a5" font-size="11">{val:.1f}</text>')
            y_grid.append(f'<line x1="{margin_left}" y1="{y}" x2="{width - margin_right}" y2="{y}" stroke="#2f3336" stroke-width="1" stroke-dasharray="4,4"/>')

    # Reference line at ratio = 1.0 (no preference)
    ref_y = y_coord(1.0)
    ref_line = f'<line x1="{margin_left}" y1="{ref_y}" x2="{width - margin_right}" y2="{ref_y}" stroke="#8b98a5" stroke-width="2" stroke-dasharray="6,3"/>'

    # Legend
    legend = f'''
        <rect x="{width - 180}" y="15" width="16" height="3" fill="#f4212e"/>
        <text x="{width - 160}" y="20" fill="#8b98a5" font-size="12">False Belief</text>
        <rect x="{width - 180}" y="35" width="16" height="3" fill="#1d9bf0"/>
        <text x="{width - 160}" y="40" fill="#8b98a5" font-size="12">True Belief</text>
    '''

    # Axis labels
    x_axis_label = f'<text x="{margin_left + plot_width / 2}" y="{height - 10}" text-anchor="middle" fill="#8b98a5" font-size="13">Layer Index</text>'
    y_axis_label = f'<text x="15" y="{margin_top + plot_height / 2}" text-anchor="middle" fill="#8b98a5" font-size="13" transform="rotate(-90, 15, {margin_top + plot_height / 2})">Belief/Reality Ratio</text>'

    svg = f'''
        <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            {x_axis}
            {y_axis}
            {''.join(y_grid)}
            {ref_line}
            {''.join(x_labels)}
            {''.join(y_labels)}
            {fb_path}
            {tb_path}
            {''.join(fb_dots)}
            {''.join(tb_dots)}
            {legend}
            {x_axis_label}
            {y_axis_label}
        </svg>
    '''

    content = f"""
        <h1>Selectivity Across Layers</h1>
        <p style="color: var(--text-secondary); margin-bottom: 2rem;">
            Mean belief/reality attention ratio by layer. Ratio > 1.0 means more attention to believed location.
            The dashed line marks the neutral ratio of 1.0.
        </p>

        <div class="heatmap-container">
            <div style="overflow-x: auto;">
                {svg}
            </div>
        </div>

        <div class="card" style="margin-top: 1.5rem;">
            <h3>Interpretation</h3>
            <ul style="color: var(--text-secondary); margin-left: 1.5rem;">
                <li><strong>False belief (red)</strong>: Expect higher ratios in later layers as model tracks believed location</li>
                <li><strong>True belief (blue)</strong>: Should stay closer to 1.0 since belief = reality</li>
                <li>Layers where red diverges from blue are likely performing ToM reasoning</li>
            </ul>
        </div>

        <p style="margin-top: 1.5rem;">
            <a href="/attention" style="color: var(--accent);">&larr; Back to Attention Heatmaps</a>
        </p>
    """

    return render_page("Layer Selectivity", content, "Attention")


def render_token_attention() -> str:
    """Render the token-level attention visualization page."""
    token_data = dashboard.get_token_attention()
    head_scores = dashboard.get_head_scores()

    if not token_data:
        return render_page("Token Attention", """
            <h1>Token-Level Attention</h1>
            <div class="empty-state">
                <p>No token attention data available.</p>
                <p>Re-run the study to generate this data.</p>
            </div>
        """, "Attention")

    # Group by head and pick one example per head
    examples_by_head = {}
    for item in token_data:
        key = (item["layer"], item["head"])
        if key not in examples_by_head:
            examples_by_head[key] = item

    # Get top head scores for ordering
    head_ranks = {}
    if head_scores:
        for i, score in enumerate(sorted(head_scores, key=lambda x: -x.get("tom_score", 0))):
            head_ranks[(score["layer"], score["head"])] = i

    # Sort examples by ToM rank
    sorted_examples = sorted(examples_by_head.items(), key=lambda x: head_ranks.get(x[0], 999))

    # Build visualizations for top 5 heads
    head_charts = []
    for (layer, head), data in sorted_examples[:5]:
        tokens = data["tokens"]
        weights = data["attention_weights"]
        regions = data.get("regions", {})
        task_type = data.get("task_type", "unknown")
        rank = head_ranks.get((layer, head), "?")

        # SVG bar chart
        chart_width = 700
        chart_height = 200
        margin_left = 50
        margin_bottom = 80
        margin_top = 30
        margin_right = 30
        plot_width = chart_width - margin_left - margin_right
        plot_height = chart_height - margin_top - margin_bottom

        num_tokens = len(tokens)
        bar_width = max(4, plot_width / num_tokens - 2)

        # Normalize bar height
        max_weight = max(weights) if weights else 1.0

        bars = []
        labels = []
        for i, (token, weight) in enumerate(zip(tokens, weights)):
            x = margin_left + i * (plot_width / num_tokens)
            bar_height = (weight / max_weight) * plot_height
            y = margin_top + plot_height - bar_height

            # Color by region
            color = "#4a5568"  # default gray
            if regions:
                for region_name, (start, end) in regions.items():
                    if start <= i < end:
                        if region_name == "protagonist":
                            color = "#1d9bf0"  # blue
                        elif region_name == "belief_location":
                            color = "#00ba7c"  # green
                        elif region_name == "reality_location":
                            color = "#f4212e"  # red
                        break

            bars.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{color}" rx="2"/>')

            # Token labels (rotated, every nth token)
            if num_tokens <= 30 or i % max(1, num_tokens // 30) == 0:
                display_token = token[:10] + "..." if len(token) > 10 else token
                display_token = html_module.escape(display_token)
                labels.append(f'<text x="{x + bar_width/2}" y="{chart_height - margin_bottom + 10}" text-anchor="start" fill="#8b98a5" font-size="9" transform="rotate(45, {x + bar_width/2}, {chart_height - margin_bottom + 10})">{display_token}</text>')

        # Y-axis
        y_axis = f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#2f3336" stroke-width="1"/>'
        y_labels = []
        for val in [0, 0.5, 1.0]:
            y = margin_top + plot_height - (val * plot_height)
            y_labels.append(f'<text x="{margin_left - 5}" y="{y + 4}" text-anchor="end" fill="#8b98a5" font-size="10">{val:.1f}</text>')

        # Title
        title_text = f'<text x="{chart_width / 2}" y="18" text-anchor="middle" fill="#e7e9ea" font-size="14" font-weight="600">Layer {layer}, Head {head} (Rank #{rank + 1}) - {task_type.replace("_", " ").title()}</text>'

        svg = f'''
            <svg width="{chart_width}" height="{chart_height}" xmlns="http://www.w3.org/2000/svg">
                {title_text}
                {y_axis}
                {''.join(y_labels)}
                {''.join(bars)}
                {''.join(labels)}
            </svg>
        '''

        head_charts.append(f'''
            <div class="card" style="margin-bottom: 1.5rem;">
                <div style="overflow-x: auto;">
                    {svg}
                </div>
            </div>
        ''')

    # Legend
    legend_html = '''
        <div class="legend" style="margin-bottom: 1.5rem;">
            <span style="display: inline-flex; align-items: center; margin-right: 1.5rem;">
                <span style="width: 16px; height: 16px; background: #1d9bf0; border-radius: 3px; margin-right: 0.5rem;"></span>
                Protagonist
            </span>
            <span style="display: inline-flex; align-items: center; margin-right: 1.5rem;">
                <span style="width: 16px; height: 16px; background: #00ba7c; border-radius: 3px; margin-right: 0.5rem;"></span>
                Belief Location
            </span>
            <span style="display: inline-flex; align-items: center; margin-right: 1.5rem;">
                <span style="width: 16px; height: 16px; background: #f4212e; border-radius: 3px; margin-right: 0.5rem;"></span>
                Reality Location
            </span>
            <span style="display: inline-flex; align-items: center;">
                <span style="width: 16px; height: 16px; background: #4a5568; border-radius: 3px; margin-right: 0.5rem;"></span>
                Other
            </span>
        </div>
    '''

    content = f"""
        <h1>Token-Level Attention</h1>
        <p style="color: var(--text-secondary); margin-bottom: 1rem;">
            Attention weights from question tokens to all other tokens for top ToM heads.
            Bar height indicates attention weight (normalized). Colors indicate token regions.
        </p>

        {legend_html}

        {''.join(head_charts)}

        <p style="margin-top: 1.5rem;">
            <a href="/attention" style="color: var(--accent);">&larr; Back to Attention Heatmaps</a>
        </p>
    """

    return render_page("Token Attention", content, "Attention")


def render_task_matrix() -> str:
    """Render per-task attention matrix visualization."""
    matrices_data = dashboard.get_attention_matrices()
    head_scores = dashboard.get_head_scores()

    if not matrices_data:
        return render_page("Task Matrix", """
            <h1>Per-Task Attention Matrix</h1>
            <div class="empty-state">
                <p>No attention matrix data available.</p>
                <p>Re-run the study to generate this data.</p>
            </div>
        """, "Attention")

    # Get head ranks
    head_ranks = {}
    if head_scores:
        for i, score in enumerate(sorted(head_scores, key=lambda x: -x.get("tom_score", 0))):
            head_ranks[(score["layer"], score["head"])] = i

    # Pick first available matrix
    matrix_data = matrices_data[0]
    tokens = matrix_data["tokens"]
    attn_matrix = matrix_data["attention_matrix"]
    regions = matrix_data.get("regions", {})
    layer = matrix_data["layer"]
    head = matrix_data["head"]
    task_id = matrix_data["task_id"]
    task_type = matrix_data["task_type"]
    rank = head_ranks.get((layer, head), "?")

    # Limit to last 40 tokens for readability
    max_tokens = 40
    if len(tokens) > max_tokens:
        tokens = tokens[-max_tokens:]
        attn_matrix = [row[-max_tokens:] for row in attn_matrix[-max_tokens:]]
        # Adjust region indices
        offset = len(matrix_data["tokens"]) - max_tokens
        regions = {
            name: (max(0, start - offset), max(0, end - offset))
            for name, (start, end) in regions.items()
        }

    num_tokens = len(tokens)
    cell_size = 16
    margin_left = 120
    margin_top = 120
    margin_right = 20
    margin_bottom = 20

    width = margin_left + num_tokens * cell_size + margin_right
    height = margin_top + num_tokens * cell_size + margin_bottom

    # Build cells
    cells_svg = []
    for i, row in enumerate(attn_matrix):
        for j, value in enumerate(row):
            x = margin_left + j * cell_size
            y = margin_top + i * cell_size
            # Color intensity
            intensity = min(1.0, value * 5)  # Scale up for visibility
            color = f"rgba(29, 155, 240, {intensity:.2f})"
            cells_svg.append(
                f'<rect x="{x}" y="{y}" width="{cell_size - 1}" height="{cell_size - 1}" '
                f'fill="{color}" rx="2"/>'
            )

    # Token labels (query - y-axis)
    y_labels = []
    for i, token in enumerate(tokens):
        y = margin_top + i * cell_size + cell_size / 2 + 4
        display = token[:12].replace("<", "&lt;").replace(">", "&gt;")
        # Color by region
        color = "#8b98a5"
        for region_name, (start, end) in regions.items():
            if start <= i < end:
                if region_name == "protagonist":
                    color = "#1d9bf0"
                elif region_name == "belief_location":
                    color = "#00ba7c"
                elif region_name == "reality_location":
                    color = "#f4212e"
                break
        y_labels.append(f'<text x="{margin_left - 5}" y="{y}" text-anchor="end" fill="{color}" font-size="9">{display}</text>')

    # Token labels (key - x-axis, rotated)
    x_labels = []
    for i, token in enumerate(tokens):
        x = margin_left + i * cell_size + cell_size / 2
        display = token[:12].replace("<", "&lt;").replace(">", "&gt;")
        color = "#8b98a5"
        for region_name, (start, end) in regions.items():
            if start <= i < end:
                if region_name == "protagonist":
                    color = "#1d9bf0"
                elif region_name == "belief_location":
                    color = "#00ba7c"
                elif region_name == "reality_location":
                    color = "#f4212e"
                break
        x_labels.append(f'<text x="{x}" y="{margin_top - 5}" text-anchor="start" fill="{color}" font-size="9" transform="rotate(-45, {x}, {margin_top - 5})">{display}</text>')

    # Title
    title = f'<text x="{width / 2}" y="25" text-anchor="middle" fill="#e7e9ea" font-size="14">L{layer}H{head} (Rank #{rank + 1 if isinstance(rank, int) else rank}) - {task_type.replace("_", " ").title()}</text>'

    svg = f'''
        <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            {title}
            {''.join(x_labels)}
            {''.join(y_labels)}
            {''.join(cells_svg)}
        </svg>
    '''

    # Task selector (if multiple tasks)
    task_selector = ""
    if len(matrices_data) > 1:
        options = []
        for m in matrices_data[:10]:  # Limit to 10
            options.append(f'<option value="{m["task_id"]}">{m["task_id"]} ({m["task_type"]})</option>')
        task_selector = f'''
            <div style="margin-bottom: 1rem; color: var(--text-secondary);">
                Showing task: <strong>{task_id}</strong> (first of {len(matrices_data)} available)
            </div>
        '''

    content = f"""
        <h1>Per-Task Attention Matrix</h1>
        <p style="color: var(--text-secondary); margin-bottom: 1rem;">
            Full attention matrix showing query→key attention for top ToM head.
            Brighter cells indicate higher attention weight.
        </p>

        {task_selector}

        <div class="heatmap-container">
            <div style="overflow-x: auto;">
                {svg}
            </div>
            <div class="legend" style="margin-top: 1rem;">
                <span style="display: inline-flex; align-items: center; margin-right: 1.5rem;">
                    <span style="width: 16px; height: 16px; background: #1d9bf0; border-radius: 3px; margin-right: 0.5rem;"></span>
                    Protagonist
                </span>
                <span style="display: inline-flex; align-items: center; margin-right: 1.5rem;">
                    <span style="width: 16px; height: 16px; background: #00ba7c; border-radius: 3px; margin-right: 0.5rem;"></span>
                    Belief Location
                </span>
                <span style="display: inline-flex; align-items: center;">
                    <span style="width: 16px; height: 16px; background: #f4212e; border-radius: 3px; margin-right: 0.5rem;"></span>
                    Reality Location
                </span>
            </div>
        </div>

        <p style="margin-top: 1.5rem;">
            <a href="/attention" style="color: var(--accent);">&larr; Back to Attention Heatmaps</a>
        </p>
    """

    return render_page("Task Matrix", content, "Attention")


def render_attention_flow() -> str:
    """Render attention flow diagram (BertViz-style)."""
    matrices_data = dashboard.get_attention_matrices()
    head_scores = dashboard.get_head_scores()

    if not matrices_data:
        return render_page("Attention Flow", """
            <h1>Attention Flow Diagram</h1>
            <div class="empty-state">
                <p>No attention matrix data available.</p>
                <p>Re-run the study to generate this data.</p>
            </div>
        """, "Attention")

    # Get first matrix
    matrix_data = matrices_data[0]
    tokens = matrix_data["tokens"]
    attn_matrix = matrix_data["attention_matrix"]
    regions = matrix_data.get("regions", {})
    layer = matrix_data["layer"]
    head = matrix_data["head"]
    task_type = matrix_data["task_type"]

    # Get head rank
    head_ranks = {}
    if head_scores:
        for i, score in enumerate(sorted(head_scores, key=lambda x: -x.get("tom_score", 0))):
            head_ranks[(score["layer"], score["head"])] = i
    rank = head_ranks.get((layer, head), "?")

    # Limit tokens for readability
    max_tokens = 30
    if len(tokens) > max_tokens:
        # Take last N tokens (question region is at the end)
        offset = len(tokens) - max_tokens
        tokens = tokens[-max_tokens:]
        attn_matrix = [row[-max_tokens:] for row in attn_matrix[-max_tokens:]]
        regions = {
            name: (max(0, start - offset), max(0, end - offset))
            for name, (start, end) in regions.items()
        }

    num_tokens = len(tokens)

    # SVG dimensions
    width = 800
    token_height = 20
    margin_left = 200
    margin_right = 200
    margin_top = 60
    margin_bottom = 40
    flow_width = width - margin_left - margin_right
    height = margin_top + num_tokens * token_height + margin_bottom

    # Get color for token based on region
    def get_token_color(idx):
        for region_name, (start, end) in regions.items():
            if start <= idx < end:
                if region_name == "protagonist":
                    return "#1d9bf0"
                elif region_name == "belief_location":
                    return "#00ba7c"
                elif region_name == "reality_location":
                    return "#f4212e"
                elif region_name == "question":
                    return "#ffad1f"
        return "#8b98a5"

    # Left column: source tokens (query)
    left_labels = []
    for i, token in enumerate(tokens):
        y = margin_top + i * token_height + token_height / 2
        display = html_module.escape(token[:15])
        color = get_token_color(i)
        left_labels.append(f'<text x="{margin_left - 10}" y="{y + 4}" text-anchor="end" fill="{color}" font-size="11">{display}</text>')

    # Right column: target tokens (key)
    right_labels = []
    for i, token in enumerate(tokens):
        y = margin_top + i * token_height + token_height / 2
        display = html_module.escape(token[:15])
        color = get_token_color(i)
        right_labels.append(f'<text x="{width - margin_right + 10}" y="{y + 4}" text-anchor="start" fill="{color}" font-size="11">{display}</text>')

    # Draw attention lines
    # Only show lines from question tokens to all other tokens
    question_region = regions.get("question", (0, num_tokens))
    lines = []

    for i in range(question_region[0], min(question_region[1], num_tokens)):
        if i >= len(attn_matrix):
            continue
        row = attn_matrix[i]
        source_y = margin_top + i * token_height + token_height / 2

        for j, weight in enumerate(row):
            if j >= num_tokens:
                continue
            # Only draw significant attention
            if weight < 0.02:
                continue

            target_y = margin_top + j * token_height + token_height / 2

            # Line properties
            opacity = min(1.0, weight * 3)
            stroke_width = max(1, min(4, weight * 10))

            # Color based on target region
            color = get_token_color(j)

            lines.append(
                f'<line x1="{margin_left}" y1="{source_y}" x2="{width - margin_right}" y2="{target_y}" '
                f'stroke="{color}" stroke-width="{stroke_width}" opacity="{opacity:.2f}"/>'
            )

    # Title
    title = f'<text x="{width / 2}" y="25" text-anchor="middle" fill="#e7e9ea" font-size="14" font-weight="600">L{layer}H{head} (Rank #{rank + 1 if isinstance(rank, int) else rank}) - {task_type.replace("_", " ").title()}</text>'
    subtitle = f'<text x="{width / 2}" y="45" text-anchor="middle" fill="#8b98a5" font-size="12">Question tokens → All tokens</text>'

    # Column headers
    left_header = f'<text x="{margin_left - 10}" y="{margin_top - 15}" text-anchor="end" fill="#8b98a5" font-size="11" font-weight="600">Source (Query)</text>'
    right_header = f'<text x="{width - margin_right + 10}" y="{margin_top - 15}" text-anchor="start" fill="#8b98a5" font-size="11" font-weight="600">Target (Key)</text>'

    svg = f'''
        <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            {title}
            {subtitle}
            {left_header}
            {right_header}
            {''.join(lines)}
            {''.join(left_labels)}
            {''.join(right_labels)}
        </svg>
    '''

    # Legend
    legend_html = '''
        <div class="legend" style="margin-bottom: 1.5rem;">
            <span style="display: inline-flex; align-items: center; margin-right: 1.5rem;">
                <span style="width: 16px; height: 3px; background: #1d9bf0; margin-right: 0.5rem;"></span>
                Protagonist
            </span>
            <span style="display: inline-flex; align-items: center; margin-right: 1.5rem;">
                <span style="width: 16px; height: 3px; background: #00ba7c; margin-right: 0.5rem;"></span>
                Belief Location
            </span>
            <span style="display: inline-flex; align-items: center; margin-right: 1.5rem;">
                <span style="width: 16px; height: 3px; background: #f4212e; margin-right: 0.5rem;"></span>
                Reality Location
            </span>
            <span style="display: inline-flex; align-items: center;">
                <span style="width: 16px; height: 3px; background: #ffad1f; margin-right: 0.5rem;"></span>
                Question
            </span>
        </div>
    '''

    content = f"""
        <h1>Attention Flow Diagram</h1>
        <p style="color: var(--text-secondary); margin-bottom: 1rem;">
            Visualizes where question tokens attend to in the sequence.
            Line thickness and opacity indicate attention weight.
        </p>

        {legend_html}

        <div class="heatmap-container">
            <div style="overflow-x: auto;">
                {svg}
            </div>
        </div>

        <div class="card" style="margin-top: 1.5rem;">
            <h3>Interpretation</h3>
            <p style="color: var(--text-secondary);">
                In Theory of Mind tasks, we expect the model to attend from question tokens (asking where the protagonist will look)
                to the <strong style="color: #00ba7c;">belief location</strong> tokens in false-belief conditions,
                rather than the <strong style="color: #f4212e;">reality location</strong>.
            </p>
        </div>

        <p style="margin-top: 1.5rem;">
            <a href="/attention" style="color: var(--accent);">&larr; Back to Attention Heatmaps</a>
        </p>
    """

    return render_page("Attention Flow", content, "Attention")


def render_head_clustering() -> str:
    """Render head clustering visualization."""
    head_scores = dashboard.get_head_scores()

    if not head_scores:
        return render_page("Head Clustering", """
            <h1>Head Clustering</h1>
            <div class="empty-state">
                <p>No head scores available.</p>
                <p>Run the study with attention analysis enabled.</p>
            </div>
        """, "Attention")

    # Simple 2D scatter plot based on FB ratio vs condition difference
    # In a full implementation, we'd use t-SNE/UMAP on attention patterns

    width = 600
    height = 500
    margin_left = 70
    margin_right = 30
    margin_top = 40
    margin_bottom = 60
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    # Get data ranges
    fb_ratios = [s.get("false_belief_ratio", 1) for s in head_scores]
    diffs = [s.get("condition_diff", 0) for s in head_scores]
    tom_scores = [s.get("tom_score", 0) for s in head_scores]

    x_min, x_max = min(fb_ratios) - 0.1, max(fb_ratios) + 0.1
    y_min, y_max = min(diffs) - 0.1, max(diffs) + 0.1

    def x_coord(val):
        return margin_left + ((val - x_min) / (x_max - x_min)) * plot_width

    def y_coord(val):
        return margin_top + plot_height - ((val - y_min) / (y_max - y_min)) * plot_height

    # Get unique layers for coloring
    layers = sorted(set(s.get("layer", 0) for s in head_scores))
    num_layers = len(layers)

    def layer_color(layer):
        if num_layers <= 1:
            return "#1d9bf0"
        idx = layers.index(layer)
        # Blue to red gradient
        t = idx / (num_layers - 1)
        r = int(30 + t * (220 - 30))
        g = int(100 + t * (50 - 100))
        b = int(200 - t * 170)
        return f"rgb({r},{g},{b})"

    # Points
    points = []
    max_tom = max(tom_scores) if tom_scores else 1
    for score in head_scores:
        x = x_coord(score.get("false_belief_ratio", 1))
        y = y_coord(score.get("condition_diff", 0))
        layer = score.get("layer", 0)
        head = score.get("head", 0)
        tom = score.get("tom_score", 0)

        # Size based on ToM score
        radius = 4 + (tom / max_tom) * 8 if max_tom > 0 else 4
        color = layer_color(layer)

        points.append(
            f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}" opacity="0.7" '
            f'data-layer="{layer}" data-head="{head}"/>'
        )
        # Labels for top heads
        if tom > max_tom * 0.5:
            points.append(
                f'<text x="{x + radius + 3}" y="{y + 4}" fill="#e7e9ea" font-size="9">L{layer}H{head}</text>'
            )

    # Axes
    x_axis = f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#2f3336" stroke-width="1"/>'
    y_axis = f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#2f3336" stroke-width="1"/>'

    # Axis labels
    x_label = f'<text x="{margin_left + plot_width / 2}" y="{height - 15}" text-anchor="middle" fill="#8b98a5" font-size="12">False Belief Ratio</text>'
    y_label = f'<text x="20" y="{margin_top + plot_height / 2}" text-anchor="middle" fill="#8b98a5" font-size="12" transform="rotate(-90, 20, {margin_top + plot_height / 2})">Condition Difference (FB - TB)</text>'

    # Reference lines at ratio=1 and diff=0
    ref_x = x_coord(1.0)
    ref_y = y_coord(0.0)
    ref_lines = f'''
        <line x1="{ref_x}" y1="{margin_top}" x2="{ref_x}" y2="{margin_top + plot_height}" stroke="#8b98a5" stroke-width="1" stroke-dasharray="4,4"/>
        <line x1="{margin_left}" y1="{ref_y}" x2="{width - margin_right}" y2="{ref_y}" stroke="#8b98a5" stroke-width="1" stroke-dasharray="4,4"/>
    '''

    # Tick marks
    x_ticks = []
    for val in [0.5, 1.0, 1.5, 2.0, 2.5]:
        if x_min <= val <= x_max:
            x = x_coord(val)
            x_ticks.append(f'<text x="{x}" y="{margin_top + plot_height + 15}" text-anchor="middle" fill="#8b98a5" font-size="10">{val:.1f}</text>')

    y_ticks = []
    for val in [-0.5, 0, 0.5, 1.0, 1.5]:
        if y_min <= val <= y_max:
            y = y_coord(val)
            y_ticks.append(f'<text x="{margin_left - 5}" y="{y + 4}" text-anchor="end" fill="#8b98a5" font-size="10">{val:.1f}</text>')

    # Layer legend
    legend_items = []
    legend_y = margin_top
    for i, layer in enumerate(layers[:8]):  # Show first 8 layers
        color = layer_color(layer)
        legend_items.append(f'<circle cx="{width - 60}" cy="{legend_y + i * 18}" r="5" fill="{color}"/>')
        legend_items.append(f'<text x="{width - 50}" y="{legend_y + i * 18 + 4}" fill="#8b98a5" font-size="10">L{layer}</text>')

    svg = f'''
        <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            {x_axis}
            {y_axis}
            {ref_lines}
            {''.join(x_ticks)}
            {''.join(y_ticks)}
            {''.join(points)}
            {''.join(legend_items)}
            {x_label}
            {y_label}
        </svg>
    '''

    content = f"""
        <h1>Head Clustering</h1>
        <p style="color: var(--text-secondary); margin-bottom: 2rem;">
            Scatter plot of attention heads based on their belief-tracking behavior.
            Point size indicates ToM score. Color indicates layer (blue=early, red=late).
        </p>

        <div class="heatmap-container">
            <div style="overflow-x: auto;">
                {svg}
            </div>
        </div>

        <div class="card" style="margin-top: 1.5rem;">
            <h3>Quadrant Interpretation</h3>
            <table>
                <tr>
                    <th>Quadrant</th>
                    <th>Characteristics</th>
                    <th>Interpretation</th>
                </tr>
                <tr>
                    <td><strong>Upper Right</strong></td>
                    <td>High FB ratio, High diff</td>
                    <td style="color: #00ba7c;">ToM-relevant heads</td>
                </tr>
                <tr>
                    <td><strong>Upper Left</strong></td>
                    <td>Low FB ratio, High diff</td>
                    <td>Reality-tracking heads</td>
                </tr>
                <tr>
                    <td><strong>Lower Right</strong></td>
                    <td>High FB ratio, Low diff</td>
                    <td>Consistent belief-tracking (both conditions)</td>
                </tr>
                <tr>
                    <td><strong>Lower Left</strong></td>
                    <td>Low FB ratio, Low diff</td>
                    <td>General attention heads</td>
                </tr>
            </table>
        </div>

        <p style="margin-top: 1.5rem;">
            <a href="/attention" style="color: var(--accent);">&larr; Back to Attention Heatmaps</a>
        </p>
    """

    return render_page("Head Clustering", content, "Attention")


def render_heads() -> str:
    """Render the ToM heads page."""
    head_scores = dashboard.get_head_scores()

    if not head_scores:
        return render_page("ToM Heads", """
            <h1>ToM Head Identification</h1>
            <div class="empty-state">
                <p>No head scores available.</p>
                <p>Run the study with attention analysis enabled.</p>
            </div>
        """, "ToM Heads")

    # Sort by ToM score
    head_scores.sort(key=lambda x: -x.get("tom_score", 0))

    # Build head comparison bar chart for top 10
    top_10 = head_scores[:10]
    chart_width = 800
    chart_height = 300
    margin_left = 80
    margin_right = 30
    margin_top = 40
    margin_bottom = 80
    plot_width = chart_width - margin_left - margin_right
    plot_height = chart_height - margin_top - margin_bottom

    num_heads = len(top_10)
    group_width = plot_width / num_heads
    bar_width = group_width / 4 - 2

    # Find max values for scaling
    max_fb_ratio = max(s.get("false_belief_ratio", 0) for s in top_10) if top_10 else 1
    max_tb_ratio = max(s.get("true_belief_ratio", 0) for s in top_10) if top_10 else 1
    max_ratio = max(max_fb_ratio, max_tb_ratio, 1.5)

    bars_svg = []
    labels_svg = []

    for i, score in enumerate(top_10):
        x_center = margin_left + i * group_width + group_width / 2
        fb_ratio = score.get("false_belief_ratio", 0)
        tb_ratio = score.get("true_belief_ratio", 0)
        diff = score.get("condition_diff", 0)
        tom = score.get("tom_score", 0)

        # FB Ratio bar (red)
        fb_height = (fb_ratio / max_ratio) * plot_height
        fb_y = margin_top + plot_height - fb_height
        bars_svg.append(f'<rect x="{x_center - bar_width * 1.5}" y="{fb_y}" width="{bar_width}" height="{fb_height}" fill="#f4212e" rx="2"/>')

        # TB Ratio bar (blue)
        tb_height = (tb_ratio / max_ratio) * plot_height
        tb_y = margin_top + plot_height - tb_height
        bars_svg.append(f'<rect x="{x_center - bar_width * 0.5}" y="{tb_y}" width="{bar_width}" height="{tb_height}" fill="#1d9bf0" rx="2"/>')

        # Condition diff bar (orange, scaled differently)
        diff_height = min(abs(diff) / max_ratio * plot_height, plot_height)
        diff_y = margin_top + plot_height - diff_height
        bars_svg.append(f'<rect x="{x_center + bar_width * 0.5}" y="{diff_y}" width="{bar_width}" height="{diff_height}" fill="#ffad1f" rx="2"/>')

        # Head label
        layer = score.get("layer", "?")
        head = score.get("head", "?")
        labels_svg.append(f'<text x="{x_center}" y="{chart_height - margin_bottom + 20}" text-anchor="middle" fill="#8b98a5" font-size="10">L{layer}H{head}</text>')
        labels_svg.append(f'<text x="{x_center}" y="{chart_height - margin_bottom + 35}" text-anchor="middle" fill="#8b98a5" font-size="9">#{i + 1}</text>')

    # Reference line at ratio = 1.0
    ref_y = margin_top + plot_height - (1.0 / max_ratio) * plot_height
    ref_line = f'<line x1="{margin_left}" y1="{ref_y}" x2="{chart_width - margin_right}" y2="{ref_y}" stroke="#8b98a5" stroke-width="1" stroke-dasharray="4,4"/>'
    ref_label = f'<text x="{margin_left - 5}" y="{ref_y + 4}" text-anchor="end" fill="#8b98a5" font-size="10">1.0</text>'

    # Y-axis
    y_axis = f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#2f3336" stroke-width="1"/>'

    # Legend
    legend_svg = f'''
        <rect x="{chart_width - 200}" y="10" width="12" height="12" fill="#f4212e" rx="2"/>
        <text x="{chart_width - 185}" y="20" fill="#8b98a5" font-size="11">FB Ratio</text>
        <rect x="{chart_width - 130}" y="10" width="12" height="12" fill="#1d9bf0" rx="2"/>
        <text x="{chart_width - 115}" y="20" fill="#8b98a5" font-size="11">TB Ratio</text>
        <rect x="{chart_width - 60}" y="10" width="12" height="12" fill="#ffad1f" rx="2"/>
        <text x="{chart_width - 45}" y="20" fill="#8b98a5" font-size="11">Diff</text>
    '''

    comparison_svg = f'''
        <svg width="{chart_width}" height="{chart_height}" xmlns="http://www.w3.org/2000/svg">
            {y_axis}
            {ref_line}
            {ref_label}
            {''.join(bars_svg)}
            {''.join(labels_svg)}
            {legend_svg}
        </svg>
    '''

    rows = []
    for i, score in enumerate(head_scores[:20]):  # Top 20
        tom_score = score.get("tom_score", 0)
        badge_class = "success" if tom_score > 0.1 else "info" if tom_score > 0 else ""

        rows.append(f"""
            <tr>
                <td>{i + 1}</td>
                <td>Layer {score.get('layer', 'N/A')}</td>
                <td>Head {score.get('head', 'N/A')}</td>
                <td>{score.get('false_belief_ratio', 0):.2f}</td>
                <td>{score.get('true_belief_ratio', 0):.2f}</td>
                <td>{score.get('condition_diff', 0):.2f}</td>
                <td><span class="badge {badge_class}">{tom_score:.3f}</span></td>
            </tr>
        """)

    content = f"""
        <h1>ToM Head Identification</h1>
        <p style="color: var(--text-secondary); margin-bottom: 2rem;">
            Heads ranked by Theory of Mind relevance score.
            ToM Score = (FB Ratio - 1) &times; Condition Difference.
            Higher scores indicate heads that attend more to belief in false-belief tasks.
        </p>

        <h2>Top 10 Head Comparison</h2>
        <div class="card" style="margin-bottom: 2rem;">
            <div style="overflow-x: auto;">
                {comparison_svg}
            </div>
            <p style="color: var(--text-secondary); margin-top: 1rem; font-size: 0.875rem;">
                Dashed line indicates neutral ratio (1.0). Higher FB Ratio with larger FB-TB difference indicates stronger ToM behavior.
            </p>
        </div>

        <h2>Full Rankings</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Layer</th>
                    <th>Head</th>
                    <th>FB Ratio</th>
                    <th>TB Ratio</th>
                    <th>Difference</th>
                    <th>ToM Score</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    """

    return render_page("ToM Heads", content, "ToM Heads")


def render_interventions() -> str:
    """Render the interventions page."""
    ablation = dashboard.get_intervention_results("ablation")
    boost = dashboard.get_intervention_results("boost")
    multihead_results = dashboard.get_multihead_intervention_results()

    if not ablation and not boost and not multihead_results:
        return render_page("Interventions", """
            <h1>Intervention Experiments</h1>
            <div class="empty-state">
                <p>No intervention results available.</p>
                <p>Run the study with interventions enabled.</p>
            </div>
        """, "Interventions")

    def render_summary(data: Dict, title: str) -> str:
        summary = data.get("summary", {})
        orig_acc = summary.get("original_accuracy", 0) * 100
        mod_acc = summary.get("modified_accuracy", 0) * 100
        delta = summary.get("accuracy_delta", 0) * 100
        flip_rate = summary.get("flip_rate", 0) * 100

        delta_class = "delta-negative" if delta < 0 else "delta-positive"
        target_heads = summary.get("target_heads", [])
        heads_str = ", ".join([f"L{l}H{h}" for l, h in target_heads[:5]])
        if len(target_heads) > 5:
            heads_str += f" (+{len(target_heads) - 5} more)"

        return f"""
            <div class="card">
                <h3>{title}</h3>
                <table>
                    <tr><th>Target Heads</th><td>{heads_str}</td></tr>
                    <tr><th>Scale Factor</th><td>{summary.get('scale_factor', 'N/A')}</td></tr>
                    <tr><th>Original Accuracy</th><td>{orig_acc:.1f}%</td></tr>
                    <tr><th>Modified Accuracy</th><td>{mod_acc:.1f}%</td></tr>
                    <tr><th>Delta</th><td class="{delta_class}">{delta:+.1f}%</td></tr>
                    <tr><th>Flip Rate</th><td>{flip_rate:.1f}%</td></tr>
                </table>
                <h4 style="margin-top: 1rem;">By Task Type</h4>
                <table>
                    <tr>
                        <th></th>
                        <th>Original</th>
                        <th>Modified</th>
                    </tr>
                    <tr>
                        <td>False Belief</td>
                        <td>{summary.get('false_belief_original_acc', 0) * 100:.1f}%</td>
                        <td>{summary.get('false_belief_modified_acc', 0) * 100:.1f}%</td>
                    </tr>
                    <tr>
                        <td>True Belief</td>
                        <td>{summary.get('true_belief_original_acc', 0) * 100:.1f}%</td>
                        <td>{summary.get('true_belief_modified_acc', 0) * 100:.1f}%</td>
                    </tr>
                </table>
            </div>
        """

    cards = []
    if ablation:
        cards.append(render_summary(ablation, "Ablation (Scale = 0)"))
    if boost:
        cards.append(render_summary(boost, f"Boost (Scale = {boost.get('summary', {}).get('scale_factor', 2.0)})"))

    # Task-type breakdown chart
    task_breakdown_chart = ""
    if ablation or boost:
        chart_width = 500
        chart_height = 250
        margin_left = 100
        margin_right = 30
        margin_top = 30
        margin_bottom = 60
        plot_width = chart_width - margin_left - margin_right
        plot_height = chart_height - margin_top - margin_bottom

        bar_groups = []
        if ablation:
            abl_summary = ablation.get("summary", {})
            bar_groups.append({
                "label": "Ablation",
                "fb_orig": abl_summary.get("false_belief_original_acc", 0) * 100,
                "fb_mod": abl_summary.get("false_belief_modified_acc", 0) * 100,
                "tb_orig": abl_summary.get("true_belief_original_acc", 0) * 100,
                "tb_mod": abl_summary.get("true_belief_modified_acc", 0) * 100,
            })
        if boost:
            boost_summary = boost.get("summary", {})
            bar_groups.append({
                "label": "Boost",
                "fb_orig": boost_summary.get("false_belief_original_acc", 0) * 100,
                "fb_mod": boost_summary.get("false_belief_modified_acc", 0) * 100,
                "tb_orig": boost_summary.get("true_belief_original_acc", 0) * 100,
                "tb_mod": boost_summary.get("true_belief_modified_acc", 0) * 100,
            })

        num_groups = len(bar_groups)
        group_width = plot_width / (num_groups * 2 + 1)
        bar_width = group_width * 0.35

        bars_svg = []
        labels_svg = []

        for g_idx, group in enumerate(bar_groups):
            # False belief group
            fb_x = margin_left + (g_idx * 2) * group_width + group_width
            fb_orig_h = (group["fb_orig"] / 100) * plot_height
            fb_mod_h = (group["fb_mod"] / 100) * plot_height

            bars_svg.append(f'<rect x="{fb_x}" y="{margin_top + plot_height - fb_orig_h}" width="{bar_width}" height="{fb_orig_h}" fill="#8b4555" rx="2"/>')
            bars_svg.append(f'<rect x="{fb_x + bar_width + 2}" y="{margin_top + plot_height - fb_mod_h}" width="{bar_width}" height="{fb_mod_h}" fill="#f4212e" rx="2"/>')

            fb_delta = group["fb_mod"] - group["fb_orig"]
            delta_color = "#00ba7c" if fb_delta > 0 else "#f4212e"
            bars_svg.append(f'<text x="{fb_x + bar_width}" y="{margin_top - 5}" text-anchor="middle" fill="{delta_color}" font-size="10">{fb_delta:+.1f}%</text>')

            labels_svg.append(f'<text x="{fb_x + bar_width}" y="{chart_height - margin_bottom + 15}" text-anchor="middle" fill="#8b98a5" font-size="10">FB</text>')
            labels_svg.append(f'<text x="{fb_x + bar_width}" y="{chart_height - margin_bottom + 30}" text-anchor="middle" fill="#8b98a5" font-size="9">{group["label"]}</text>')

            # True belief group
            tb_x = margin_left + (g_idx * 2 + 1) * group_width + group_width
            tb_orig_h = (group["tb_orig"] / 100) * plot_height
            tb_mod_h = (group["tb_mod"] / 100) * plot_height

            bars_svg.append(f'<rect x="{tb_x}" y="{margin_top + plot_height - tb_orig_h}" width="{bar_width}" height="{tb_orig_h}" fill="#2a5a7a" rx="2"/>')
            bars_svg.append(f'<rect x="{tb_x + bar_width + 2}" y="{margin_top + plot_height - tb_mod_h}" width="{bar_width}" height="{tb_mod_h}" fill="#1d9bf0" rx="2"/>')

            tb_delta = group["tb_mod"] - group["tb_orig"]
            delta_color = "#00ba7c" if tb_delta > 0 else "#f4212e"
            bars_svg.append(f'<text x="{tb_x + bar_width}" y="{margin_top - 5}" text-anchor="middle" fill="{delta_color}" font-size="10">{tb_delta:+.1f}%</text>')

            labels_svg.append(f'<text x="{tb_x + bar_width}" y="{chart_height - margin_bottom + 15}" text-anchor="middle" fill="#8b98a5" font-size="10">TB</text>')
            labels_svg.append(f'<text x="{tb_x + bar_width}" y="{chart_height - margin_bottom + 30}" text-anchor="middle" fill="#8b98a5" font-size="9">{group["label"]}</text>')

        # Y-axis
        y_axis = f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#2f3336" stroke-width="1"/>'
        y_labels = []
        for val in [0, 25, 50, 75, 100]:
            y = margin_top + plot_height - (val / 100) * plot_height
            y_labels.append(f'<text x="{margin_left - 5}" y="{y + 4}" text-anchor="end" fill="#8b98a5" font-size="10">{val}%</text>')

        # Legend
        legend_svg = f'''
            <rect x="{chart_width - 130}" y="8" width="10" height="10" fill="#8b4555" rx="1"/>
            <rect x="{chart_width - 115}" y="8" width="10" height="10" fill="#f4212e" rx="1"/>
            <text x="{chart_width - 100}" y="17" fill="#8b98a5" font-size="10">FB</text>
            <rect x="{chart_width - 75}" y="8" width="10" height="10" fill="#2a5a7a" rx="1"/>
            <rect x="{chart_width - 60}" y="8" width="10" height="10" fill="#1d9bf0" rx="1"/>
            <text x="{chart_width - 45}" y="17" fill="#8b98a5" font-size="10">TB</text>
            <text x="{chart_width - 130}" y="32" fill="#8b98a5" font-size="9">Dark=Orig, Bright=Mod</text>
        '''

        breakdown_svg = f'''
            <svg width="{chart_width}" height="{chart_height}" xmlns="http://www.w3.org/2000/svg">
                {y_axis}
                {''.join(y_labels)}
                {''.join(bars_svg)}
                {''.join(labels_svg)}
                {legend_svg}
            </svg>
        '''

        task_breakdown_chart = f'''
            <h2>Task Type Breakdown</h2>
            <div class="card" style="margin-bottom: 1.5rem;">
                <h3>Accuracy by Task Type (Before/After)</h3>
                <div style="overflow-x: auto; padding: 1rem 0;">
                    {breakdown_svg}
                </div>
                <p style="color: var(--text-secondary); font-size: 0.875rem;">
                    Shows how interventions affect False Belief (FB) vs True Belief (TB) tasks differently.
                    Delta shown above each pair.
                </p>
            </div>
        '''

    # Render multi-head analysis chart
    multihead_chart = ""
    if multihead_results:
        sorted_heads = sorted(multihead_results.keys())

        # Build data for the chart
        chart_data = []
        for num_heads in sorted_heads:
            data = multihead_results[num_heads]
            abl_data = data.get("ablation", {}).get("summary", {})
            boost_data = data.get("boost", {}).get("summary", {})
            chart_data.append({
                "heads": num_heads,
                "ablation_delta": abl_data.get("accuracy_delta", 0) * 100,
                "boost_delta": boost_data.get("accuracy_delta", 0) * 100,
                "baseline": abl_data.get("original_accuracy", 0) * 100,
            })

        # Find optimal (most negative ablation, most positive boost)
        best_ablation = min(chart_data, key=lambda x: x["ablation_delta"])
        best_boost = max(chart_data, key=lambda x: x["boost_delta"])

        # Create SVG bar chart
        chart_width = 600
        chart_height = 300
        bar_width = 40
        group_gap = 60
        margin_left = 60
        margin_bottom = 40
        margin_top = 30

        # Scale for y-axis (delta values typically -10% to +5%)
        y_min = -8
        y_max = 4
        y_range = y_max - y_min
        y_scale = (chart_height - margin_top - margin_bottom) / y_range
        zero_y = margin_top + (y_max * y_scale)

        bars_svg = []
        labels_svg = []

        for i, data in enumerate(chart_data):
            x_center = margin_left + i * (bar_width * 2 + group_gap) + bar_width

            # Ablation bar (red)
            abl_height = abs(data["ablation_delta"]) * y_scale
            abl_y = zero_y if data["ablation_delta"] <= 0 else zero_y - abl_height
            abl_color = "#f4212e" if data["ablation_delta"] < 0 else "#00ba7c"
            is_best_abl = data["heads"] == best_ablation["heads"]
            abl_stroke = "stroke='#fff' stroke-width='2'" if is_best_abl else ""
            bars_svg.append(
                f'<rect x="{x_center - bar_width}" y="{abl_y}" width="{bar_width - 4}" '
                f'height="{abl_height}" fill="{abl_color}" rx="4" {abl_stroke}/>'
            )
            bars_svg.append(
                f'<text x="{x_center - bar_width/2}" y="{abl_y - 5}" text-anchor="middle" '
                f'fill="{abl_color}" font-size="11">{data["ablation_delta"]:+.1f}%</text>'
            )

            # Boost bar (green)
            boost_height = abs(data["boost_delta"]) * y_scale
            boost_y = zero_y - boost_height if data["boost_delta"] >= 0 else zero_y
            boost_color = "#00ba7c" if data["boost_delta"] > 0 else "#f4212e"
            is_best_boost = data["heads"] == best_boost["heads"]
            boost_stroke = "stroke='#fff' stroke-width='2'" if is_best_boost else ""
            bars_svg.append(
                f'<rect x="{x_center + 4}" y="{boost_y}" width="{bar_width - 4}" '
                f'height="{boost_height}" fill="{boost_color}" rx="4" {boost_stroke}/>'
            )
            bars_svg.append(
                f'<text x="{x_center + bar_width/2 + 4}" y="{boost_y - 5}" text-anchor="middle" '
                f'fill="{boost_color}" font-size="11">{data["boost_delta"]:+.1f}%</text>'
            )

            # X-axis label
            labels_svg.append(
                f'<text x="{x_center}" y="{chart_height - 10}" text-anchor="middle" '
                f'fill="#8b98a5" font-size="12">{data["heads"]} heads</text>'
            )

        # Y-axis
        y_axis_svg = f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{chart_height - margin_bottom}" stroke="#2f3336" stroke-width="1"/>'

        # Zero line
        zero_line_svg = f'<line x1="{margin_left}" y1="{zero_y}" x2="{chart_width - 20}" y2="{zero_y}" stroke="#8b98a5" stroke-width="1" stroke-dasharray="4,4"/>'

        # Y-axis labels
        y_labels = []
        for val in range(-6, 5, 2):
            y_pos = zero_y - val * y_scale
            y_labels.append(f'<text x="{margin_left - 10}" y="{y_pos + 4}" text-anchor="end" fill="#8b98a5" font-size="11">{val:+d}%</text>')

        # Legend
        legend_svg = f'''
            <rect x="{chart_width - 150}" y="10" width="12" height="12" fill="#f4212e" rx="2"/>
            <text x="{chart_width - 132}" y="20" fill="#8b98a5" font-size="11">Ablation</text>
            <rect x="{chart_width - 70}" y="10" width="12" height="12" fill="#00ba7c" rx="2"/>
            <text x="{chart_width - 52}" y="20" fill="#8b98a5" font-size="11">Boost</text>
        '''

        chart_svg = f'''
            <svg width="{chart_width}" height="{chart_height}" xmlns="http://www.w3.org/2000/svg">
                {y_axis_svg}
                {zero_line_svg}
                {''.join(y_labels)}
                {''.join(bars_svg)}
                {''.join(labels_svg)}
                {legend_svg}
            </svg>
        '''

        # Build results table
        table_rows = []
        for data in chart_data:
            abl_class = "delta-negative" if data["ablation_delta"] < 0 else "delta-positive"
            boost_class = "delta-positive" if data["boost_delta"] > 0 else "delta-negative"
            best_marker = ""
            if data["heads"] == best_ablation["heads"] and data["heads"] == best_boost["heads"]:
                best_marker = ' <span class="badge success">Best</span>'
            elif data["heads"] == best_ablation["heads"]:
                best_marker = ' <span class="badge error">Best Ablation</span>'
            elif data["heads"] == best_boost["heads"]:
                best_marker = ' <span class="badge success">Best Boost</span>'

            table_rows.append(f'''
                <tr>
                    <td>{data["heads"]} heads{best_marker}</td>
                    <td>{data["baseline"]:.1f}%</td>
                    <td class="{abl_class}">{data["ablation_delta"]:+.1f}%</td>
                    <td class="{boost_class}">{data["boost_delta"]:+.1f}%</td>
                </tr>
            ''')

        multihead_chart = f'''
            <h2>Effect of Number of Target Heads</h2>
            <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">
                Intervention effects vary with the number of ToM heads targeted.
                Including too many lower-ranked heads dilutes the causal effect.
            </p>

            <div class="card" style="margin-bottom: 1.5rem;">
                <h3>Accuracy Delta by Number of Heads</h3>
                <div style="overflow-x: auto; padding: 1rem 0;">
                    {chart_svg}
                </div>
            </div>

            <div class="card">
                <h3>Summary Table</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Target Heads</th>
                            <th>Baseline Accuracy</th>
                            <th>Ablation Delta</th>
                            <th>Boost Delta</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(table_rows)}
                    </tbody>
                </table>
                <p style="color: var(--text-secondary); margin-top: 1rem; font-size: 0.875rem;">
                    <strong>Key Finding:</strong> {best_ablation["heads"]} heads shows the strongest ablation effect
                    ({best_ablation["ablation_delta"]:+.1f}%), while {best_boost["heads"]} heads shows the best boost
                    ({best_boost["boost_delta"]:+.1f}%). This suggests the top ~10 heads are most causally relevant to ToM processing.
                </p>
            </div>
        '''

    content = f"""
        <h1>Intervention Experiments</h1>
        <p style="color: var(--text-secondary); margin-bottom: 2rem;">
            Results from ablating (zeroing) and boosting (scaling up) identified ToM heads.
        </p>

        <div class="card-grid">
            {''.join(cards)}
        </div>

        {task_breakdown_chart}

        {multihead_chart}
    """

    return render_page("Interventions", content, "Interventions")


def parse_response(raw: str, prompt: str) -> str:
    """
    Parse model response to extract the predicted container.

    NOTE: After each behavioral analysis run, review errors and improve this parsing.
    Common patterns to handle:
    - "1st" / "1st place" / "1st," → first container from preamble
    - "box. What" / "suitcase. Where" → strip trailing punctuation/words
    - "chest first," → strip "first,"
    - "box or the" → strip "or the"
    - "backpack if he" → strip "if he/she"
    - Repeated prompt text → extract container after last "the "
    """
    if not raw or raw == "-":
        return "-"

    raw = raw.strip()

    # Extract containers from prompt preamble: "There is a X and a Y."
    containers = []
    if "There is" in prompt:
        import re
        match = re.search(r"There is an? (\w+) and an? (\w+)\.", prompt)
        if match:
            containers = [match.group(1), match.group(2)]

    # Handle ordinal references (1st, 2nd, first, second)
    lower = raw.lower()
    if containers:
        if lower.startswith("1st") or lower.startswith("first"):
            return containers[0]
        if lower.startswith("2nd") or lower.startswith("second"):
            return containers[1]

    # Handle repeated prompt text - extract after last "the "
    if "will look for" in raw:
        parts = raw.split("the ")
        if parts:
            raw = parts[-1]

    # Strip common suffixes
    for suffix in [". What", ". Where", " first,", " or the", " if he", " if she",
                   ". How", ". Why", ",", "."]:
        if raw.endswith(suffix) or (suffix in raw):
            raw = raw.split(suffix)[0]

    # Clean up
    raw = raw.strip(" .,;:!?")

    return raw if raw else "-"


def render_tasks() -> str:
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
        """, "Tasks")

    # Build results lookup
    results_by_id = {}
    for b in behavioral:
        for detail in b.get("details", []):
            results_by_id[detail["task_id"]] = detail

    # Sort tasks by ID (natural sort)
    import re
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
            correct = result.get("correct", False)
            raw = result.get("response", "N/A")
            parsed = parse_response(raw, prompt)
            # Re-evaluate correctness based on parsed response
            parsed_correct = parsed.lower() == expected.lower()
            badge = f'<span class="badge {"success" if parsed_correct else "error"}">{"✓" if parsed_correct else "✗"}</span>'
        else:
            raw = "-"
            parsed = "-"
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
            <tr data-type="{base_type}" data-correct="{str(result.get('correct', '')).lower()}" data-id="{task_id}">
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
    correct_count = sum(1 for t in tasks if results_by_id.get(t.get("task_id", {}), {}).get("correct", False))

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
                    const prompt = row.querySelector('.col-prompt code').textContent.toLowerCase();

                    let show = true;
                    if (typeFilter && type !== typeFilter) show = false;
                    if (resultFilter && correct !== resultFilter) show = false;
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
                rows.forEach(row => {{
                    const detailsRow = document.getElementById('details-' + row.rowIndex);
                    tbody.appendChild(row);
                    if (row.nextElementSibling && row.nextElementSibling.classList.contains('details-row')) {{
                        tbody.appendChild(row.nextElementSibling);
                    }}
                }});
            }}
        </script>
    """

    return render_page("Tasks", content, "Tasks")


def render_api_stats() -> str:
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
        parsed = urlparse(self.path)
        path = parsed.path

        # Route requests
        if path == "/" or path == "":
            content = render_overview()
            content_type = "text/html"
        elif path == "/attention":
            content = render_attention()
            content_type = "text/html"
        elif path == "/attention/layers":
            content = render_layer_selectivity()
            content_type = "text/html"
        elif path == "/attention/token":
            content = render_token_attention()
            content_type = "text/html"
        elif path == "/attention/matrix":
            content = render_task_matrix()
            content_type = "text/html"
        elif path == "/attention/flow":
            content = render_attention_flow()
            content_type = "text/html"
        elif path == "/attention/clustering":
            content = render_head_clustering()
            content_type = "text/html"
        elif path == "/heads":
            content = render_heads()
            content_type = "text/html"
        elif path == "/interventions":
            content = render_interventions()
            content_type = "text/html"
        elif path == "/tasks":
            content = render_tasks()
            content_type = "text/html"
        elif path == "/api/stats":
            content = render_api_stats()
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
