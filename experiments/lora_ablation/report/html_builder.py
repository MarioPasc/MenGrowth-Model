"""HTML report builder using Jinja2.

Renders the report template with sections, figures, and tables
into a self-contained HTML file with base64-embedded images.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from experiments.lora_ablation.report.figures import FigureResult
from experiments.lora_ablation.report.narrative import SectionContent

logger = logging.getLogger(__name__)

# Template directory
_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _get_css() -> str:
    """Return inline CSS for the report."""
    return """
/* Reset & base */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: "Georgia", "Times New Roman", serif;
    font-size: 14px;
    line-height: 1.7;
    color: #333;
    max-width: 920px;
    margin: 0 auto;
    padding: 40px 20px;
    background: #fafafa;
}

/* Header */
header {
    text-align: center;
    margin-bottom: 40px;
    padding-bottom: 20px;
    border-bottom: 2px solid #2166ac;
}

header h1 {
    font-size: 24px;
    color: #1a1a2e;
    margin-bottom: 8px;
}

header .subtitle {
    font-size: 15px;
    color: #555;
    font-style: italic;
}

header .meta {
    font-size: 12px;
    color: #888;
    margin-top: 8px;
    font-family: monospace;
}

/* Navigation / TOC */
nav {
    background: #f0f4f8;
    padding: 20px 30px;
    border-radius: 6px;
    margin-bottom: 40px;
}

nav h2 {
    font-size: 16px;
    margin-bottom: 10px;
    color: #2166ac;
}

nav ol {
    padding-left: 20px;
}

nav li {
    margin: 4px 0;
    font-size: 13px;
}

nav a {
    color: #2166ac;
    text-decoration: none;
}

nav a:hover {
    text-decoration: underline;
}

/* Sections */
section {
    margin-bottom: 40px;
    page-break-inside: avoid;
}

section h2 {
    font-size: 18px;
    color: #1a1a2e;
    border-bottom: 1px solid #ddd;
    padding-bottom: 6px;
    margin-bottom: 16px;
}

section p {
    margin-bottom: 12px;
    text-align: justify;
}

/* Figures */
figure {
    margin: 24px 0;
    text-align: center;
}

figure img {
    max-width: 100%;
    height: auto;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}

figcaption {
    font-size: 12px;
    color: #666;
    margin-top: 8px;
    font-style: italic;
    max-width: 80%;
    margin-left: auto;
    margin-right: auto;
}

/* Tables */
.table-container {
    overflow-x: auto;
    margin: 20px 0;
}

table {
    border-collapse: collapse;
    width: 100%;
    font-size: 12px;
    font-family: "Helvetica Neue", Arial, sans-serif;
}

thead th {
    background: #2166ac;
    color: white;
    padding: 8px 10px;
    text-align: left;
    font-weight: 600;
}

tbody td {
    padding: 6px 10px;
    border-bottom: 1px solid #e8e8e8;
}

tbody tr:nth-child(even) {
    background: #f8f9fa;
}

tbody tr:hover {
    background: #e8f0fe;
}

/* Footer */
footer {
    margin-top: 60px;
    padding-top: 20px;
    border-top: 1px solid #ddd;
    text-align: center;
    font-size: 12px;
    color: #999;
}

/* Print */
@media print {
    body { max-width: 100%; padding: 0; background: white; }
    nav { display: none; }
    section { page-break-inside: avoid; }
    figure { page-break-inside: avoid; }
}
"""


def html_table(df: pd.DataFrame, caption: str = "") -> str:
    """Convert a DataFrame to a styled HTML table.

    Args:
        df: Data to render.
        caption: Optional table caption.

    Returns:
        HTML string.
    """
    html = "<table>\n"
    if caption:
        html += f"<caption>{caption}</caption>\n"

    # Header
    html += "<thead><tr>"
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead>\n"

    # Body
    html += "<tbody>\n"
    for _, row in df.iterrows():
        html += "<tr>"
        for val in row:
            if isinstance(val, float):
                html += f"<td>{val:.4f}</td>"
            else:
                html += f"<td>{val}</td>"
        html += "</tr>\n"
    html += "</tbody>\n</table>"

    return html


def build_report(
    sections: List[SectionContent],
    figures: List[FigureResult],
    tables: Dict[str, str],
    output_path: Path,
    mode: str = "lora",
    num_experiments: int = 1,
) -> Path:
    """Build the complete HTML report.

    Args:
        sections: Ordered list of narrative sections.
        figures: Generated figure results.
        tables: Named HTML table strings.
        output_path: Where to write report.html.
        mode: Experiment mode (lora/dora/both).
        num_experiments: Number of experiments included.

    Returns:
        Path to generated report.html.
    """
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError:
        logger.error("Jinja2 required for HTML report generation. Install with: pip install Jinja2")
        raise

    # Build figure lookup by name
    fig_lookup: Dict[str, FigureResult] = {f.name: f for f in figures}

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=False,
    )
    template = env.get_template("report.html.j2")

    html = template.render(
        title="LoRA Ablation: Domain Adaptation Analysis",
        subtitle="Does LoRA correctly apply a domain shift from glioma to meningioma without forgetting basic MRI anatomy?",
        generated_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        mode=mode,
        num_experiments=num_experiments,
        sections=sections,
        figures=fig_lookup,
        tables=tables,
        css=_get_css(),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    logger.info("Report written to %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)

    return output_path
