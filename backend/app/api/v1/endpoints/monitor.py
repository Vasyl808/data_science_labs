"""FastAPI endpoint for ML model monitoring dashboards.

Returns Evidently AI HTML reports via ``/monitor`` routes.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.services.monitoring_service import (
    REPORT_GENERATORS,
    generate_all_reports,
    get_current_data,
    get_reference_data,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitor", tags=["Monitoring"])


@router.get(
    "",
    response_class=HTMLResponse,
    summary="Combined monitoring dashboard",
    description=(
        "Returns a single HTML page that embeds all available monitoring "
        "reports: Data Drift, Target Drift, Classification Performance, "
        "and Data Quality."
    ),
)
def get_combined_dashboard(db: Session = Depends(get_db)):
    """Generate and return a combined HTML dashboard with all reports."""
    try:
        reports = generate_all_reports(session=db)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    sections = []
    for name, html in reports.items():
        title = name.replace("_", " ").title()
        sections.append(
            f'<div style="margin-bottom:40px;">'
            f'<h2 style="font-family:Arial,sans-serif;color:#333;">'
            f"📊 {title} Report</h2>"
            f'<iframe srcdoc="{_escape_for_srcdoc(html)}" '
            f'width="100%" height="800px" frameborder="0"></iframe>'
            f"</div>"
        )

    combined_html = f"""<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Model Monitoring Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f7fa;
        }}
        h1 {{
            color: #1a1a2e;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2em;
        }}
        .nav {{
            display: flex;
            gap: 12px;
            justify-content: center;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        .nav a {{
            padding: 10px 20px;
            background: #4361ee;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 500;
            transition: background 0.2s;
        }}
        .nav a:hover {{
            background: #3a0ca3;
        }}
    </style>
</head>
<body>
    <h1>🔍 ML Model Monitoring Dashboard</h1>
    <div class="nav">
        <a href="/api/v1/monitor/data_drift">Data Drift</a>
        <a href="/api/v1/monitor/target_drift">Target Drift</a>
        <a href="/api/v1/monitor/classification">Classification</a>
        <a href="/api/v1/monitor/data_quality">Data Quality</a>
    </div>
    {''.join(sections)}
</body>
</html>"""

    return HTMLResponse(content=combined_html)


@router.get(
    "/{report_type}",
    response_class=HTMLResponse,
    summary="Individual monitoring report",
    description=(
        "Returns a single Evidently AI HTML report. "
        "Valid types: data_drift, target_drift, classification, data_quality."
    ),
)
def get_single_report(
    report_type: str,
    db: Session = Depends(get_db),
):
    """Generate and return a single monitoring report by type."""
    if report_type not in REPORT_GENERATORS:
        valid = ", ".join(REPORT_GENERATORS.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown report type: {report_type!r}. Valid types: {valid}",
        )

    try:
        reference = get_reference_data(db)
        current = get_current_data(db)

        if reference.empty:
            raise ValueError(
                "Reference dataset is empty — run seed_features.py first."
            )
        if current.empty:
            raise ValueError(
                "Current dataset is empty — send inference requests or "
                "run generate_inference_data.py."
            )

        generator = REPORT_GENERATORS[report_type]
        html = generator(reference, current)

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return HTMLResponse(content=html)


def _escape_for_srcdoc(html: str) -> str:
    """Escape HTML for safe embedding inside an iframe srcdoc attribute."""
    return (
        html.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
