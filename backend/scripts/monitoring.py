"""
scripts/monitoring.py
---------------------
Standalone script that connects to the database, extracts reference and
current data, generates 4 Evidently AI HTML monitoring reports, and saves
them to ``backend/reports/``.

Usage:
    cd backend
    python scripts/monitoring.py
"""

import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    from pathlib import Path
    from app.database import SessionLocal
    from app.services.monitoring_service import MonitoringService, REPORTS_DIR

    logger.info("=" * 60)
    logger.info("ML Model Monitoring — Report Generation")
    logger.info("=" * 60)

    reports_dir = REPORTS_DIR
    logger.info("Reports will be saved to: %s", reports_dir)

    try:
        with SessionLocal() as session:
            service = MonitoringService(session)
            reports = service.generate_all_reports(save_dir=reports_dir)
    except ValueError as exc:
        logger.error("Cannot generate reports: %s", exc)
        sys.exit(1)

    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    for name, html in reports.items():
        filepath = reports_dir / f"{name}_report.html"
        status = "✓ OK" if not html.startswith("<html><body><h1>Error") else "✗ FAILED"
        size_kb = len(html.encode("utf-8")) / 1024
        logger.info("  %-25s %s  (%.1f KB)", f"{name}_report.html", status, size_kb)

    logger.info("")
    logger.info("Open the HTML files in your browser to view the dashboards.")
    logger.info("Done!")


if __name__ == "__main__":
    main()
