
import csv
import json

from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path

from backend.app.evaluation.engine import EvaluationResult
from backend.app.evaluation.ranking import DEFAULT_WEIGHTS


def export_reports(
    results: list[EvaluationResult],
    summary: dict,
    ranking: list[dict],
    output_dir: str = "reports"
) -> dict[str, str]:

    # Create the reports directory if it does not exist.
    reports_path = Path(output_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    # Use one timestamp for both files so they belong to the same run.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = reports_path / f"evaluation_report_{timestamp}.json"
    csv_path = reports_path / f"evaluation_results_{timestamp}.csv"

    # Convert dataclass results into dictionaries.
    result_data = [asdict(result) for result in results]

    # Build the complete JSON report.
    report_data = {
        "report_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scoring_weights": DEFAULT_WEIGHTS,
        "summary": summary,
        "ranking": ranking,
        "results": result_data
    }

    # Export JSON.
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report_data, file, indent=4, ensure_ascii=False)

    # Use EvaluationResult fields as CSV columns.
    fieldnames = [field.name for field in fields(EvaluationResult)]

    # Export individual evaluation results to CSV.
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_data)

    return {
        "json_report": str(json_path),
        "csv_report": str(csv_path)
    }
