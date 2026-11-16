
import csv
import json
import tempfile

from pathlib import Path

from backend.app.evaluation.engine import EvaluationResult
from backend.app.evaluation.report import export_reports


def main():

    print("=" * 65)
    print("REPORT GENERATION TEST")
    print("=" * 65)

    # Create sample evaluation data without calling an LLM.
    results = [
        EvaluationResult(
            sample_id="Q001",
            model="llama3.2:3b",
            question="What is Artificial Intelligence?",
            expected_answer="AI enables machines to perform intelligent tasks.",
            predicted_answer="AI enables computers to perform tasks requiring intelligence.",
            exact_match=0.0,
            token_f1=0.75,
            semantic_similarity=0.91,
            latency=12.5,
            prompt_tokens=30,
            completion_tokens=45,
            judge_overall_score=4.5
        ),
        EvaluationResult(
            sample_id="Q001",
            model="qwen2.5-coder:1.5b",
            question="What is Artificial Intelligence?",
            expected_answer="AI enables machines to perform intelligent tasks.",
            predicted_answer="AI simulates human intelligence in machines.",
            exact_match=0.0,
            token_f1=0.80,
            semantic_similarity=0.93,
            latency=8.2,
            prompt_tokens=34,
            completion_tokens=25,
            judge_overall_score=5.0
        )
    ]

    summary = {
        "total_questions": 2,
        "successful_questions": 2,
        "failed_questions": 0,
        "avg_token_f1": 0.775,
        "avg_semantic_similarity": 0.92
    }

    ranking = [
        {
            "rank": 1,
            "model": "qwen2.5-coder:1.5b",
            "quality_score": 95.0
        },
        {
            "rank": 2,
            "model": "llama3.2:3b",
            "quality_score": 85.0
        }
    ]

    # Use a temporary directory so the test does not
    # create permanent files in the project.
    with tempfile.TemporaryDirectory() as temp_dir:

        report_paths = export_reports(
            results=results,
            summary=summary,
            ranking=ranking,
            output_dir=temp_dir
        )

        json_path = Path(report_paths["json_report"])
        csv_path = Path(report_paths["csv_report"])

        # Check that both files exist.
        assert json_path.exists()
        assert csv_path.exists()

        # Check JSON contents.
        with json_path.open("r", encoding="utf-8") as file:
            json_data = json.load(file)

        assert len(json_data["results"]) == 2
        assert len(json_data["ranking"]) == 2
        assert json_data["summary"]["total_questions"] == 2

        # Check CSV contents.
        with csv_path.open(
            "r",
            newline="",
            encoding="utf-8-sig"
        ) as file:
            csv_data = list(csv.DictReader(file))

        assert len(csv_data) == 2
        assert csv_data[0]["model"] == "llama3.2:3b"
        assert csv_data[1]["model"] == "qwen2.5-coder:1.5b"

        print("\nJSON report: PASSED")
        print("CSV report: PASSED")
        print("File existence checks: PASSED")
        print("Report content checks: PASSED")

    print("\nAll report generation tests passed!")


if __name__ == "__main__":
    main()
