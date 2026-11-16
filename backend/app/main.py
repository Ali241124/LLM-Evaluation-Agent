
from pathlib import Path
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.app.datasets.loader import DatasetLoader
from backend.app.evaluation.engine import EvaluationEngine
from backend.app.evaluation.ranking import calculate_model_ranking
from backend.app.evaluation.report import export_reports
from backend.app.llm.model_manager import ModelManager


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data"
REPORT_DIRECTORY = PROJECT_ROOT / "reports"


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="LLM Evaluation Agent API",
    description=(
        "An API for evaluating, comparing, and ranking "
        "large language models."
    ),
    version="1.0.0"
)


# --------------------------------------------------
# REQUEST SCHEMA
# --------------------------------------------------

class EvaluationRequest(BaseModel):

    dataset_file: str = "sample_dataset.jsonl"

    models: list[str] = Field(
        default=["llama3.2:3b"],
        min_length=1
    )

    judge_model: str = "llama3.2:3b"

    enable_judge: bool = True


# --------------------------------------------------
# HEALTH ENDPOINT
# --------------------------------------------------

@app.get("/api/v1/health")
def health_check():

    return {
        "status": "healthy",
        "application": "LLM Evaluation Agent API",
        "version": "1.0.0"
    }


# --------------------------------------------------
# AVAILABLE MODELS ENDPOINT
# --------------------------------------------------

@app.get("/api/v1/models")
def get_available_models():

    try:
        manager = ModelManager()

        return {
            "available_models": manager.available_models
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# --------------------------------------------------
# RUN EVALUATION ENDPOINT
# --------------------------------------------------

@app.post("/api/v1/evaluations")
def run_evaluation(request: EvaluationRequest):

    # Validate the dataset filename.
    dataset_name = request.dataset_file

    if (
        Path(dataset_name).name != dataset_name
        or Path(dataset_name).suffix.lower() not in [".json", ".jsonl"]
    ):
        raise HTTPException(
            status_code=400,
            detail="Provide a valid JSON or JSONL dataset filename."
        )

    dataset_path = DATA_DIRECTORY / dataset_name

    if not dataset_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {dataset_name}"
        )

    try:

        # STEP 1: Load dataset.
        loader = DatasetLoader()
        dataset = loader.load(str(dataset_path))

        # STEP 2: Initialize evaluation engine.
        engine = EvaluationEngine(
            judge_model=request.judge_model,
            enable_judge=request.enable_judge
        )

        # STEP 3: Evaluate selected models.
        results = engine.evaluate_dataset(
            dataset=dataset,
            models=request.models
        )

        # STEP 4: Generate summary.
        summary = engine.summarize_results(results)

        # STEP 5: Rank models.
        ranking = calculate_model_ranking(results)

        # STEP 6: Export reports.
        report_paths = export_reports(
            results=results,
            summary=summary,
            ranking=ranking,
            output_dir=str(REPORT_DIRECTORY)
        )

        # STEP 7: Return API response.
        return {
            "status": "completed",
            "dataset": dataset.dataset_name,
            "summary": summary,
            "ranking": ranking,
            "results": [
                asdict(result)
                for result in results
            ],
            "reports": report_paths
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(error)}"
        ) from error
