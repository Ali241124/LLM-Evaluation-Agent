
from dataclasses import dataclass
from statistics import mean
from typing import Optional

from backend.app.datasets.loader import DatasetFile
from backend.app.evaluation.metrics import (
    exact_match,
    token_f1_score,
)
from backend.app.evaluation.semantic_similarity import (
    SemanticSimilarityEvaluator,
)
from backend.app.llm.model_manager import ModelManager


@dataclass
class EvaluationResult:
    """
    Stores the evaluation result for one question and one model.
    """

    sample_id: str
    model: str
    question: str
    expected_answer: str

    predicted_answer: Optional[str] = None

    exact_match: Optional[float] = None
    token_f1: Optional[float] = None
    semantic_similarity: Optional[float] = None

    latency: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    error: Optional[str] = None
    semantic_error: Optional[str] = None


class EvaluationEngine:

    def __init__(self):
        self.model_manager = ModelManager()
        self.semantic_evaluator = SemanticSimilarityEvaluator()

    def evaluate_dataset(
        self,
        dataset: DatasetFile,
        models: list[str],
    ) -> list[EvaluationResult]:
        """
        Evaluate every dataset question using each selected model.
        """

        if not models:
            raise ValueError("At least one model must be selected.")

        results = []

        for model in models:

            print(f"\nEvaluating model: {model}")

            for index, sample in enumerate(
                dataset.samples, start=1
            ):

                print(
                    f"Question {index}/{len(dataset.samples)}: "
                    f"{sample.id}"
                )

                result = EvaluationResult(
                    sample_id=sample.id,
                    model=model,
                    question=sample.question,
                    expected_answer=sample.expected_answer,
                )

                # Step 1: Generate an answer from the LLM.
                try:
                    response = self.model_manager.generate_response(
                        prompt=sample.question,
                        model=model,
                        system_prompt=(
                            "Answer the user's question accurately "
                            "and clearly. Do not invent information."
                        ),
                    )

                    result.predicted_answer = response.content
                    result.latency = response.latency
                    result.prompt_tokens = response.prompt_tokens
                    result.completion_tokens = (
                        response.completion_tokens
                    )

                except Exception as error:
                    result.error = str(error)
                    results.append(result)

                    print(f"Model request failed: {error}")
                    continue

                # Step 2: Calculate text-based metrics.
                result.exact_match = exact_match(
                    response.content,
                    sample.expected_answer,
                )

                result.token_f1 = token_f1_score(
                    response.content,
                    sample.expected_answer,
                )

                # Step 3: Calculate semantic similarity.
                try:
                    result.semantic_similarity = (
                        self.semantic_evaluator.evaluate(
                            response.content,
                            sample.expected_answer,
                        )
                    )

                except Exception as error:
                    result.semantic_error = str(error)
                    print(
                        "Semantic evaluation failed:",
                        error,
                    )

                results.append(result)

            print(f"Completed model: {model}")

        return results

    def summarize_results(
        self,
        results: list[EvaluationResult],
    ) -> list[dict]:
        """
        Calculate average evaluation metrics for each model.
        """

        models = list(dict.fromkeys(
            result.model for result in results
        ))

        summaries = []

        def average(values):
            valid_values = [
                value for value in values
                if value is not None
            ]

            if not valid_values:
                return None

            return round(mean(valid_values), 4)

        for model in models:

            model_results = [
                result for result in results
                if result.model == model
            ]

            successful = [
                result for result in model_results
                if result.error is None
            ]

            summaries.append({
                "model": model,
                "total_questions": len(model_results),
                "successful_questions": len(successful),
                "failed_questions": (
                    len(model_results) - len(successful)
                ),
                "avg_exact_match": average([
                    r.exact_match for r in successful
                ]),
                "avg_token_f1": average([
                    r.token_f1 for r in successful
                ]),
                "avg_semantic_similarity": average([
                    r.semantic_similarity for r in successful
                ]),
                "avg_latency": average([
                    r.latency for r in successful
                ]),
                "avg_prompt_tokens": average([
                    r.prompt_tokens for r in successful
                ]),
                "avg_completion_tokens": average([
                    r.completion_tokens for r in successful
                ]),
            })

        return summaries
