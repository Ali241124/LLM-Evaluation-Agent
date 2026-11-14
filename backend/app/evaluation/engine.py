
from dataclasses import dataclass
from typing import Optional

from backend.app.schemas.dataset import DatasetFile
from backend.app.llm.model_manager import ModelManager
from backend.app.evaluation.metrics import exact_match, token_f1_score
from backend.app.evaluation.semantic_similarity import SemanticSimilarityEvaluator
from backend.app.evaluation.judge import LLMJudgeEvaluator


@dataclass
class EvaluationResult:

    # Dataset information
    sample_id: str
    model: str
    question: str
    expected_answer: str

    # Generated answer
    predicted_answer: Optional[str] = None

    # Automatic evaluation metrics
    exact_match: Optional[float] = None
    token_f1: Optional[float] = None
    semantic_similarity: Optional[float] = None

    # Performance metrics
    latency: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    # LLM-as-a-Judge metrics
    judge_correctness: Optional[int] = None
    judge_relevance: Optional[int] = None
    judge_completeness: Optional[int] = None
    judge_clarity: Optional[int] = None
    judge_overall_score: Optional[float] = None
    judge_reasoning: Optional[str] = None

    # Error tracking
    error: Optional[str] = None
    semantic_error: Optional[str] = None
    judge_error: Optional[str] = None


class EvaluationEngine:

    def __init__(
        self,
        judge_model: str = "llama3.2:3b",
        enable_judge: bool = True
    ):

        self.model_manager = ModelManager()

        self.semantic_evaluator = SemanticSimilarityEvaluator()

        self.enable_judge = enable_judge

        self.judge_evaluator = (
            LLMJudgeEvaluator(judge_model=judge_model)
            if enable_judge
            else None
        )

    def evaluate_dataset(
        self,
        dataset: DatasetFile,
        models: list[str]
    ) -> list[EvaluationResult]:

        results = []

        # Evaluate each selected model
        for model in models:

            print(f"\nEvaluating model: {model}")

            # Evaluate every question in the dataset
            for index, sample in enumerate(
                dataset.samples,
                start=1
            ):

                print(
                    f"Question {index}/{len(dataset.samples)}: "
                    f"{sample.id}"
                )

                result = EvaluationResult(
                    sample_id=sample.id,
                    model=model,
                    question=sample.question,
                    expected_answer=sample.expected_answer
                )

                # ---------------------------------------------
                # STEP 1: Generate answer using the LLM
                # ---------------------------------------------

                try:

                    response = self.model_manager.generate_response(
                        prompt=sample.question,
                        model=model,
                        temperature=0.2
                    )

                    result.predicted_answer = response.content
                    result.latency = response.latency
                    result.prompt_tokens = response.prompt_tokens
                    result.completion_tokens = response.completion_tokens

                except Exception as error:

                    result.error = str(error)

                    print(
                        f"Generation failed for {sample.id}: "
                        f"{result.error}"
                    )

                    results.append(result)
                    continue

                # ---------------------------------------------
                # STEP 2: Calculate Exact Match
                # ---------------------------------------------

                result.exact_match = exact_match(
                    result.predicted_answer,
                    sample.expected_answer
                )

                # ---------------------------------------------
                # STEP 3: Calculate Token F1
                # ---------------------------------------------

                result.token_f1 = token_f1_score(
                    result.predicted_answer,
                    sample.expected_answer
                )

                # ---------------------------------------------
                # STEP 4: Calculate Semantic Similarity
                # ---------------------------------------------

                try:

                    result.semantic_similarity = (
                        self.semantic_evaluator.evaluate(
                            predicted=result.predicted_answer,
                            expected=sample.expected_answer
                        )
                    )

                except Exception as error:

                    result.semantic_error = str(error)

                    print(
                        f"Semantic evaluation failed for {sample.id}: "
                        f"{result.semantic_error}"
                    )

                # ---------------------------------------------
                # STEP 5: Evaluate using LLM-as-a-Judge
                # ---------------------------------------------

                if self.enable_judge and self.judge_evaluator:

                    try:

                        judge_result = self.judge_evaluator.evaluate(
                            question=sample.question,
                            expected_answer=sample.expected_answer,
                            predicted_answer=result.predicted_answer
                        )

                        result.judge_correctness = (
                            judge_result.correctness
                        )

                        result.judge_relevance = (
                            judge_result.relevance
                        )

                        result.judge_completeness = (
                            judge_result.completeness
                        )

                        result.judge_clarity = (
                            judge_result.clarity
                        )

                        result.judge_overall_score = (
                            judge_result.overall_score
                        )

                        result.judge_reasoning = (
                            judge_result.reasoning
                        )

                    except Exception as error:

                        result.judge_error = str(error)

                        print(
                            f"Judge evaluation failed for {sample.id}: "
                            f"{result.judge_error}"
                        )

                # Store the evaluation result
                results.append(result)

        return results

    def summarize_results(
        self,
        results: list[EvaluationResult]
    ) -> dict:

        total_questions = len(results)

        # Successfully generated answers
        successful = [
            result
            for result in results
            if result.predicted_answer is not None
        ]

        # Failed answer generations
        failed = [
            result
            for result in results
            if result.predicted_answer is None
        ]

        # Helper function to calculate averages
        def average(values):

            valid_values = [
                value
                for value in values
                if value is not None
            ]

            if not valid_values:
                return None

            return round(
                sum(valid_values) / len(valid_values),
                4
            )

        # ---------------------------------------------
        # Judge evaluation statistics
        # ---------------------------------------------

        judge_successful = [
            result
            for result in successful
            if result.judge_overall_score is not None
        ]

        judge_failed = [
            result
            for result in successful
            if result.judge_error is not None
        ]

        # ---------------------------------------------
        # Semantic evaluation statistics
        # ---------------------------------------------

        semantic_failed = [
            result
            for result in successful
            if result.semantic_error is not None
        ]

        # ---------------------------------------------
        # Build summary
        # ---------------------------------------------

        summary = {

            "total_questions": total_questions,

            "successful_questions": len(successful),

            "failed_questions": len(failed),

            # Automatic evaluation metrics
            "avg_exact_match": average(
                [r.exact_match for r in successful]
            ),

            "avg_token_f1": average(
                [r.token_f1 for r in successful]
            ),

            "avg_semantic_similarity": average(
                [r.semantic_similarity for r in successful]
            ),

            # Judge evaluation metrics
            "judge_successful_questions": len(judge_successful),

            "judge_failed_questions": len(judge_failed),

            "avg_judge_correctness": average(
                [r.judge_correctness for r in successful]
            ),

            "avg_judge_relevance": average(
                [r.judge_relevance for r in successful]
            ),

            "avg_judge_completeness": average(
                [r.judge_completeness for r in successful]
            ),

            "avg_judge_clarity": average(
                [r.judge_clarity for r in successful]
            ),

            "avg_judge_overall_score": average(
                [r.judge_overall_score for r in successful]
            ),

            # Performance metrics
            "avg_latency": average(
                [r.latency for r in successful]
            ),

            "avg_prompt_tokens": average(
                [r.prompt_tokens for r in successful]
            ),

            "avg_completion_tokens": average(
                [r.completion_tokens for r in successful]
            ),

            # Semantic evaluation errors
            "semantic_failed_questions": len(semantic_failed)
        }

        return summary
