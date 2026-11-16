
from collections import defaultdict


DEFAULT_WEIGHTS = {
    "judge": 0.50,
    "semantic": 0.30,
    "token_f1": 0.20,
}


def calculate_model_ranking(
    results: list,
    weights: dict = None
) -> list[dict]:

    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    # Validate weights
    required_metrics = ["judge", "semantic", "token_f1"]

    if set(weights) != set(required_metrics):
        raise ValueError(
            "Weights must contain judge, semantic, and token_f1."
        )

    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value < 0
        for value in weights.values()
    ):
        raise ValueError("Weights must be non-negative numbers.")

    if sum(weights.values()) <= 0:
        raise ValueError("At least one weight must be positive.")

    # Group results by model
    model_results = defaultdict(list)

    for result in results:
        model_results[result.model].append(result)

    ranking = []

    for model, model_data in model_results.items():

        successful = [
            result
            for result in model_data
            if result.predicted_answer is not None
        ]

        # Normalize available metrics
        judge_scores = []
        semantic_scores = []
        token_f1_scores = []
        latencies = []

        for result in successful:

            # Judge: 1-5 becomes 0-1
            if result.judge_overall_score is not None:
                score = result.judge_overall_score

                if 1 <= score <= 5:
                    judge_scores.append((score - 1) / 4)

            # Cosine similarity: -1 to 1 becomes 0-1
            if result.semantic_similarity is not None:
                score = result.semantic_similarity

                semantic_scores.append(
                    max(0.0, min(1.0, (score + 1) / 2))
                )

            # Token F1 is already between 0 and 1
            if result.token_f1 is not None:
                score = result.token_f1

                token_f1_scores.append(
                    max(0.0, min(1.0, score))
                )

            if result.latency is not None:
                latencies.append(result.latency)

        # Calculate averages
        metric_averages = {
            "judge": (
                sum(judge_scores) / len(judge_scores)
                if judge_scores else None
            ),
            "semantic": (
                sum(semantic_scores) / len(semantic_scores)
                if semantic_scores else None
            ),
            "token_f1": (
                sum(token_f1_scores) / len(token_f1_scores)
                if token_f1_scores else None
            ),
        }

        # Only use weights for metrics that are available
        available_weight = sum(
            weights[name]
            for name, score in metric_averages.items()
            if score is not None
        )

        if available_weight > 0:
            weighted_score = sum(
                metric_averages[name] * weights[name]
                for name in required_metrics
                if metric_averages[name] is not None
            ) / available_weight

            quality_score = round(weighted_score * 100, 2)

        else:
            quality_score = None

        ranking.append({
            "model": model,
            "quality_score": quality_score,

            "judge_score": (
                round(metric_averages["judge"], 4)
                if metric_averages["judge"] is not None
                else None
            ),

            "semantic_score": (
                round(metric_averages["semantic"], 4)
                if metric_averages["semantic"] is not None
                else None
            ),

            "token_f1": (
                round(metric_averages["token_f1"], 4)
                if metric_averages["token_f1"] is not None
                else None
            ),

            "successful_questions": len(successful),
            "total_questions": len(model_data),

            "judge_coverage": len(judge_scores),
            "semantic_coverage": len(semantic_scores),
            "token_f1_coverage": len(token_f1_scores),

            "avg_latency": (
                round(sum(latencies) / len(latencies), 2)
                if latencies else None
            ),
        })

    # Sort scored models first, highest score first
    ranking.sort(
        key=lambda item: (
            item["quality_score"] is not None,
            item["quality_score"] or 0
        ),
        reverse=True
    )

    # Assign ranks only to models with a score
    rank = 0

    for item in ranking:
        if item["quality_score"] is not None:
            rank += 1
            item["rank"] = rank
        else:
            item["rank"] = None

    return ranking
