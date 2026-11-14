
from backend.app.evaluation.semantic_similarity import (
    SemanticSimilarityEvaluator
)


def main():

    evaluator = SemanticSimilarityEvaluator()

    expected = (
        "Artificial Intelligence enables machines to "
        "perform tasks that normally require human intelligence."
    )

    similar_answer = (
        "AI allows computers to perform tasks that "
        "normally require human thinking."
    )

    unrelated_answer = (
        "The weather is sunny and the temperature is high."
    )

    print("=" * 60)
    print("SEMANTIC SIMILARITY EVALUATION")
    print("=" * 60)

    print("\nComparing similar answers...")

    similar_score = evaluator.evaluate(
        similar_answer,
        expected
    )

    print(
        "Similarity score:",
        similar_score
    )

    print("\nComparing unrelated answers...")

    unrelated_score = evaluator.evaluate(
        unrelated_answer,
        expected
    )

    print(
        "Similarity score:",
        unrelated_score
    )

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
