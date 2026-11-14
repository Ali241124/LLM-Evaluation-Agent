
from backend.app.evaluation.judge import LLMJudgeEvaluator


def main():

    judge = LLMJudgeEvaluator(
        judge_model="llama3.2:3b"
    )

    question = "What is Machine Learning?"

    expected_answer = (
        "Machine Learning is a subset of Artificial Intelligence "
        "that enables computers to learn patterns from data "
        "and improve their performance."
    )

    predicted_answer = (
        "Machine Learning is a branch of AI where computers "
        "learn patterns from data and use those patterns "
        "to make predictions without being explicitly programmed."
    )

    print("=" * 60)
    print("LLM-AS-A-JUDGE EVALUATION")
    print("=" * 60)

    print("\nEvaluating answer...")

    result = judge.evaluate(
        question=question,
        expected_answer=expected_answer,
        predicted_answer=predicted_answer,
    )

    print("\nJUDGE RESULTS")
    print("-" * 40)

    print("Correctness:", result.correctness, "/ 5")
    print("Relevance:", result.relevance, "/ 5")
    print("Completeness:", result.completeness, "/ 5")
    print("Clarity:", result.clarity, "/ 5")

    print(
        "\nOverall Score:",
        result.overall_score,
        "/ 5"
    )

    print("\nJudge Reasoning:")
    print(result.reasoning)


if __name__ == "__main__":
    main()
