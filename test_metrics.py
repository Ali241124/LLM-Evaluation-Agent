
from backend.app.evaluation.metrics import (
    normalize_text,
    exact_match,
    token_f1_score,
)


def main():

    expected = (
        "Machine Learning enables computers to learn patterns."
    )

    predicted_1 = (
        "machine learning enables computers to learn patterns"
    )

    predicted_2 = (
        "Machine Learning helps computers learn patterns"
    )

    predicted_3 = (
        "Deep Learning uses neural networks"
    )

    print("EXACT MATCH TESTS")
    print("-" * 40)

    print(
        "Identical wording:",
        exact_match(predicted_1, expected)
    )

    print(
        "Different wording:",
        exact_match(predicted_2, expected)
    )

    print("\nTOKEN F1 TESTS")
    print("-" * 40)

    print(
        "Identical wording:",
        token_f1_score(predicted_1, expected)
    )

    print(
        "Partial overlap:",
        token_f1_score(predicted_2, expected)
    )

    print(
        "Different answer:",
        token_f1_score(predicted_3, expected)
    )

    print("\nNORMALIZATION TEST")
    print("-" * 40)

    print(
        normalize_text("Hello, WORLD!   AI.")
    )


if __name__ == "__main__":
    main()
