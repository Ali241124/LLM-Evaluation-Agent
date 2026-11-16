
from types import SimpleNamespace

from backend.app.evaluation.ranking import (
    calculate_model_ranking
)


def create_result(
    model,
    judge,
    semantic,
    token_f1,
    latency=10.0
):
    return SimpleNamespace(
        model=model,
        predicted_answer="Generated answer",
        judge_overall_score=judge,
        semantic_similarity=semantic,
        token_f1=token_f1,
        latency=latency
    )


def main():

    print("=" * 60)
    print("MODEL SCORING AND RANKING TEST")
    print("=" * 60)

    results = [

        # Model A
        create_result(
            "model-a",
            judge=5,
            semantic=0.90,
            token_f1=0.80
        ),

        # Model B
        create_result(
            "model-b",
            judge=4,
            semantic=0.85,
            token_f1=0.70
        ),

        # Model C
        create_result(
            "model-c",
            judge=3,
            semantic=0.75,
            token_f1=0.60
        ),
    ]

    ranking = calculate_model_ranking(results)

    print("\nMODEL LEADERBOARD")
    print("=" * 60)

    for item in ranking:

        print(f"\nRank: {item['rank']}")
        print(f"Model: {item['model']}")
        print(f"Quality Score: {item['quality_score']}/100")
        print(f"Judge Score: {item['judge_score']}")
        print(f"Semantic Score: {item['semantic_score']}")
        print(f"Token F1: {item['token_f1']}")
        print(f"Successful Questions: {item['successful_questions']}")
        print(f"Average Latency: {item['avg_latency']} seconds")

    print("\nRanking test completed!")


if __name__ == "__main__":
    main()
