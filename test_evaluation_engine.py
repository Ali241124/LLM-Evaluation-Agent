
from backend.app.datasets.loader import DatasetLoader
from backend.app.evaluation.engine import EvaluationEngine


def main():

    print("=" * 65)
    print("LLM EVALUATION ENGINE")
    print("=" * 65)

    # Step 1: Load the evaluation dataset.
    loader = DatasetLoader()

    dataset = loader.load(
        "data/sample_dataset.jsonl"
    )

    print("\nDataset:", dataset.dataset_name)
    print("Questions:", len(dataset.samples))

    # Step 2: Select models.
    # Start with one model to keep this first test manageable.
    selected_models = [
        "llama3.2:3b"
    ]

    # Step 3: Initialize the evaluation engine.
    engine = EvaluationEngine()

    # Step 4: Evaluate the dataset.
    results = engine.evaluate_dataset(
        dataset=dataset,
        models=selected_models,
    )

    # Step 5: Display individual results.
    print("\n" + "=" * 65)
    print("INDIVIDUAL EVALUATION RESULTS")
    print("=" * 65)

    for result in results:

        print(f"\nQuestion ID: {result.sample_id}")
        print(f"Model: {result.model}")
        print(f"Question: {result.question}")

        if result.error:
            print(f"Generation Error: {result.error}")
            continue

        print(f"Generated Answer: {result.predicted_answer}")
        print(f"Exact Match: {result.exact_match}")
        print(f"Token F1: {result.token_f1}")
        print(
            f"Semantic Similarity: "
            f"{result.semantic_similarity}"
        )
        print(f"Latency: {result.latency:.2f} seconds")
        print(f"Prompt Tokens: {result.prompt_tokens}")
        print(
            f"Completion Tokens: "
            f"{result.completion_tokens}"
        )

        if result.semantic_error:
            print(
                f"Semantic Evaluation Error: "
                f"{result.semantic_error}"
            )

    # Step 6: Calculate model averages.
    summaries = engine.summarize_results(results)

    print("\n" + "=" * 65)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 65)

    for summary in summaries:
        for key, value in summary.items():
            print(f"{key}: {value}")
        print("-" * 40)


if __name__ == "__main__":
    main()
