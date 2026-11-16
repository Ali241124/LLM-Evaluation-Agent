
from backend.app.datasets.loader import DatasetLoader

from backend.app.evaluation.engine import EvaluationEngine


def main():

    print("=" * 65)
    print("LLM EVALUATION ENGINE")
    print("=" * 65)

    # STEP 1: Load dataset
    loader = DatasetLoader()

    dataset = loader.load(
        "data/sample_dataset.jsonl"
    )

    print("\nDataset:", dataset.dataset_name)
    print("Questions:", len(dataset.samples))

    # STEP 2: Select models
    selected_models = [
        "llama3.2:3b"
    ]

    # STEP 3: Initialize engine
    engine = EvaluationEngine(
        judge_model="llama3.2:3b",
        enable_judge=True,
    )

    # STEP 4: Run evaluation
    results = engine.evaluate_dataset(
        dataset=dataset,
        models=selected_models,
    )

    # STEP 5: Display individual results
    print("\n" + "=" * 65)
    print("INDIVIDUAL EVALUATION RESULTS")
    print("=" * 65)

    for result in results:

        print(f"\nQuestion ID: {result.sample_id}")
        print(f"Model: {result.model}")
        print(f"Question: {result.question}")

        if result.error:
            print("\nGENERATION FAILED")
            print("-" * 35)
            print("Sample ID:", result.sample_id)
            print("Model:", result.model)
            print("Error:", result.error)
            continue

        print(f"Generated Answer: {result.predicted_answer}")

        print("\nTEXT METRICS")
        print("-" * 35)

        print("Exact Match:", result.exact_match)
        print("Token F1:", result.token_f1)
        print("Semantic Similarity:", result.semantic_similarity)

        if result.semantic_error:
            print("Semantic Error:", result.semantic_error)

        print("\nPERFORMANCE METRICS")
        print("-" * 35)

        print(
            f"Latency: {result.latency:.2f} seconds"
            if result.latency is not None
            else "Latency: N/A"
        )

        print("Prompt Tokens:", result.prompt_tokens)
        print("Completion Tokens:", result.completion_tokens)

        print("\nLLM JUDGE SCORES")
        print("-" * 35)

        print("Correctness:", result.judge_correctness, "/ 5")
        print("Relevance:", result.judge_relevance, "/ 5")
        print("Completeness:", result.judge_completeness, "/ 5")
        print("Clarity:", result.judge_clarity, "/ 5")
        print("Judge Overall:", result.judge_overall_score, "/ 5")
        print("Judge Reasoning:", result.judge_reasoning)

        if result.judge_error:
            print("Judge Error:", result.judge_error)

    # STEP 6: Calculate model summary
    summary = engine.summarize_results(results)

    # STEP 7: Display model performance summary
    print("\n" + "=" * 65)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 65)

    print("Summary type:", type(summary))

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("-" * 40)
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
