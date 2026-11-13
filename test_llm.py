
from backend.app.llm.model_manager import ModelManager


def main():

    manager = ModelManager()

    prompt = """
    Explain what Artificial Intelligence is
    in simple terms. Give one real-world example.
    """

    models_to_test = [
        "llama3.2:3b",
        "qwen2.5-coder:1.5b"
    ]

    print("=" * 60)
    print("LLM EVALUATION AGENT - FIRST MODEL TEST")
    print("=" * 60)

    results = manager.compare_models(
        prompt=prompt,
        models=models_to_test
    )

    for result in results:

        print("\n" + "=" * 60)

        print(f"MODEL: {result.model}")

        print("-" * 60)

        print("RESPONSE:")
        print(result.content)

        print("-" * 60)

        print(f"Latency: {result.latency:.2f} seconds")

        print(f"Prompt Tokens: {result.prompt_tokens}")

        print(
            f"Completion Tokens: {result.completion_tokens}"
        )

    print("\nAll model tests completed!")


if __name__ == "__main__":
    main()
