
from typing import Optional

from backend.app.llm.base import LLMResponse
from backend.app.llm.ollama_provider import OllamaProvider


class ModelManager:

    def __init__(self):

        self.provider = OllamaProvider()

        self.available_models = [
            "llama3.2:3b",
            "qwen2.5:7b",
            "qwen2.5-coder:1.5b"
        ]

    def list_models(self):

        return self.available_models

    def generate_response(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        response_format: Optional[str] = None,
    ) -> LLMResponse:

        if model not in self.available_models:

            raise ValueError(
                f"Model {model} is not configured."
            )

        return self.provider.generate(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            response_format=response_format,
        )

    def compare_models(
        self,
        prompt: str,
        models: Optional[list[str]] = None
    ) -> list[LLMResponse]:

        selected_models = models or self.available_models

        results = []

        for model in selected_models:

            print(f"\nEvaluating model: {model}")

            result = self.generate_response(
                prompt=prompt,
                model=model
            )

            results.append(result)

            print(f"Completed: {model}")

        return results
