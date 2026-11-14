
import time
import requests

from typing import Optional

from backend.app.llm.base import (
    BaseLLMProvider,
    LLMResponse,
)


class OllamaProvider(BaseLLMProvider):

    def __init__(
        self,
        base_url: str = "http://localhost:11434"
    ):
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        response_format: Optional[str] = None,
    ) -> LLMResponse:

        url = f"{self.base_url}/api/chat"

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        # Enable structured JSON output when requested.
        if response_format:
            payload["format"] = response_format

        start_time = time.perf_counter()

        try:

            response = requests.post(
                url,
                json=payload,
                timeout=(10, 600)
            )

            response.raise_for_status()

            result = response.json()

            latency = time.perf_counter() - start_time

            content = result.get(
                "message", {}
            ).get("content", "")

            return LLMResponse(
                model=model,
                content=content,
                latency=latency,
                prompt_tokens=result.get(
                    "prompt_eval_count"
                ),
                completion_tokens=result.get(
                    "eval_count"
                ),
                total_duration=result.get(
                    "total_duration"
                )
            )

        except requests.exceptions.RequestException as error:

            raise RuntimeError(
                f"Error communicating with Ollama: {error}"
            ) from error
