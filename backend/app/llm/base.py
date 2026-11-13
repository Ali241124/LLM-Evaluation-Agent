
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """
    Standard response format returned by every LLM provider.
    """

    model: str
    content: str
    latency: float

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_duration: Optional[float] = None


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2
    ) -> LLMResponse:
        """
        Generate a response using the selected LLM.
        """
        pass
