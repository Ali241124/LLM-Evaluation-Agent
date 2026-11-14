
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class DatasetSample(BaseModel):
    """
    Represents one evaluation question.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the question"
    )

    question: str = Field(
        ...,
        min_length=1,
        description="Question sent to the LLM"
    )

    expected_answer: str = Field(
        ...,
        min_length=1,
        description="Reference answer for evaluation"
    )

    category: Optional[str] = None

    metadata: dict = Field(default_factory=dict)


class DatasetFile(BaseModel):
    """
    Represents a complete evaluation dataset.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_name: str = Field(
        ...,
        min_length=1
    )

    description: Optional[str] = None

    samples: list[DatasetSample] = Field(
        ...,
        min_length=1
    )
