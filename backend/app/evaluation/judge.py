
import json
from dataclasses import dataclass

from backend.app.llm.model_manager import ModelManager


@dataclass
class JudgeResult:
    correctness: int
    relevance: int
    completeness: int
    clarity: int
    overall_score: float
    reasoning: str


class LLMJudgeEvaluator:

    def __init__(
        self,
        judge_model: str = "llama3.2:3b"
    ):
        self.model_manager = ModelManager()
        self.judge_model = judge_model

    def build_prompt(
        self,
        question: str,
        expected_answer: str,
        predicted_answer: str,
    ) -> str:

        return f"""
You are an impartial AI evaluation judge.

Evaluate the predicted answer using the question
and reference answer.

QUESTION:
{question}

REFERENCE ANSWER:
{expected_answer}

PREDICTED ANSWER:
{predicted_answer}

Evaluate these four dimensions:

1. correctness:
   Is the answer factually accurate and consistent
   with the reference answer?

2. relevance:
   Does the answer directly address the question?

3. completeness:
   Does the answer cover the important information
   required by the question?

4. clarity:
   Is the answer clear, understandable, and organized?

Scoring rules:
- 1 = Very poor
- 2 = Poor
- 3 = Acceptable
- 4 = Good
- 5 = Excellent

Do not reward an answer simply because it is long.
Do not penalize a correct answer for using different wording.
Do not assume unsupported details are correct.
Use the reference answer as guidance, not as an
exclusive wording template.

Return ONLY valid JSON in this exact structure:

{{
    "correctness": 1,
    "relevance": 1,
    "completeness": 1,
    "clarity": 1,
    "reasoning": "Brief explanation of the evaluation."
}}
"""

    def evaluate(
        self,
        question: str,
        expected_answer: str,
        predicted_answer: str,
    ) -> JudgeResult:

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if not expected_answer.strip():
            raise ValueError("Expected answer cannot be empty.")

        if not predicted_answer.strip():
            raise ValueError("Predicted answer cannot be empty.")

        prompt = self.build_prompt(
            question,
            expected_answer,
            predicted_answer,
        )

        response = self.model_manager.generate_response(
            prompt=prompt,
            model=self.judge_model,
            temperature=0.0,
            system_prompt=(
                "You are a strict and impartial evaluation judge. "
                "Return valid JSON only."
            ),
        )

        # Extract the first JSON object from the response.
        try:
            json_start = response.content.find("{")

            if json_start == -1:
                raise ValueError(
                    "Judge did not return a JSON object."
                )

            data, _ = json.JSONDecoder().raw_decode(
                response.content[json_start:]
            )

        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"Invalid JSON returned by judge: {error}"
            ) from error

        required_fields = [
            "correctness",
            "relevance",
            "completeness",
            "clarity",
            "reasoning",
        ]

        for field in required_fields:

            if field not in data:
                raise ValueError(
                    f"Judge response missing field: {field}"
                )

        scores = {}

        for field in required_fields[:4]:

            value = data[field]

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise ValueError(
                    f"Invalid score for {field}: {value}"
                )

            if value < 1 or value > 5 or value != int(value):
                raise ValueError(
                    f"{field} must be an integer from 1 to 5."
                )

            scores[field] = int(value)

        if not isinstance(data["reasoning"], str):
            raise ValueError(
                "Judge reasoning must be text."
            )

        overall_score = round(
            sum(scores.values()) / 4,
            2,
        )

        return JudgeResult(
            correctness=scores["correctness"],
            relevance=scores["relevance"],
            completeness=scores["completeness"],
            clarity=scores["clarity"],
            overall_score=overall_score,
            reasoning=data["reasoning"],
        )
