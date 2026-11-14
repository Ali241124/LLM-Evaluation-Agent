
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
        predicted_answer: str
    ) -> str:

        return f"""
You are an impartial AI evaluation judge.

Evaluate the predicted answer using the question and reference answer.

QUESTION:
{question}

REFERENCE ANSWER:
{expected_answer}

PREDICTED ANSWER:
{predicted_answer}

Evaluate the predicted answer using these four dimensions:

1. correctness:
Is the answer factually accurate and consistent with the reference?

2. relevance:
Does the answer directly address the question?

3. completeness:
Does the answer cover the important information?

4. clarity:
Is the answer clear, understandable, and well organized?

SCORING SCALE:
1 = Very poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent

IMPORTANT RULES:
- Do not reward unnecessary answer length.
- Do not penalize different wording when the meaning is correct.
- Do not assume unsupported details are correct.
- Use the reference answer as guidance, not as an exclusive wording template.
- Give honest and balanced scores.
- Each score must be an integer between 1 and 5.

Return a JSON object with exactly these fields:

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
        predicted_answer: str
    ) -> JudgeResult:

        # Validate input
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not expected_answer or not expected_answer.strip():
            raise ValueError("Expected answer cannot be empty.")

        if not predicted_answer or not predicted_answer.strip():
            raise ValueError("Predicted answer cannot be empty.")

        # Build evaluation prompt
        prompt = self.build_prompt(
            question=question,
            expected_answer=expected_answer,
            predicted_answer=predicted_answer
        )

        # Generate structured JSON response using Ollama
        response = self.model_manager.generate_response(
            prompt=prompt,
            model=self.judge_model,
            temperature=0.0,
            system_prompt=(
                "You are a strict and impartial evaluation judge. "
                "Return a valid JSON object only. "
                "Do not use Markdown or additional text."
            ),
            response_format="json"
        )

        # Extract JSON from the response
        response_text = response.content.strip()

        json_start = response_text.find("{")

        if json_start == -1:
            raise ValueError(
                "Invalid JSON returned by judge: "
                f"No JSON object found. Response: {response_text}"
            )

        try:
            data, _ = json.JSONDecoder().raw_decode(
                response_text[json_start:]
            )

        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON returned by judge: {error}. "
                f"Response: {response_text}"
            ) from error

        # Ensure the response is a JSON object
        if not isinstance(data, dict):
            raise ValueError(
                "Judge response must be a JSON object."
            )

        # Check required fields
        required_fields = [
            "correctness",
            "relevance",
            "completeness",
            "clarity",
            "reasoning"
        ]

        for field in required_fields:
            if field not in data:
                raise ValueError(
                    f"Judge response is missing required field: {field}"
                )

        # Validate scores
        score_fields = [
            "correctness",
            "relevance",
            "completeness",
            "clarity"
        ]

        for field in score_fields:

            value = data[field]

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"Judge field '{field}' must be a number."
                )

            if not 1 <= value <= 5:
                raise ValueError(
                    f"Judge field '{field}' must be between 1 and 5."
                )

            if value != int(value):
                raise ValueError(
                    f"Judge field '{field}' must be an integer."
                )

        # Validate reasoning
        if not isinstance(data["reasoning"], str):
            raise ValueError(
                "Judge reasoning must be a string."
            )

        # Calculate overall score
        overall_score = round(
            sum(data[field] for field in score_fields) / 4,
            2
        )

        # Return structured evaluation result
        return JudgeResult(
            correctness=int(data["correctness"]),
            relevance=int(data["relevance"]),
            completeness=int(data["completeness"]),
            clarity=int(data["clarity"]),
            overall_score=overall_score,
            reasoning=data["reasoning"]
        )
