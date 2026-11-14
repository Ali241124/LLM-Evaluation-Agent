
import re
from collections import Counter


def normalize_text(text: str) -> str:
    """
    Normalize text before comparing answers.

    - Convert text to lowercase.
    - Remove punctuation.
    - Remove extra whitespace.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def exact_match(predicted: str, expected: str) -> float:
    """
    Return 1.0 if both answers match after normalization.
    Otherwise, return 0.0.
    """
    return float(
        normalize_text(predicted) == normalize_text(expected)
    )


def token_f1_score(predicted: str, expected: str) -> float:
    """
    Calculate token-level F1 using word overlap.

    Precision = Common Tokens / Predicted Tokens
    Recall = Common Tokens / Expected Tokens

    F1 = 2 * Precision * Recall / (Precision + Recall)
    """

    predicted_tokens = normalize_text(predicted).split()
    expected_tokens = normalize_text(expected).split()

    if not predicted_tokens or not expected_tokens:
        return 0.0

    predicted_counts = Counter(predicted_tokens)
    expected_counts = Counter(expected_tokens)

    common_tokens = predicted_counts & expected_counts

    common_count = sum(common_tokens.values())

    if common_count == 0:
        return 0.0

    precision = common_count / len(predicted_tokens)

    recall = common_count / len(expected_tokens)

    f1 = (
        2 * precision * recall
    ) / (
        precision + recall
    )

    return round(f1, 4)
