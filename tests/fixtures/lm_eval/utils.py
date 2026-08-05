"""lm-eval utility functions for LLMTrace smoke task.

This module provides the process_results function referenced by
llmtrace_smoke.yaml. It implements deterministic exact_match scoring.
"""


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    """Compute exact_match for the smoke task.

    Args:
        doc: The data document with 'output' key.
        results: List of model-generated outputs.

    Returns:
        Dict with 'exact_match' score (1.0 or 0.0).
    """
    target = doc.get("output", "").strip()
    if not results:
        return {"exact_match": 0.0}

    # lm-eval extracts the generation until the stop token "\n"
    generated = results[0].strip() if results else ""
    score = 1.0 if generated == target else 0.0
    return {"exact_match": score}
