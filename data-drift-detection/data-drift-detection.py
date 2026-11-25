def detect_drift(reference_counts: list, production_counts: list, threshold: float) -> dict:
    """
    Returns a dictionary with score and drift_detected.
    """
    # Write code here
    reference_total = sum(reference_counts)
    production_total = sum(production_counts)
    reference = [count / reference_total for count in reference_counts]
    production = [count / production_total for count in production_counts]
    score = 0.5 * sum(abs(p - q) for p, q in zip(reference, production))
    return {"score": score, "drift_detected": score > threshold}