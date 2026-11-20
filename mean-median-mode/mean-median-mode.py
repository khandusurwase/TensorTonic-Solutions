from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    # Write code here
    n = np.asarray(x, dtype= float)
    counts = Counter(n)
    highest_frequency = max(counts.values())
    mode = min(value for value, count in counts.items() if count == highest_frequency)

    mmm = {"mean": float(np.mean(x)), "median": float(np.median(x)), "mode": float(mode)}
    return mmm