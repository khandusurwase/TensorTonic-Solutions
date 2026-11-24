import numpy as np

def gini_impurity(y_left: list, y_right: list) -> float:
    """
    Returns the impurity as a float.
    """
    # Write code here
    def gini(y):
        if not len(y):
            return 0.0
        p = np.unique(y, return_counts=True)[1] / len(y)
        return 1 - np.sum(p ** 2)

    total = len(y_left) + len(y_right)

    return float(sum(
        len(y) / total * gini(y)
        for y in (y_left, y_right)
    )) if total else 0.0