import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    # Write code here
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2 or axis not in (None, 0, 1):
        return -1

    if norm_type == "l1":
        norms = np.sum(np.abs(matrix), axis=axis, keepdims=True)

    elif norm_type == "l2":
        norms = np.sqrt(np.sum(matrix * matrix, axis=axis, keepdims=True))

    elif norm_type == "max":
        norms = np.max(np.abs(matrix), axis=axis, keepdims=True)

    else:
        return -1

    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)