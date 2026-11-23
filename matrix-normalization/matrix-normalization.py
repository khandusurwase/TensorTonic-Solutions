import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    # Write code here
    matrix = np.array(matrix, dtype=float)

    if matrix.ndim != 2 or axis not in (None, 0, 1):
        return -1

    if norm_type == "l1":
        norm = np.sum(np.abs(matrix), axis=axis, keepdims=True)

    elif norm_type == "l2":
        norm = np.sqrt(
            np.sum(matrix ** 2, axis=axis, keepdims=True)
        )

    elif norm_type == "max":
        norm = np.max(np.abs(matrix), axis=axis, keepdims=True)

    else:
        return -1

    return np.divide(
        matrix,
        norm,
        out=np.zeros_like(matrix),
        where=norm != 0
    )