import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    pos = np.arange(seq_len)[:, None]
    dimensions = np.arange(d_model)[None, :]
    angles = pos / base ** (2 * (dimensions // 2) /d_model)
    PE = np.where(dimensions % 2 == 0, np.sin(angles), np.cos(angles))
    return PE