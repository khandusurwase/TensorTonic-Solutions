import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if not seqs:
        return np.empty((0, 0), dtype = int)
    max_len = max_len if max_len is not None else max(map(len, seqs))

    return np.array([
        seq[: max_len] + [pad_value] * max(0, max_len - len(seq)) for seq in seqs
    ])