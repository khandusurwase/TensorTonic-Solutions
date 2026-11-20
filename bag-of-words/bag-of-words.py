import numpy as np

def bag_of_words_vector(tokens: list, vocab: list) -> np.ndarray:
    """
    Returns a NumPy array with length len(vocab).
    """
    # Write code here
    vocab_index = {word: i for i, word in enumerate(vocab)}

    counts = [0] * len (vocab)

    for token in tokens:
        if token in vocab_index:
            counts[vocab_index[token]] +=1

    return np.array(counts, dtype=int)