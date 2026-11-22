def min_max_scaling(data: list) -> list:
    """
    Returns each data column scaled to the range from 0 through 1.
    """
    # Write code here
    cols = list(zip(*data))
    mins = [min(col) for col in cols]
    maxs = [max(col) for col in cols]

    return [
        [(v - mn) / (mx - mn) if mx != mn else 0.0
         for v, mn, mx in zip(row, mins, maxs)]
        for row in data
    ]