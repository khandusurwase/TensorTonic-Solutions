def ordinal_encoding(values: list, ordering: list) -> list:
    """
    Returns the ordinal index of every input value.
    """
    # Write code here
    mapping = {v: i for i, v in enumerate(ordering)}
    return [mapping[v] for v in values]