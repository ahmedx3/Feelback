
def to_boolean(value):
    """
    Converts string to boolean
    """

    if isinstance(value, bool):
        return value

    if value in ["True", "T", "true", "t", "1"]:
        return True
    return False
