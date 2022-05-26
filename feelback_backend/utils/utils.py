
def to_boolean(value):
    """
    Converts multiple data types to boolean
    """

    if hasattr(value, '__iter__') and not isinstance(value, str):
        if len(value) == 0:
            return False
        elif len(value) == 1:
            value = value[0]
        else:
            return True

    if isinstance(value, bool):
        return value

    if str(value).lower() in ["true", "t", "1"]:
        return True

    return False
