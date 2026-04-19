def extra_repr(module):
    """
    Returns a string representation of the module's attributes.
    """
    info = ""
    for name, value in module.__dict__.items():
        if isinstance(value, (str, int, float)):
            info += f"{name}={value}, "

    if info:
        info = info[:-2]  # Remove the last comma and space

    return info
