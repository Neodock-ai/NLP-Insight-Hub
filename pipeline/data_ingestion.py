def read_file(file):
    """
    Reads an uploaded file and returns its content as a string.
    """
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        raise ValueError("Error reading file: " + str(e))
