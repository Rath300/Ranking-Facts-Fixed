import json


def read_json_file(file_path):
    """Read a JSON file and return the parsed data."""
    with open(file_path, "r") as input_file:
        return json.load(input_file)
