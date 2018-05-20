import json


def save_classification_data(filename, comments_classified, comments_processed):
    """Save updated data to 'properties.txt' as a json."""
    with open("properties.txt", mode="r+") as p:
        file_data = json.loads(p.read())
        file_data["comments_classified"] = comments_classified
        file_data[filename]["comments_processed"] = comments_processed
        p.seek(0)
        p.write(json.dumps(file_data))
        p.truncate()


def load_classification_data(filename):
    """

    :param filename:
    :return:
    """
    try:
        with open("properties.txt") as p:
            file_data = json.loads(p.read())

            if filename in file_data:
                return file_data["comments_classified"], file_data[filename]["comments_processed"]
            else:
                return file_data["comments_classified"], 0
    except (OSError, ValueError):
        return 0, 0


def save_comments_number(filename, number):
    with open("properties.txt", mode="r+") as p:
        file_data = json.loads(p.read())
        if filename in file_data:
            file_data[filename]["comments_number"] = number
        else:
            file_data[filename] = {"comments_number": number, "comments_processed": 0}
        p.seek(0)
        p.write(json.dumps(file_data))
        p.truncate()


def load_comments_number(filename):
    try:
        with open("properties.txt") as p:
            file_data = json.loads(p.read())

            if filename in file_data:
                return file_data[filename]["comments_number"]
            else:
                return 0
    except (OSError, ValueError):
        return 0
