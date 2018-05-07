import json


def save_properties(data):
    with open("properties.txt", mode="w+") as p:
        p.write(json.dumps(data))


def load_properties():
    try:
        with open("properties.txt") as p:
            return json.loads(p.read())
    except (OSError, ValueError):
        return {"comments_received": 0, "comments_classified": 0, "comments_skipped": 0}
