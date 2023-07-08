import uuid


def generate_image_key() -> str:
    return str(uuid.uuid4())
