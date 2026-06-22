import os

BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)

CHROMA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "chroma_db"
)
