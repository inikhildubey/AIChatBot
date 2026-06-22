import chromadb
from configs.settings import CHROMA_PATH

client = chromadb.PersistentClient(CHROMA_PATH)


# client = chromadb.Client(
#     chromadb.Settings(
#         persist_directory="./chroma_db"
#     )
# )

def get_collection():
    collection = client.get_or_create_collection(name="documents")
    return collection
