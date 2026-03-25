import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()


def init_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "multimodal-rag-index")

    if not api_key:
        print("Error: PINECONE_API_KEY not found in .env")
        return

    # Modern Pinecone client initialization
    pc = Pinecone(api_key=api_key)

    # Check if index exists and has correct dimension
    existing_indexes = pc.list_indexes()
    index_names = [idx.name for idx in existing_indexes]

    if index_name in index_names:
        desc = pc.describe_index(index_name)
        if desc.dimension != 3072:
            print(
                f"Index {index_name} has wrong dimension ({desc.dimension}). Recreating with 3072..."
            )
            pc.delete_index(index_name)
            import time

            time.sleep(5)  # Wait for deletion
        else:
            print(f"Index {index_name} already exists with correct dimension (3072).")
            return

    print(f"Creating Pinecone index: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=3072,  # Dimension for gemini-embedding-2-preview
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("Index created successfully.")


if __name__ == "__main__":
    init_pinecone()
