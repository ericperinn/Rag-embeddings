import os
from google import genai
from google.genai import types
from pinecone import Pinecone
from dotenv import load_dotenv

# Load API keys and settings
load_dotenv()

# Configure Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def query_rag(query_text, top_k=5):
    """
    Performs vector search (RAG) and generates a response based on multimodal context.
    """
    # 1. Generate embedding for query using the same model (3072 dims)
    response = client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=query_text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_vector = response.embeddings[0].values

    # 2. Search most similar vectors in Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    # 3. Format retrieved context for the LLM
    context = ""
    for match in results["matches"]:
        metadata = match["metadata"]
        m_type = metadata.get("type")
        source = metadata.get("source")

        if m_type in ["text", "pdf"]:
            context += f"\n[Document: {source}]\n{metadata.get('content')}\n"
        elif m_type == "image":
            context += f"\n[Image found: {source}]\n"
        elif m_type == "video":
            context += f"\n[Video: {source} at {metadata.get('timestamp')}s]\n"

    # 4. Generate final answer with Gemini 2.5 Flash
    prompt = f"""
    Based on the multimodal context below, answer the user's question as accurately as possible.
    If the context contains information from specific files (like PDFs, images, or videos), cite them in your response.
    
    Context:
    {context}
    
    Question: {query_text}
    """

    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text, results["matches"]


if __name__ == "__main__":
    query = input("O que você deseja saber? ")
    answer, matches = query_rag(query)
    print("\n--- Answer ---\n")
    print(answer)
    print("\n--- Consulted Sources ---")
    for m in matches:
        print(f"- {m['metadata'].get('source')} (Score: {m['score']:.4f})")
