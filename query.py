import os
from google import genai
from google.genai import types
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Configure Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def query_rag(query_text, top_k=5):
    # 1. Embed query using gemini-embedding-2-preview (3072 dimensions)
    response = client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=query_text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_vector = response.embeddings[0].values

    # 2. Search Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    # 3. Format Context
    context = ""
    for match in results["matches"]:
        metadata = match["metadata"]
        m_type = metadata.get("type")
        source = metadata.get("source")

        if m_type in ["text", "pdf"]:
            context += f"\n[Documento: {source}]\n{metadata.get('content')}\n"
        elif m_type == "image":
            context += f"\n[Imagem encontrada: {source}]\n"
        elif m_type == "video":
            context += f"\n[Vídeo: {source} no tempo {metadata.get('timestamp')}s]\n"

    # 4. Generate Answer with Gemini 2.5 Flash
    prompt = f"""
    Baseado no contexto multimodal abaixo, responda à pergunta do usuário.
    Se o contexto contiver informações sobre imagens ou vídeos, cite-os de forma inteligente.
    
    Contexto:
    {context}
    
    Pergunta: {query_text}
    """

    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text, results["matches"]


if __name__ == "__main__":
    query = input("O que você deseja saber? ")
    answer, matches = query_rag(query)
    print("\n--- Resposta ---\n")
    print(answer)
    print("\n--- Fontes ---")
    for m in matches:
        print(f"- {m['metadata'].get('source')} (Score: {m['score']:.4f})")
