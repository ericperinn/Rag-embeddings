import os
import base64
import hashlib
from google import genai
from google.genai import types
from pinecone import Pinecone
from dotenv import load_dotenv
import PIL.Image
import cv2

# Load environment variables
load_dotenv()

# Configure modern Google GenAI Client (Gemini 2026)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_ascii_id(prefix, text):
    """Generates an ASCII-safe ID using MD5 hash of the original string."""
    hash_obj = hashlib.md5(text.encode("utf-8"))
    return f"{prefix}_{hash_obj.hexdigest()}"


def upsert_to_pinecone(id_prefix, vector, metadata):
    """Sends vector and metadata to the Pinecone index."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    # Use hashed ID to avoid non-ASCII character errors
    vector_id = get_ascii_id(id_prefix, id_prefix + metadata.get("source", ""))
    index.upsert(vectors=[(vector_id, vector, metadata)])


def chunk_text(text, chunk_size=2000, overlap=200):
    """Splits text into chunks of fixed size with overlap to maintain context."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def process_text_file(filepath):
    """Generates embeddings for simple text files, splitting them into chunks."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        response = client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=chunk,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT", output_dimensionality=3072
            ),
        )
        embedding = response.embeddings[0].values

        upsert_to_pinecone(
            id_prefix=f"txt_chunk_{i}",
            vector=embedding,
            metadata={
                "type": "text",
                "source": filepath,
                "chunk_index": i,
                "content": chunk[
                    :1000
                ],  # Store more content in metadata for context retrieval
            },
        )
    print(f"Processed text file in {len(chunks)} chunks: {filepath}")


def process_pdf_file(filepath):
    """Extracts text from PDF, chunks it, and generates embeddings."""
    import PyPDF2

    text = ""
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"

    if text.strip():
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            response = client.models.embed_content(
                model="gemini-embedding-2-preview",
                contents=chunk,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT", output_dimensionality=3072
                ),
            )
            embedding = response.embeddings[0].values

            upsert_to_pinecone(
                id_prefix=f"pdf_chunk_{i}",
                vector=embedding,
                metadata={
                    "type": "pdf",
                    "source": filepath,
                    "chunk_index": i,
                    "content": chunk[:1000],
                },
            )
        print(f"Processed PDF file in {len(chunks)} chunks: {filepath}")


def process_image_file(filepath):
    """Generates a multimodal embedding for an image."""
    img = PIL.Image.open(filepath)

    # gemini-embedding-2-preview is multimodal by default
    response = client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=img,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    embedding = response.embeddings[0].values

    upsert_to_pinecone(
        id_prefix="img",
        vector=embedding,
        metadata={"type": "image", "source": filepath},
    )
    print(f"Processed image file: {filepath}")


def process_video_file(filepath, interval_seconds=5):
    """Extracts keyframes at intervals and generates multimodal embeddings."""
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        return
    interval_frames = int(fps * interval_seconds)

    count = 0
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval_frames == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = PIL.Image.fromarray(rgb_frame)

            response = client.models.embed_content(
                model="gemini-embedding-2-preview",
                contents=pil_img,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            embedding = response.embeddings[0].values

            timestamp = frame_count / fps
            upsert_to_pinecone(
                id_prefix=f"vid_sec_{int(timestamp)}",
                vector=embedding,
                metadata={
                    "type": "video",
                    "source": filepath,
                    "timestamp": timestamp,
                    "label": f"Frame at {int(timestamp)}s",
                },
            )
            print(f"Processed video frame at {int(timestamp)}s")
            count += 1
        frame_count += 1
    cap.release()
    print(f"Finished processing video: {filepath}")


def process_all_files(directory):
    """Reads all files in a directory and calls the appropriate processor."""
    if not os.path.exists(directory):
        return
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        ext = filename.lower().split(".")[-1]
        if ext in ["txt", "md"]:
            process_text_file(filepath)
        elif ext in ["jpg", "jpeg", "png", "webp"]:
            process_image_file(filepath)
        elif ext in ["mp4", "avi", "mov"]:
            process_video_file(filepath)
        elif ext == "pdf":
            process_pdf_file(filepath)


if __name__ == "__main__":
    # Ensure the data folder exists
    os.makedirs("data", exist_ok=True)
    process_all_files("data")
