import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

def remove_yaml_front_matter(text):
    """
    Remove YAML front matter if present.
    YAML front matter is usually delimited by '---' at the beginning and end.
    """
    if text.startswith('---'):
        parts = text.split('---', 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return text

def load_markdown_contents(directory):
    """
    Load markdown files from the given directory.
    This function walks through the directory and collects files ending with .md or .markdown.
    """
    markdown_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".md", ".markdown")):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Remove YAML front matter if it exists
                content = remove_yaml_front_matter(content)
                markdown_files.append({"file_path": file_path, "content": content})
    return markdown_files

def split_into_chunks(text, chunk_size=200, overlap=50):
    """
    Split text into chunks of approximately 'chunk_size' words with an 'overlap' between chunks.
    This ensures that context isn't lost between adjacent chunks.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += (chunk_size - overlap)
    return chunks

def build_faiss_index(chunks, model):
    """
    Build a FAISS index for the list of text chunks using embeddings from the provided model.
    Returns the FAISS index and the corresponding embeddings.
    """
    print("Generating embeddings for chunks...")
    embeddings = model.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    # Create a FAISS index for L2 (Euclidean) distance
    index = faiss.IndexFlatL2(dimension)
    print("Adding embeddings to FAISS index...")
    index.add(embeddings)
    return index, embeddings

if __name__ == "__main__":
    # Since this script is inside the 'oscar-documentation' folder, use the current directory
    directory = ".."
    print(f"Loading markdown files from {directory}...")
    markdown_docs = load_markdown_contents(directory)
    print(f"Loaded {len(markdown_docs)} markdown files.")

    all_chunks = []
    chunk_metadata = []  # To map each chunk back to its source file and text
    for doc in markdown_docs:
        # Optionally, you could remove YAML front matter here as in the previous script
        chunks = split_into_chunks(doc["content"], chunk_size=200, overlap=50)
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_metadata.append({
                "file_path": doc["file_path"],
                "chunk": chunk
            })

    print(f"Created {len(all_chunks)} chunks from the documents.")

    # Load a pre-trained SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index, embeddings = build_faiss_index(all_chunks, model)
    print("FAISS index built successfully.")

    # Save the FAISS index and metadata for later retrieval during query time
    faiss.write_index(index, "faiss_index.index")
    with open("chunk_metadata.pkl", "wb") as f:
        pickle.dump(chunk_metadata, f)

    print("Index and metadata saved.")
