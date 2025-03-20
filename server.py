import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from flask import Flask, request, jsonify, send_from_directory

# OpenAI API Key
client = OpenAI()

# Define constants for system messages and responses.
SYSTEM_MESSAGE_OSCAR = (
    "You are a helpful assistant specialized in Oscar documentation. "
    "If the question is related to Oscar documentation, provide detailed answers using the given context. "
    "For greetings or goodbyes or general queries, feel free to engage in a friendly manner."
)

# Python Flask Backend
app = Flask(__name__, static_url_path='', static_folder='static')

def load_index_and_metadata(index_path="faiss_index.index", metadata_path="chunk_metadata.pkl"):
    """
    Load the FAISS index and corresponding chunk metadata.
    """
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        chunk_metadata = pickle.load(f)
    return index, chunk_metadata

def query_chatbot(query, model, index, chunk_metadata, top_k=5):
    """
    Process the query: generate its embedding, retrieve the most similar document chunks,
    and then generate an answer via OpenAI's GPT-3.5-turbo.
    """
    # Generate query embedding
    query_embedding = model.encode(query, convert_to_numpy=True)
    query_embedding = np.expand_dims(query_embedding, axis=0)

    # Retrieve top-k similar chunks from the index
    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = []
    for idx in indices[0]:
        retrieved_chunks.append(chunk_metadata[idx]["chunk"])

    # Combine the retrieved chunks to form the context
    context = "\n\n".join(retrieved_chunks)

    # Build the prompt with the context and the question
    prompt = (
        f"{SYSTEM_MESSAGE_OSCAR}\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Question: " + query + "\nAnswer:"
    )

    # Call OpenAI's API to generate an answer
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE_OSCAR},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.2,
    )

    answer = response.choices[0].message.content.strip()
    return answer

# Load the FAISS index, metadata, and SentenceTransformer model at startup.
index, chunk_metadata = load_index_and_metadata()
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/')
def index_page():
    # Serve the frontend HTML file from the static folder.
    return send_from_directory('static', 'index.html')

@app.route('/query', methods=['POST'])
def query_endpoint():
    data = request.get_json()
    question = data.get("question", "")
    try:
        answer = query_chatbot(question, model, index, chunk_metadata, top_k=5)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask server
    app.run(debug=True)
