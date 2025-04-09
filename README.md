# Oscar Chatbot

Oscar Chatbot is an open-source AI-powered chatbot assistant that answers questions from Oscar's documentation. It uses FAISS for indexing, Sentence Transformers for embeddings, and OpenAI’s GPT-3.5 for generating responses. The project features a Flask backend and a JavaScript frontend styled with a GitBook-inspired design.

## Requirements

- Python 3.8+
- pip

## Setup

1. **Clone the repository** and navigate to this directory.
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Index your documentation:**  
   Run the following command that generates index and pickle files (`chunk_metadata.pkl` and `faiss_index.index`) for all the markdown documents inside oscar-documentation:
   ```bash
   python index_documents.py
   ```
5. **Set your OpenAI API key:**  
   Export your API key as an environment variable:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

## Running

Start the Flask server:
```bash
python server.py
```
Then open [http://localhost:5000](http://localhost:5000) in your browser to chat about Oscar.

## Demo

https://github.com/user-attachments/assets/e72e67c6-d14d-45ec-b2ca-5c6c0a41a896
