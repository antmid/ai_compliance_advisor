# Ai Compliance Chatbot

The Compliance Chatbot is an AI-powered web application designed to provide accurate, regulation-compliant answers to legal and compliance-related questions. It uses a multi-agent architecture: the **Query Agent** retrieves the most relevant documents from a vector database built with embeddings, while the **Compliance Agent** processes the retrieved content and generates a precise answer using a Gemini LLM.

## Features
- Document retrieval using vector embeddings.
- AI-based response generation.

## Installation
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Add your API keys (Gemini API) to a .env file like this GEMINIKEY=<your-gemini-api-key>
4. Run the app locally: `streamlit run app.py`.

## Technical Stack
- **Language Model**: Gemini-1.5-flash-latest
- **Frameworks**: LangChain, Streamlit
- **Database**: ChromaDB (vector database for document embeddings)
- **Embeddings**: HuggingFace Sentence Transformers
  
## Developed By
This project was developed by **Keita Vigano** and **Sara Borello**.
