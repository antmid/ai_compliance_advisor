# Ai Compliance Chatbot

An AI assistant that provides a list of regulations and guidelines to follow when developing artificial intelligence software, with a particular focus on computer vision. The system answers specific questions such as "What regulations must I comply with for a facial recognition system in Europe?" or "Are there guidelines for ethics in the use of computer vision?" It uses a multi-agent architecture: the **Query Agent** retrieves the most relevant documents from a vector database built with embeddings, while the **Compliance Agent** processes the retrieved content and generates a precise answer using a Gemini LLM.

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
