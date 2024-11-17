from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
#from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize HuggingFace embeddings.
        :param model_name: Pretrained HuggingFace model.
        """
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str):
        """
        Generate an embedding for a query string.
        :param text: Input text.
        :return: Embedding vector.
        """
        return self.model.encode(text)

    def embed_documents(self, texts: list):
        """
        Generate embeddings for a list of documents.
        :param texts: List of input texts.
        :return: List of embedding vectors.
        """
        return self.model.encode(texts)


class QueryAgent:
    def __init__(self, db_path="data/database/gdpr_database"):
        """
        Initialize the QueryAgent with HuggingFace embeddings and vector database.
        :param db_path: Path to the directory containing the vector database.
        """
        # Initialize HuggingFace embeddings
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # Load and split raw documents
        raw_documents = TextLoader('data/cleaned/GDPR.txt').load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)

        # Load the existing database with the embedding function
        self.vector_db = Chroma.from_documents(documents, embedding_model, persist_directory=db_path)

    def query(self, question: str) -> str:
        """
        Retrieve relevant documents from the vector database.

        :param question: The user question as a string.
        :return: Content of the most relevant document as a string.
        """
        results = self.vector_db.similarity_search(question)
        return results[0].page_content if results else "No relevant documents found."

