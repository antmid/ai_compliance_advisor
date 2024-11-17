from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv
import os

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
    def __init__(self, db_path="data/database/absolute_database", embedding_model="sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the QueryAgent with HuggingFace embeddings and vector database.
        :param db_path: Path to the directory containing the vector database.
        :param embedding_model: Pretrained HuggingFace embedding model.
        """
        # Initialize HuggingFace embeddings
        self.embedding_model = HuggingFaceEmbeddings(embedding_model)

        # Load the existing database with the embedding function
        self.vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=self.embedding_model  # Embedding function is required for queries
        )
        self.retriever = self.vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def query(self, question):
        """
        Retrieve relevant documents from the vector database.

        :param question: The user question as a string.
        :return: List of relevant documents.
        """
        results = self.retriever.invoke(question)
        return results


    def list_documents(self):
        """
        List all the documents stored in the vector database.
        """
        docs = self.vector_db._collection.get()  # Access the stored documents
        if docs and "documents" in docs:
            for idx, doc in enumerate(docs["documents"]):
                print(f"Document {idx + 1}: {doc}")
        else:
            print("No documents found in the database.")

    def debug_database(self):
        """
        Check the number of documents in the existing vector database.
        """
        try:
            count = self.vector_db._collection.count()
            print(f"Number of documents in the database: {count}")
        except Exception as e:
            print(f"Error checking database: {e}")

