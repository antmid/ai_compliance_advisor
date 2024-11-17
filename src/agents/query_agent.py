from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma


class QueryAgent:
    def __init__(self, db_path="vector_db", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the QueryAgent with HuggingFace embeddings and vector database.
        :param db_path: Path to the directory containing the vector database.
        :param embedding_model: Pretrained HuggingFace embedding model.
        """
        # Initialize HuggingFace embeddings
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize the vector database
        self.vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=self.embed_text
        )
        self.retriever = self.vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def embed_text(self, text):
        """
        Generate embeddings for the given text using HuggingFace's model.
        :param text: Text to embed.
        :return: Embedding vector.
        """
        return self.embedding_model.encode(text)

    def query(self, question):
        """
        Retrieve relevant documents from the vector database.

        :param question: The user question as a string.
        :return: List of relevant documents.
        """
        results = self.retriever.get_relevant_documents(question)
        return results
