from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader

class QueryAgent:
    def __init__(self, db_path="data/database/gdpr_database"):
        """
        Initialize the QueryAgent with HuggingFace embeddings and vector database.
        :param db_path: Path to the directory containing the vector database.
        """
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        raw_documents = TextLoader('data/cleaned/GDPR.txt').load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)

        self.vector_db = Chroma.from_documents(documents, embedding_model, persist_directory=db_path)

    def query(self, question: str) -> str:
        """
        Retrieve relevant documents from the vector database.

        :param question: The user question as a string.
        :return: Content of the most relevant document as a string.
        """
        results = self.vector_db.similarity_search(question)
        return results[0].page_content if results else "No relevant documents found."
