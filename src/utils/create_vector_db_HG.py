from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from uuid import uuid4
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# Load environment variables
load_dotenv()

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize Chroma vector store
'''vector_store = Chroma(
    collection_name="gdpr_collection",
    embedding_function=embeddings,
    persist_directory="data/database/absolute_database",  # Directory where the database will be saved
)'''

raw_documents = TextLoader('data/cleaned/GDPR.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

db = Chroma.from_documents(documents, embeddings, collection_name="gdpr_collection",persist_directory="data/database/gdpr_database" )


