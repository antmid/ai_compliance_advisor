from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from uuid import uuid4

# Load environment variables
load_dotenv()

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name="gdpr_collection",
    embedding_function=embeddings,
    persist_directory="data/database/absolute_database",  # Directory where the database will be saved
)

# GDPR Sections
gdpr_sections = [
    "Introduction to GDPR: The General Data Protection Regulation...",
    "Objectives of GDPR: Protect personal data...",
    "What is Personal Data?: Personal data includes...",
    # Add the remaining sections here...
]

# Convert GDPR sections into Documents
documents_vector = []
for idx, section in enumerate(gdpr_sections):
    document = Document(
        page_content=section.strip(),
        metadata={"source": f"GDPR Section {idx + 1}"},  # Metadata to identify the section
    )
    documents_vector.append(document)

# Add documents to the vector store
uuids = [str(uuid4()) for _ in range(len(documents_vector))]
vector_store.add_documents(documents=documents_vector, ids=uuids)

# The database will persist automatically if persist_directory is set
print("GDPR sections have been added to the vector database and saved successfully!")
