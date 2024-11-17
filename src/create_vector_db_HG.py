from dotenv import load_dotenv
import os

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

from uuid import uuid4

from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="data/database",
)

cleaned_folder = "data/cleaned/"


documents = []
for file_name in os.listdir(cleaned_folder):
    if file_name.endswith(".txt"):
        file_path = os.path.join(cleaned_folder, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            paragraphs = text.split("\n\n")
            for paragraph in paragraphs:
                if paragraph.strip():
                    documents.append(paragraph)

documents_vector = []
i = 0

for document in documents:
    document1 = Document(
        page_content=document,
        metadata={"source": "data"},
        id=i,
    )
    i += 1
    documents_vector.append(document1)

uuids = [str(uuid4()) for _ in range(len(documents_vector))]

vector_store.add_documents(documents=documents_vector, ids=uuids)