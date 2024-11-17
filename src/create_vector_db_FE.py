from langchain_core.embeddings import FakeEmbeddings
from uuid import uuid4
from langchain_core.documents import Document
from langchain_chroma import Chroma

embeddings = FakeEmbeddings(size=4096)
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)


file1_path = "data/cleaned/EU_AI_Act.txt"
file2_path = "data/cleaned/GDPR.txt"
file3_path = "data/cleaned/ISO_AI_Guidelines.txt"

# Leggi i file e assegnali alle variabili
with open(file1_path, 'r', encoding='utf-8') as file:
    DOCUMENT1 = file.read()

with open(file2_path, 'r', encoding='utf-8') as file:
    DOCUMENT2 = file.read()

with open(file3_path, 'r', encoding='utf-8') as file:
    DOCUMENT3 = file.read()


document_1 = Document(
    page_content=DOCUMENT1,
    metadata={"source": "law"},
    id=1,
)

document_2 = Document(
    page_content= DOCUMENT2,
    metadata={"source": "law"},
    id=2,
)

document_3 = Document(
    page_content=DOCUMENT3,
    metadata={"source": "news"},
    id=3,
)

documents = [
    document_1,
    document_2,
    document_3]

uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)