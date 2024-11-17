import os
import chromadb
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Client
from chromadb.config import Settings  # Import corretto per Settings

# Configurazione della cartella con i documenti
cleaned_folder = "data/cleaned/"

# Configurazione del percorso per salvare il database ChromaDB su disco
persist_directory = "data/chroma_vector_db"

# Configura l'API Key di Google Gemini
API_KEY = ''  # Usa una variabile d'ambiente per la chiave API
if not API_KEY:
    raise ValueError("Assicurati di configurare GOOGLE_API_KEY con la tua chiave Gemini API.")
genai.configure(api_key=API_KEY)

# Classe di Embedding personalizzata utilizzando il modello Gemini
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> list:
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model, content=input, task_type="RETRIEVAL_DOCUMENT", title=title)["embedding"]

# Funzione per la creazione del database vettoriale
def create_vector_database():
    embedding_function = GeminiEmbeddingFunction()

    # Suddivisione dei documenti in paragrafi
    documents = []
    for file_name in os.listdir(cleaned_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(cleaned_folder, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                paragraphs = text.split("\n\n")  # Dividi il testo in paragrafi usando due newline come delimitatore
                for paragraph in paragraphs:
                    if paragraph.strip():  # Aggiungi solo paragrafi non vuoti
                        documents.append(paragraph)

    # Creazione del client Chroma e della collection con persistenza su disco
    chroma_client = chromadb.Client(Settings(persist_directory=persist_directory))
    db = chroma_client.create_collection(name="googlecarsdatabase", embedding_function=embedding_function)

    # Aggiunta dei paragrafi al database
    for i, document in enumerate(documents):
        db.add(documents=document, ids=str(i))

    print(f"Vectorial DB created and saved at: {persist_directory}")

# Creazione del database vettoriale
create_vector_database()

# Caricamento del database salvato per verificare la persistenza
def load_and_verify_db():
    chroma_client = chromadb.Client(Settings(persist_directory=persist_directory))
    collection = chroma_client.get_collection(name="googlecarsdatabase")
    # Verifica il numero di documenti nel database
    print(f"Numero di documenti nella collezione: {collection.count()}")

load_and_verify_db()






