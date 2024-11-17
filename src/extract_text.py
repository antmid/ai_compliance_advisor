import os
from PyPDF2 import PdfReader


raw_folder = "data/raw/"
processed_folder = "data/processed/"


os.makedirs(processed_folder, exist_ok=True)

def extract_text_from_pdf(pdf_path, output_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(text)
        print(f"Testo estratto e salvato in: {output_path}")
    except Exception as e:
        print(f"Errore nell'estrazione del testo da {pdf_path}: {e}")


for file_name in os.listdir(raw_folder):
    if file_name.endswith(".pdf"):
        pdf_path = os.path.join(raw_folder, file_name)
        output_path = os.path.join(processed_folder, f"{os.path.splitext(file_name)[0]}.txt")
        extract_text_from_pdf(pdf_path, output_path)

