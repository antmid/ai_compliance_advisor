import os
import re


processed_folder = "data/processed/"
cleaned_folder = "data/cleaned/"

os.makedirs(cleaned_folder, exist_ok=True)

def clean_text(text):
    text = re.sub(r"Page \d+ of \d+", "", text)
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    text = re.sub(r"[^\w\s.,;:'\"!?-]", "", text)
    return text


for file_name in os.listdir(processed_folder):
    if file_name.endswith(".txt"):
        input_path = os.path.join(processed_folder, file_name)
        output_path = os.path.join(cleaned_folder, file_name)

        with open(input_path, "r", encoding="utf-8") as file:
            raw_text = file.read()

        cleaned_text = clean_text(raw_text)

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(cleaned_text)

        print(f"Text saved in: {output_path}")
