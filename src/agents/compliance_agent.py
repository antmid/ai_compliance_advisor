import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

class ComplianceAgent:
    def __init__(self):
        """
        Initialize the ComplianceAgent with the Gemini API.
        """
        # Configure Gemini API with the key from .env
        genai.configure(api_key=os.getenv("GEMINIKEY"))
        self.model = "gemini-1.5-flash"  # Specify the Gemini model
        self.prompt_template = """
        Answer the following question using only the provided documents as context.
        If no useful information is found in the documents, state that no relevant data was found.

        Question: {question}

        Documents:
        {context}

        Answer:
        """

    def generate_response(self, question, documents):
        """
        Generate a structured response based on the retrieved documents.

        :param question: The user question as a string.
        :param documents: List of documents retrieved by the QueryAgent.
        :return: The generated answer as a string.
        """
        # Combine the content of the documents
        context = "\n\n".join([doc.page_content for doc in documents])
        # Format the prompt
        prompt = self.prompt_template.format(question=question, context=context)

        # Call the Gemini API for response generation
        response = genai.generate_text(
            model=self.model,
            prompt=prompt,
            temperature=0.7,
            max_output_tokens=500
        )

        # Extract the generated text
        return response.result
