from transformers import pipeline


class ComplianceAgent:
    def __init__(self, model="Gemini", temperature=0.7):
        """
        Initialize the ComplianceAgent with the Gemini LLM.
        :param model: Gemini model identifier (or a HuggingFace-compatible LLM).
        :param temperature: Generation temperature.
        """
        # Initialize the LLM pipeline
        self.llm = pipeline("text-generation", model=model, device=0)  # Ensure your model is on the correct device
        self.temperature = temperature
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
        # Generate the response
        response = self.llm(prompt, max_length=500, num_return_sequences=1, temperature=self.temperature)
        return response[0]['generated_text']
