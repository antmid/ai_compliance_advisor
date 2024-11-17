from agents.query_agent import QueryAgent
from agents.compliance_agent import ComplianceAgent

# Initialize the agents
query_agent = QueryAgent()
compliance_agent = ComplianceAgent()

# Example user query
user_question = "What are the European regulations for facial recognition systems?"

# 1. Retrieve relevant documents
documents = query_agent.query(user_question)

# 2. Generate a structured response
if documents:
    response = compliance_agent.generate_response(user_question, documents)
    print("Generated Answer:")
    print(response)
else:
    print("No relevant documents found for the query.")
