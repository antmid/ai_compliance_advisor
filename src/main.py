from agents.query_agent import QueryAgent
from agents.compliance_agent import ComplianceAgent

# Initialize the agents
query_agent = QueryAgent()
compliance_agent = ComplianceAgent()

#query_agent.debug_database()

# Example user query
user_question = "I want to create a software for facial recognition what i have to respect?"

# 1. Retrieve relevant documents
documents = query_agent.query("Computer Vision guidelines")

# 2. Generate a structured response
if documents:
    response = compliance_agent.generate_response(user_question, documents)
    print("Generated Answer:")
    print(response)
else:
    print("No relevant documents found for the query.")
