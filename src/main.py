from agents.query_agent import QueryAgent
from agents.compliance_agent import ComplianceAgent

# Initialize the agents
query_agent = QueryAgent()
compliance_agent = ComplianceAgent()

query_agent.debug_database()

# Example user query
user_question = "What does the GDPR regulate?"

# 1. Retrieve relevant documents
documents = query_agent.query(user_question)


# 2. Generate a structured response
if documents:
    response = compliance_agent.generate_response(user_question, documents)
    print("Generated Answer:")
    print(response)
else:
    print("No relevant documents found for the query.")
