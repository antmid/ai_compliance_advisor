from agents.query_agent import QueryAgent
from agents.compliance_agent import ComplianceAgent

query_agent = QueryAgent()
compliance_agent = ComplianceAgent()


user_question = "I want to create a software for facial recognition what i have to respect?"

documents = query_agent.query("Computer Vision guidelines")

if documents:
    response = compliance_agent.generate_response(user_question, documents)
    print("Generated Answer:")
    print(response)
else:
    print("No relevant documents found for the query.")
