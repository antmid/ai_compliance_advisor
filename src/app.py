import streamlit as st
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from pipeline import graph  # Import the compiled workflow from pipeline.py

# Initialize Streamlit app
st.set_page_config(
    page_title="Agentic RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤖 Agentic RAG Chatbot")
st.markdown(
    """
    Welcome to the **Agentic Retrieval-Augmented Generation (RAG) Chatbot**!
    Ask questions, and the chatbot will fetch relevant documents, analyze them, and generate concise responses.
    """
)

# Sidebar for instructions
with st.sidebar:
    st.header("How it works:")
    st.markdown(
        """
        1. Enter your query in the text box.
        2. The chatbot processes your query using an agentic pipeline.
        3. It retrieves documents, checks relevance, and generates an answer.
        4. Intermediate steps are shown in the output.
        """
    )
    st.markdown("---")
    st.markdown("### Example Queries:")
    st.markdown("- What does the GDPR regulate?")
    st.markdown("- What are the principles of GDPR?")
    st.markdown("- Who needs to comply with GDPR?")
    st.markdown("---")
    st.info("💡 Use this interface to explore the capabilities of the RAG pipeline!")

# Input area
query = st.text_input("💬 Enter your question:", value="")

# Output areas
st.markdown("### Intermediate Steps:")
intermediate_steps = st.empty()

st.markdown("### Final Answer:")
final_answer = st.empty()

# Run the RAG pipeline
if st.button("Submit Query") and query.strip():
    st.info("Processing your query... Please wait.")
    inputs = {"messages": [{"role": "user", "content": query}], "documents": [], "response": ""}
    intermediate_steps_output = []

    try:
        # Execute the graph
        for event in graph.astream(inputs):
            # Capture intermediate steps
            for key, value in event.items():
                intermediate_steps_output.append((key, value))
                intermediate_steps.markdown(
                    f"#### Node: `{key}`\n```json\n{value}\n```", unsafe_allow_html=True
                )

        # Extract final response
        final_step = intermediate_steps_output[-1]
        if final_step[0] == "generate_response" and "response" in final_step[1]:
            final_answer.markdown(f"**Response:** {final_step[1]['response']}")
        else:
            final_answer.error("⚠️ Unable to generate a response.")

    except Exception as e:
        st.error(f"An error occurred during execution: {e}")
