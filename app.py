import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph
from src.pipeline import graph  # Import the compiled graph from your orchestrator

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
    inputs = {"messages": [HumanMessage(content=query)]}
    outputs = []

    # Execute the pipeline and display intermediate steps
    for output in graph.stream(inputs):
        for key, value in output.items():
            # Log each node's output
            outputs.append((key, value))
            intermediate_steps.markdown(
                f"#### Node: `{key}`\n```json\n{value}\n```", unsafe_allow_html=True
            )

    # Extract final response
    final = outputs[-1]
    if isinstance(final[1], list) and isinstance(final[1][0], (AIMessage, ToolMessage)):
        final_answer.markdown(f"**Response:** {final[1][0].content}")
    else:
        final_answer.error("⚠️ Unable to generate a response.")
