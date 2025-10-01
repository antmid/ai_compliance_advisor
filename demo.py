"""
AI Compliance Advisor - Gradio Demo
For University Assignment - Anthony Middlemiss
"""

import gradio as gr
import os
from langchain_core.messages import HumanMessage
from src.pipeline import graph

def compliance_query(question):
    """
    Real AI Compliance Advisor using LangGraph + ChromaDB + Gemini.
    """
    try:
        inputs = {"messages": [HumanMessage(content=question)]}

        outputs = []
        for output in graph.stream(inputs):
            for key, value in output.items():
                outputs.append((key, value))

        if outputs:
            final = outputs[-1]
            response = final[1].get("response", None)

            if response:
                return f"🤖 **AI Compliance Advisor Response:**\n\n{response}\n\n---\n✅ *Generated using LangGraph multi-agent system with ChromaDB document retrieval and Google Gemini LLM.*"
            else:
                return "⚠️ No response generated. Check API key configuration."
        else:
            return "⚠️ No output from AI pipeline."

    except Exception as e:
        return f"❌ **Error:** {str(e)}\n\nPlease verify:\n• GEMINIKEY environment variable is set\n• All dependencies installed\n• Internet connection active"

# Create Gradio interface
demo = gr.Interface(
    fn=compliance_query,
    inputs=gr.Textbox(
        label="Ask a compliance question",
        placeholder="Try: 'What does GDPR say about AI systems?' or 'How does the EU AI Act classify AI?'",
        lines=3
    ),
    outputs=gr.Textbox(
        label="AI Response",
        lines=15
    ),
    title="🦷 AI Compliance Advisor for Dental Dashboard",
    description="""
    **Real AI System for University Assignment**

    Technologies:
    - **LangGraph**: Multi-agent pipeline for intelligent query processing
    - **ChromaDB**: Vector database with GDPR, EU AI Act, and ISO compliance documents
    - **Google Gemini**: Large language model for contextual response generation

    **Application:** Provides automated regulatory compliance guidance for dental AI dashboard,
    covering data privacy (HIPAA/GDPR), medical device regulations (FDA), and AI governance standards.
    """,
    examples=[
        ["What does the GDPR regulate regarding AI systems?"],
        ["How does the EU AI Act classify different types of AI?"],
        ["What are the data privacy requirements for healthcare AI?"],
        ["What regulations apply to AI diagnostic tools?"],
        ["What consent requirements exist for AI processing patient data?"]
    ],
    cache_examples=False
)

if __name__ == "__main__":
    print("🚀 Launching AI Compliance Advisor...")
    print("📊 Using LangGraph + ChromaDB + Google Gemini")
    print("\nWait for public URL to appear below...\n")

    demo.launch(share=True, debug=True)
