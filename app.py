import streamlit as st
import os
from populate_database import load_documents, split_documents, add_to_chroma, CHROMA_PATH
from query_data import query_rag

def initialize_database():
    """
    Initialize the database if it doesn't exist.
    """
    if not os.path.exists(CHROMA_PATH):
        st.info("Initializing database. This may take a moment...")
        documents = load_documents()
        chunks = split_documents(documents)
        add_to_chroma(chunks)
        st.success("Database initialized successfully!")
    else:
        st.success("Database already exists.")

def get_rag_response(question: str) -> str:
    """
    Get a response from the RAG system.

    Args:
        question (str): The user's question.

    Returns:
        str: The RAG system's response.
    """
    return query_rag(question)

def display_chat_history():
    """
    Display the chat history.
    """
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        st.info(f"You: {question}")
        st.success(f"Assistant: {answer}")
        if i < len(st.session_state.chat_history) - 1:
            st.markdown("---")

def main():
    st.title("RAG Document Question Answering")

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize database
    initialize_database()

    # User input
    user_question = st.text_input("Ask a question about the documents:")

    if user_question:
        with st.spinner("Searching for an answer..."):
            response = get_rag_response(user_question)
        
        # Split the response into the actual response and sources
        response_text, sources = response.split("Sources:", 1)
        response_text = response_text.replace("Response:", "").strip()
        
        # Add the question and answer to the chat history
        st.session_state.chat_history.append((user_question, response_text))

        # Display the latest answer
        st.write("Answer:")
        st.write(response_text)
        
        with st.expander("View Sources"):
            st.write(sources.strip())

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("## Chat History")
        display_chat_history()

    # Option to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

if __name__ == "__main__":
    main()