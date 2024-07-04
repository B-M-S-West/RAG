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

def main():
    st.title("RAG Document Question Answering")

    # Initialize database
    initialize_database()

    # User input
    user_question = st.text_input("Ask a question about the documents:")

    if user_question:
        with st.spinner("Searching for an answer..."):
            response = query_rag(user_question)
        
        # Split the response into the actual response and sources
        response_text, sources = response.split("Sources:", 1)
        
        st.write("Answer:")
        st.write(response_text.replace("Response:", "").strip())
        
        with st.expander("View Sources"):
            st.write(sources.strip())

if __name__ == "__main__":
    main()