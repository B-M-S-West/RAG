# RAG Document Question Answering System

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system for question answering based on PDF documents. It uses LangChain, Chroma vector store, and Streamlit to create an interactive interface for users to query information from a collection of documents.

## Features

- PDF document ingestion and processing
- Vector storage using Chroma DB
- Efficient document retrieval using similarity search
- Natural language question answering using a Language Model
- User-friendly web interface with Streamlit

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/rag-qa-system.git
   cd rag-qa-system
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Prepare your documents:
   - Place your PDF documents in the `data` directory.

## Usage

1. Populate the database (if not already done):
   ```
   python populate_database.py
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

4. Use the interface to ask questions about your documents.

## Project Structure

- `app.py`: Streamlit application for the user interface
- `populate_database.py`: Script to process documents and populate the vector store
- `query_data.py`: Functions for querying the RAG system
- `get_embedding_function.py`: Defines the embedding function used for document vectorization

## Optimizations

Several optimizations have been implemented to improve the system's performance:

1. **Fast Embedding Model**: Using a lightweight, efficient model from the sentence-transformers library.
2. **Optimized Chroma Settings**: Configuring Chroma for better performance.
3. **Faster LLM**: Option to use a smaller, quicker language model for faster responses.
4. **Caching**: Implementing Streamlit's caching mechanism to store and reuse results.
5. **Precomputed Embeddings**: Computing document embeddings during the database population phase.

## Customization

- To change the embedding model, modify the `get_embedding_function.py` file.
- To adjust the LLM, update the `get_llm()` function in `query_data.py`.
- To modify the chunking strategy, edit the `split_documents()` function in `populate_database.py`.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses [LangChain](https://github.com/hwchase17/langchain) for document processing and LLM integration.
- Vector storage is handled by [Chroma](https://github.com/chroma-core/chroma).
- The web interface is built with [Streamlit](https://streamlit.io/).
