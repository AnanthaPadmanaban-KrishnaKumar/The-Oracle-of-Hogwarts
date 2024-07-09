# The-Oracle-of-Hogwarts


## Overview
This project demonstrates the implementation of a Retrieval-Augmented Generation (RAG) model that utilizes state-of-the-art AI tools to generate text responses based on a corpus of books. The project leverages LangChain for data loading and preprocessing, OpenAI's embedding models for transforming text into vector space, and Pinecone to handle the vector database operations. The model is designed to answer queries by generating relevant text using the GPT-4 model, ensuring contextually rich and accurate responses.
## How It Works
1. Data Loading and Preprocessing:
    * The text data from books is loaded using PyPDFDirectoryLoader from LangChain, which handles the extraction of text from PDF files stored in a directory structure.
    * Text data is then segmented into manageable chunks using LangChain's RecursiveTextSplitter, which splits the text based on logical divisions within the content.
2. Embedding Text into Vector Space:
    * Each text chunk is embedded into a high-dimensional vector space using OpenAI's powerful embedding models. This transformation allows us to perform semantic search on the text data.
3. Storing and Retrieving Data:
    * The generated embeddings, along with their associated text chunks, are stored in Pinecone, a vector database optimized for scalability and fast retrieval.
    * For a given query, the system first converts the text of the query into its vector representation.
4. Generating Responses:
    * Using the query's vector, the system retrieves the most relevant text chunks from Pinecone.
    * These chunks are then fed into the GPT-4 model along with the query to generate coherent and contextually relevant text responses.
## Technologies Used
* LangChain: For loading and preprocessing text data from books.
* OpenAI Embedding Models: For converting text into embeddings.
* Pinecone: For storing and retrieving vector data efficiently.
* OpenAI GPT-4: For generating text based on retrieved context.
