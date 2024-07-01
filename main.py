import os
from dotenv import load_dotenv
import json


import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

import openai
from pinecone.grpc import PineconeGRPC as pinecone
from pinecone import ServerlessSpec

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import PineconeRetriever

# Load API keys from environment variables
load_dotenv()
OpenaiAPI = os.getenv("OpenaiAPI")
PineconeAPI = os.getenv("PineconeAPI")
index_name = "harrypotter"

os.environ['PINECONE_API_KEY'] = PineconeAPI

## Data ingestion
def data_ingestion(directory):
    loader=PyPDFDirectoryLoader(directory)
    documents=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
                                                 chunk_overlap=50)
    
    docs=text_splitter.split_documents(documents)
    return docs

# Function to save text and embeddings to a JSON file
def save_text_embeddings(texts, embeddings, filename):
    data = []
    for text, embedding in zip(texts, embeddings):
        data.append({
            'text': text,
            'embedding': embedding
        })
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# Function to create a serverless index using Pinecone
def create_serverless_index(name, dimension, metric, cloud, region, api_key):
    pc = pinecone(api_key=api_key)
    pc.create_index(
        name=name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud=cloud,
            region=region
        )
    )
    print(f"Serverless index '{name}' created successfully.")
    return pc

# Function to embed a question
def embed_question(question, embedding_model):
    question_embedding = embedding_model.embed([question])
    return question_embedding[0]

# Function to retrieve relevant documents using embedded question
def retrieve_documents(vector_store, question_embedding, top_k=5):
    retriever = PineconeRetriever(vector_store)
    retrieved_docs = retriever.retrieve_by_vector(question_embedding, top_k=top_k)
    return retrieved_docs

# Define a function to generate answers using GPT-3
def generate_answers(retrieved_docs, question):
    combined_context = " ".join([doc['text'] for doc in retrieved_docs])
    prompt = f"Context: {combined_context}\n\nQuestion: {question}\nAnswer:"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        n=1,
        stop=["\n"]
    )
    answer = response.choices[0].text.strip()
    return answer

def rag_pipeline(question):
    question_embedding = embed_question(question, embeddings)
    retrieved_docs = retrieve_documents(vectorstore_from_docs, question_embedding, top_k=5)
    answer = generate_answers(retrieved_docs, question) 
    
    return answer



doc_directory = 'datasets/'

chunks = data_ingestion(directory=doc_directory)
print(f"Number of chunks created: {len(chunks)}")

pc = create_serverless_index(index_name, 1536, "cosine", "aws", "us-east-1", PineconeAPI)
index = pc.Index(index_name)

embeddings=OpenAIEmbeddings(api_key=OpenaiAPI)
print(embeddings)

vectorstore_from_docs = PineconeVectorStore.from_documents(
    chunks,
    index_name=index_name,
    embedding=embeddings
)

question = "What is the main theme of the first Harry Potter book?"

answer = rag_pipeline(question)
print(f"Answer: {answer}")