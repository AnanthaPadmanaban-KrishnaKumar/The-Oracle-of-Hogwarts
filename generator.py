from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

from openai import OpenAI
from tqdm.auto import tqdm

import os
from dotenv import load_dotenv

# Load API keys from environment variables
load_dotenv()
openai_api_key = os.getenv("OpenaiAPI")
os.environ['OPENAI_API_KEY'] = openai_api_key
openai_client = OpenAI()

pinecone_api_key = os.getenv("PineconeAPI")
index_name = "harrypotter"

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)
model_name = "text-embedding-3-small"

def embed(documents: list[str], model_name: str) -> list[list[float]]:
    """
    Generate embeddings for a list of documents using OpenAI.

    Args:
    documents (list[str]): A list of documents to embed.
    model_name (str): The name of the OpenAI model to use.

    Returns:
    list[list[float]]: A list of embeddings.
    """
    response = openai_client.embeddings.create(input=documents, model=model_name)
    document_embeddings = [result.embedding for result in response.data]
    return document_embeddings

def get_docs(query: str, top_k: int, model_name: str) -> list[str]:
    """
    Retrieve documents matching a query from the Pinecone index.

    Args:
    query (str): The query string.
    top_k (int): The number of top documents to retrieve.

    Returns:
    list[str]: A list of document texts matching the query.
    """
    print(f"Getting docs with {index_name}")
    query_embedding = embed([query], model_name=model_name)[0]
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    documents = [result["metadata"]['text'] for result in results["matches"]]
    return documents

def generate_answer(query: str, retrieved_docs: list[str], model_name: str) -> str:
    """
    Generate an answer to the query based on the retrieved documents using OpenAI.

    Args:
    query (str): The query string.
    retrieved_docs (list[str]): The list of retrieved documents.

    Returns:
    str: The generated answer.
    """
    context = "\n\n".join(retrieved_docs)
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "The following is a context and a question. Based on the context, provide a detailed and accurate answer to the question."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\nAnswer:"}
        ]
    )
    return response.choices[0].message.content

def execute_sample_query(query: str, top_k: int, embed_model_name: str, completion_model_name: str) -> None:
    """
    Execute a sample query and print the generated answer.

    Args:
    query (str): The query string.
    top_k (int): The number of top documents to retrieve.
    embed_model_name (str): The name of the embedding model to use.
    completion_model_name (str): The name of the completion model to use.

    Returns:
    None
    """
    retrieved_docs = get_docs(query=query, top_k=top_k, model_name=embed_model_name)
    answer = generate_answer(query=query, retrieved_docs=retrieved_docs, model_name=completion_model_name)
    print(">>>")
    print(f"Question: {query}")
    print(f"Answer: {answer}")
    print(">>>")

if __name__ == "__main__":
    # Execute a sample query
    execute_sample_query(
        query="What is the function of the Marauder's Map?",
        top_k=5, 
        embed_model_name=model_name,
        completion_model_name="gpt-4"  
    )
