from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

from pinecone.grpc import PineconeGRPC as pinecone
from pinecone import ServerlessSpec

from openai import OpenAI
from tqdm.auto import tqdm

import os
from dotenv import load_dotenv
import json


# Load API keys from environment variables
load_dotenv()
OpenaiAPI = os.getenv("OpenaiAPI")
os.environ['OPENAI_API_KEY'] = OpenaiAPI
client = OpenAI()

PineconeAPI = os.getenv("PineconeAPI")
index_name = "harrypotter"

## Data ingestion
def data_ingestion(directory):
    loader=PyPDFDirectoryLoader(directory)
    documents=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=800,
                                                 chunk_overlap=50)
    
    docs=text_splitter.split_documents(documents)

    for i, doc in enumerate(docs):
        doc.page_content = doc.page_content.replace("\n", " ")
        doc.metadata['id'] = f"chunk_{i}"
        doc.metadata['text'] = doc.page_content
    return docs

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


def embed(docs: list[str], name: str) -> list[list[float]]:
    res = client.embeddings.create(
        input=docs, model=name
    )
    doc_embeds = [r.embedding for r in res.data]
    return doc_embeds

# Define the query function
def get_docs(query: str, top_k: int) -> list[str]:
    print(f"Getting docs with {index_name}")
    # Encode the query
    xq = embed([query], name=model_name)[0]
    # Search the Pinecone index
    res = index.query(vector=xq, top_k=top_k, include_metadata=True)
    # Get document text
    docs = [x["metadata"]['text'] for x in res["matches"]]
    return docs

doc_directory = 'datasets/'

chunks = data_ingestion(directory=doc_directory)
print(f"Number of chunks created: {len(chunks)}")

num_chunks_to_print = 5  
for i, chunk in enumerate(chunks[:num_chunks_to_print]):
    print(f"Chunk {i+1}:")
    print(chunk)
    print("\n" + "-"*80 + "\n")

pc = create_serverless_index(index_name, 1536, "cosine", "aws", "us-east-1", PineconeAPI)
index = pc.Index(index_name)

batch_size = 200
model_name = "text-embedding-3-small"
for i in tqdm(range(0, len(chunks), batch_size)):
    i_end = min(len(chunks), i + batch_size)
    batch = chunks[i:i_end]
    embeds = embed([chunk.page_content for chunk in batch], name=model_name)
    to_upsert = [(chunk.metadata['id'], embed, chunk.metadata) for chunk, embed in zip(batch, embeds)]
    index.upsert(vectors=to_upsert)

# Execute a sample query
docs = get_docs(
    query="What is the function of the Marauder's Map?",
    top_k=5
)
print(">>>")
for doc in docs:
    print(doc)
    print(">>>")