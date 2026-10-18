import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# This script demonstrates a simple "Needle in a Haystack" scenario,
# where we have a large context of text (the "haystack")
# and we want to see if the model can find a specific piece of information (the "needle")
# when asked a question about it.

# In that scenario, we will use the full text of 'Crime and Punishment' by Fyodor Dostoevsky as our haystack,
# in which we have already injected a specific piece of information about a character named Denis.


# for this to work, you need to set the GOOGLE_API_KEY environment variable
# with your API key from https://aistudio.google.com/api-keys

# Setup OpenAI client pointing to Google's OpenAI-compatible endpoint
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

MODEL = "gemini-2.5-flash-lite" # gemini-2.0-flash, gemini-3-flash-preview, gemini-2.5-flash-lite
STREAM_ON = True

# Use this function to list available models and their details
def list_available_models():
    print("Available models:")
    models = client.models.list()
    for m in models:
        print(f"  id: {m.id}")

# Setup ChromaDB with LOCAL Embeddings.
# The model used to generate embeddings is "all-MiniLM-L6-v2", which is a small but effective model for many tasks.
# See https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 for more details.
# This model runs on your machine. No API calls, no quotas, and very fast.
local_ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# We'll use a persistent client so you don't have to re-index every time you run the script
chroma_client = chromadb.PersistentClient(path="./chroma_db")
COLLECTION_NAME = "literary_analysis_local"

# Create or get the collection where we will store our document chunks and their embeddings
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, 
    embedding_function=local_ef
)

# RAG Logic: Processing the file
# The idea is to index the text of 'Crime and Punishment'
# into a local ChromaDB collection, and then perform a similarity search 
# to retrieve relevant snippets based on a question, which will then be sent to the model 
# for generating a response.
def index_document(filepath):
    if collection.count() > 0:
        print("Collection already indexed. Skipping...")
        return

    print(f"Indexing {filepath} locally (this will take a moment)...")
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Simple paragraph chunking
    chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 100]
    
    # Indexing locally is much faster—no batching for quotas needed!
    collection.add(
        documents=chunks,
        ids=[f"id_{i}" for i in range(len(chunks))]
    )
    print(f"Indexed {len(chunks)} chunks locally.")


# get_rag_context(query, n_results=5) performs a similarity search in the ChromaDB collection 
# to retrieve the most relevant snippets based on the query.
def get_rag_context(query, n_results=5):
    # Search the vector DB for the most relevant snippets
    results = collection.query(query_texts=[query], n_results=n_results)
    # Flatten the list of documents retrieved
    return "\n---\n".join(results['documents'][0])

# Generation Logic
# Build a prompt that includes the retrieved context and the question,
# and send it to the model to generate a response.
def generate_response(question, context):
    full_prompt = f"""
    You are a literary analyst. Using only the following excerpts from the book,
    answer the question provided. If the information is not in the excerpts, say you don't know.

    EXCERPTS:
    {context}

    QUESTION:
    {question}
    """

    if STREAM_ON:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end='', flush=True)
    else:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}]
        )
        print(response.choices[0].message.content)

# Execution
def main():

    # Optional: List Google models
    # list_available_models()

    # Only index once
    if collection.count() == 0:
        index_document("docs/CrimeAndPunishment.txt")

    question = "What does this say about Denis?"
    
    print(f"\n--- RAG RETRIEVAL START ---")
    context = get_rag_context(question, n_results=5)

    # Debug - print the retrieved context to verify it's working
    print(f"Retrieved context:\n{context}")

    continue_processing = True
    if continue_processing:
        print(f"Retrieved relevant snippets. Sending to Gemini...")
        print(f"--- RESPONSE ---\n")
    
        generate_response(question, context)


# Note: You can run this script multiple times without re-indexing, 
# since the ChromaDB client is persistent and checks if the collection already has data before indexing again.

if __name__ == "__main__":
  try:
    main()
  except Exception as e:
    print(f"An error occurred: {e}")