"""
Voyage AI Configuration Module
Initialize and manage Voyage AI client for embeddings and semantic search.
"""

import os
from dotenv import load_dotenv
import voyageai

# Load environment variables from .env file
load_dotenv()

# Initialize Voyage AI client
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

if not VOYAGE_API_KEY:
    raise ValueError(
        "VOYAGE_API_KEY environment variable is not set. "
        "Please add your Voyage AI API key to a .env file."
    )

# Initialize the client
voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)

# Default embedding model (can be changed)
DEFAULT_MODEL = "voyage-3"


def embed_text(text: str, model: str = DEFAULT_MODEL) -> list[float]:
    """
    Generate embeddings for a single text using Voyage AI.

    Args:
        text: The text to embed
        model: The embedding model to use (default: voyage-3)

    Returns:
        A list of floats representing the embedding vector
    """
    result = voyage_client.embed(
        [text],
        model=model
    )
    return result.embeddings[0]


def embed_texts(texts: list[str], model: str = DEFAULT_MODEL) -> list[list[float]]:
    """
    Generate embeddings for multiple texts using Voyage AI.

    Args:
        texts: A list of texts to embed
        model: The embedding model to use (default: voyage-3)

    Returns:
        A list of embedding vectors
    """
    result = voyage_client.embed(
        texts,
        model=model
    )
    return result.embeddings


def rerank_results(query: str, documents: list[str], top_k: int = 5) -> list[dict]:
    """
    Rerank documents based on relevance to a query using Voyage AI.

    Args:
        query: The search query
        documents: List of documents to rerank
        top_k: Number of top results to return

    Returns:
        List of reranked documents with scores
    """
    result = voyage_client.rerank(
        query=query,
        documents=documents,
        model="rerank-1",
        top_k=top_k
    )

    reranked = []
    for item in result.results:
        reranked.append({
            "document": documents[item.index],
            "index": item.index,
            "relevance_score": item.relevance_score
        })

    return reranked
