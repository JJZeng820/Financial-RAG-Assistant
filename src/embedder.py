"""
src/embedder.py
Unified embedding interface for OpenAI and local models.
Week 5 core module.
"""

import os
import time
from typing import Literal

import numpy as np
from dotenv import load_dotenv

load_dotenv()
# Load environment variables from a .env file.
# Typically used to load OPENAI_API_KEY for accessing OpenAI API.


# Define allowed embedding model names using Literal type
# This restricts the function input to these exact strings.
EmbedModel = Literal["openai-small", "openai-large", "minilm", "finbert"]


def get_embedder(model: EmbedModel = "minilm"):
    """
    Factory function returning an embedder callable.

    If OPENAI_API_KEY is not set → automatically fallback to local model.
    """

    openai_key = os.getenv("OPENAI_API_KEY")

    # If user requests OpenAI but no key exists → fallback to local
    if model in ["openai-small", "openai-large"] and not openai_key:
        print("⚠️ No OPENAI_API_KEY found. Falling back to local MiniLM.")
        return _LocalEmbedder("all-MiniLM-L6-v2", dim=384)

    if model == "openai-small":
        return _OpenAIEmbedder("text-embedding-3-small", dim=1536)

    elif model == "openai-large":
        return _OpenAIEmbedder("text-embedding-3-large", dim=3072)

    elif model == "minilm":
        return _LocalEmbedder("all-MiniLM-L6-v2", dim=384)

    elif model == "finbert":
        return _LocalEmbedder("ProsusAI/finbert", dim=768)

    else:
        raise ValueError(f"Unknown model: {model}")


# ── OpenAI embedder ────────────────────────────────────────────────────────────

class _OpenAIEmbedder:
    """
    Embedding generator using OpenAI API.

    Converts text into numerical vectors using OpenAI embedding models.
    """

    def __init__(self, model_name: str, dim: int):
        """
        Constructor.

        Parameters
        ----------
        model_name : str
            Name of OpenAI embedding model.

        dim : int
            Dimension of embedding vectors produced by the model.
        """

        from openai import OpenAI

        # Create OpenAI client using API key from environment variable
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Store model name
        self.model_name = model_name

        # Store embedding dimension
        self.dim = dim

    def __call__(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Parameters
        ----------
        texts : list[str]
            List of input texts to embed.

        batch_size : int
            Number of texts sent to API per request.
            Helps avoid API rate limits.

        Returns
        -------
        np.ndarray
            Matrix of shape (N, dim)
            N = number of input texts
            dim = embedding vector size
        """

        all_embeddings = []  # list to store embeddings

        # Process texts in batches
        for i in range(0, len(texts), batch_size):

            batch = texts[i : i + batch_size]

            # Retry once if API request fails
            for attempt in range(2):
                try:
                    resp = self.client.embeddings.create(
                        model=self.model_name,
                        input=batch
                    )

                    # Extract embedding vectors from API response
                    vecs = [d.embedding for d in resp.data]

                    all_embeddings.extend(vecs)

                    break

                except Exception as e:

                    # Retry once after short delay
                    if attempt == 0:
                        print(f"  [retry] {e}")
                        time.sleep(2)

                    else:
                        raise

        # Convert list of embeddings to numpy float32 matrix
        return np.array(all_embeddings, dtype="float32")

    def __repr__(self):
        """
        Return a readable representation of the embedder object.
        Useful for debugging or logging.
        """
        return f"OpenAIEmbedder({self.model_name}, dim={self.dim})"


# ── Local (HuggingFace) embedder ───────────────────────────────────────────────

class _LocalEmbedder:
    """
    Embedding generator using local HuggingFace SentenceTransformer models.
    """

    def __init__(self, model_name: str, dim: int):
        """
        Constructor.

        Parameters
        ----------
        model_name : str
            HuggingFace model name.

        dim : int
            Embedding vector dimension.
        """

        from sentence_transformers import SentenceTransformer

        # Print loading message so user knows model is loading
        print(f"Loading local model: {model_name}")

        # Load local embedding model
        self.model = SentenceTransformer(model_name)

        self.model_name = model_name
        self.dim = dim

    def __call__(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Generate embeddings using local model.

        Parameters
        ----------
        texts : list[str]
            List of input texts.

        batch_size : int
            Number of texts processed per batch.

        Returns
        -------
        np.ndarray
            Normalized embedding matrix (N, dim)
        """

        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # normalize vectors for cosine similarity
            show_progress_bar=len(texts) > 100,  # show progress bar if many texts
        ).astype("float32")

    def __repr__(self):
        """
        Return readable description of embedder.
        """
        return f"LocalEmbedder({self.model_name}, dim={self.dim})"


# ── Similarity utilities ───────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Parameters
    ----------
    a : np.ndarray
        First vector

    b : np.ndarray
        Second vector

    Returns
    -------
    float
        Similarity score between -1 and 1.

    Meaning
    -------
    1   -> identical meaning
    0   -> unrelated
    -1  -> opposite meaning
    """

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def top_k_similar(
    query_vec: np.ndarray,
    corpus_vecs: np.ndarray,
    k: int = 5,
) -> list[tuple[int, float]]:
    """
    Find top-k most similar vectors.

    Parameters
    ----------
    query_vec : np.ndarray
        Embedding vector of the query.

    corpus_vecs : np.ndarray
        Matrix of document embeddings.

    k : int
        Number of top results to return.

    Returns
    -------
    list[(index, score)]
        Index of similar document and similarity score.
    """

    # Normalize query vector
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)

    # Normalize corpus vectors
    corpus_norm = corpus_vecs / (
        np.linalg.norm(corpus_vecs, axis=1, keepdims=True) + 1e-9
    )

    # Compute similarity scores
    scores = corpus_norm @ query_vec

    # Get indices of top k scores
    indices = np.argsort(scores)[::-1][:k]

    return [(int(i), float(scores[i])) for i in indices]


def compare_models(texts: list[str]) -> dict:
    """
    Compare similarity results between different embedding models.

    Parameters
    ----------
    texts : list[str]
        List of input texts.

    Returns
    -------
    dict
        Dictionary containing embeddings and similarity matrices.
    """

    results = {}

    for name, model_key in [("OpenAI-small", "openai-small"), ("MiniLM", "minilm")]:

        try:
            embedder = get_embedder(model_key)

            # Generate embeddings
            vecs = embedder(texts)

            # Create similarity matrix
            n = len(texts)
            matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(i, n):
                    sim = cosine_similarity(vecs[i], vecs[j])
                    matrix[i, j] = matrix[j, i] = sim

            results[name] = {"embedder": embedder, "vectors": vecs, "matrix": matrix}

            print(f"\n{name} similarity matrix:")

            for i, t in enumerate(texts):
                row = "  ".join(f"{matrix[i,j]:.3f}" for j in range(n))
                print(f"  [{t[:30]:30s}] {row}")

        except Exception as e:
            print(f"  {name} failed: {e}")

    return results


# ── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    DEMO_TEXTS = [
        "Apple's quarterly revenue exceeded analyst expectations",
        "Apple beat earnings estimates this quarter",
        "The Federal Reserve raised interest rates by 25 bps",
        "AAPL stock surged after strong quarterly results",
        "Nvidia reported record data center revenue",
    ]

    # Run embedding model comparison
    compare_models(DEMO_TEXTS)