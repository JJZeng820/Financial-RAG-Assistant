"""
src/vectorstore.py
Vector store implementations: FAISS (local) and Pinecone (cloud).
Week 6 core module.
"""
from __future__ import annotations
# Enables postponed evaluation of type annotations.
# This allows modern type hints like `list[dict]` instead of `List[Dict]`.
# It also prevents circular import issues when referencing classes not yet defined.


import json
# Built-in module used to read/write JSON data.
# In this project it is typically used to save metadata to disk.


import os
# Provides access to operating system functionality.
# Commonly used to read environment variables like API keys.


import time
# Provides time-related functions.
# Often used for measuring latency or adding delays in API calls.


import uuid
# Generates universally unique identifiers.
# Useful when assigning unique IDs to vectors stored in a database.


from pathlib import Path
# Modern object-oriented path handling.
# Safer and cleaner than using os.path.
# Example: creating directories or reading/writing files.


from typing import Protocol
# Protocol defines an interface (a structural type).
# Any class implementing the same methods automatically satisfies this interface.
# This is commonly used to define a common API for multiple implementations.


import numpy as np
# NumPy is the core numerical computing library in Python.
# It is used to handle embedding vectors and matrix operations.


from dotenv import load_dotenv
# Loads environment variables from a .env file.
# This is commonly used to store API keys securely.


load_dotenv()
# Executes the loading of the .env file so environment variables become available.
# Example:
# PINECONE_API_KEY=xxxx


# ── Protocol (common interface) ────────────────────────────────────────────────

class VectorStore(Protocol):
# Defines a protocol (interface) named VectorStore.
# Any class implementing these methods can be treated as a VectorStore.
# This allows different vector database implementations to be interchangeable.


    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None: ...
    # Method definition for adding vectors to the database.
    #
    # Parameters:
    # embeddings : np.ndarray
    #     A matrix of embedding vectors with shape (N, dimension)
    #
    # metadata : list[dict]
    #     A list of metadata dictionaries associated with each vector
    #
    # Return:
    # None (the method modifies the vector store)


    def search(self, query_vec: np.ndarray, k: int) -> list[dict]: ...
    # Method definition for performing similarity search.
    #
    # Parameters:
    # query_vec : np.ndarray
    #     The embedding vector of the query
    #
    # k : int
    #     Number of nearest results to return
    #
    # Return:
    # list[dict]
    #     Top-k results with metadata and similarity scores


    def save(self, path: str) -> None: ...
    # Method definition for saving the vector index to disk.
    #
    # Parameter:
    # path : str
    #     File path where the vector index should be saved.


    def load(self, path: str) -> None: ...
    # Method definition for loading a vector index from disk.
    #
    # Parameter:
    # path : str
    #     File path where the vector index is stored.


    def __len__(self) -> int: ...
    # Special method returning the number of vectors stored in the database.
    #
    # Example usage:
    # len(vector_store)


# ── FAISS ──────────────────────────────────────────────────────────────────────

class FAISSStore:
    """
    Local FAISS index with JSON metadata sidecar.
    Best for dev / offline use. Supports IndexFlatIP (exact) and
    IndexIVFFlat (approximate, faster for >100k vectors).
    """

    def __init__(self, dim: int = 1536, index_type: str = "flat"):
        """
        Args:
            dim:        embedding dimension
            index_type: 'flat'  — exact search, always accurate
                        'ivf'   — approximate, 10-100× faster at scale
        """
        import faiss  # type: ignore

        self.dim = dim
        self.metadata: list[dict] = []

        if index_type == "flat":
            self.index = faiss.IndexFlatIP(dim)   # inner product = cosine on unit vecs
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            nlist = 100                            # number of Voronoi cells
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown index_type: {index_type}")

        self.index_type = index_type
        self._trained = index_type == "flat"      # flat needs no training

    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """
        Add vectors and their metadata.
        embeddings: float32 (N, dim)
        metadata:   list of dicts (same length as embeddings)
        """
        import faiss  # type: ignore

        assert len(embeddings) == len(metadata), "Embeddings and metadata length mismatch"

        vecs = embeddings.astype("float32").copy()
        faiss.normalize_L2(vecs)                  # cosine similarity via inner product

        if not self._trained:
            print(f"  Training IVF index on {len(vecs)} vectors...")
            self.index.train(vecs)
            self._trained = True

        self.index.add(vecs)
        self.metadata.extend(metadata)

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 5,
        filter_fn=None,
    ) -> list[dict]:
        """
        Return top-k results sorted by score descending.
        filter_fn: optional callable(metadata_dict) -> bool to post-filter results.
        """
        import faiss  # type: ignore

        q = query_vec.reshape(1, -1).astype("float32")
        faiss.normalize_L2(q)

        # Over-fetch if filtering
        fetch_k = k * 3 if filter_fn else k
        scores, indices = self.index.search(q, min(fetch_k, len(self.metadata)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx]
            if filter_fn and not filter_fn(meta):
                continue
            results.append({**meta, "score": float(score)})
            if len(results) >= k:
                break

        return results

    def save(self, path: str) -> None:
        import faiss  # type: ignore
        # Import the FAISS library used to handle vector indices.
        # `type: ignore` tells static type checkers (like mypy) to ignore potential typing issues.

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Ensure that the directory where the index will be saved exists.
        # Path(path).parent gets the directory of the given path.
        # parents=True allows creating nested directories.
        # exist_ok=True avoids raising an error if the directory already exists.

        faiss.write_index(self.index, f"{path}.faiss")
        # Save the FAISS index object to disk.
        # The index contains all stored vectors and the search structure.
        # The file is saved with extension ".faiss".

        Path(f"{path}_meta.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False)
        )
        # Save the metadata associated with each vector.
        # Since FAISS stores only vectors, metadata must be saved separately.
        # Metadata is serialized into JSON format.
        # ensure_ascii=False preserves non-ASCII characters (important for multilingual text).

        print(f"Saved {len(self.metadata)} vectors to {path}.faiss")
        # Print a confirmation message showing how many vectors were saved
        # and where the FAISS index file is located.

    def load(self, path: str) -> None:
        import faiss  # type: ignore
        # Import the FAISS library again for loading the stored index.

        self.index = faiss.read_index(f"{path}.faiss")
        # Load the FAISS index from disk.
        # This restores the vector database structure and stored embeddings.

        self.metadata = json.loads(
            Path(f"{path}_meta.json").read_text()
        )
        # Load the metadata file corresponding to the vectors.
        # The JSON file is read and converted back into a Python list/dictionary structure.

        self._trained = True
        # Mark the index as trained.
        # This is important for certain FAISS index types (like IVF) which require training
        # before vectors can be added or searched.

        print(f"Loaded {len(self.metadata)} vectors from {path}.faiss")
        # Print a message confirming that the vectors and metadata
        # have been successfully loaded from disk.

    def __len__(self) -> int:
        return self.index.ntotal

    def stats(self) -> dict:
        return {
            "total_vectors": len(self),
            "dimension": self.dim,
            "index_type": self.index_type,
            "metadata_keys": list(self.metadata[0].keys()) if self.metadata else [],
        }


# ── Pinecone ───────────────────────────────────────────────────────────────────

class PineconeStore:
    """
    Managed Pinecone vector database.
    Supports metadata filtering and scales to billions of vectors.
    Requires PINECONE_API_KEY in environment.
    """

    def __init__(
        self,
        index_name: str = "financial-rag",
        dim: int = 1536,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        # Import the Pinecone client and serverless configuration class.
        from pinecone import Pinecone, ServerlessSpec  # type: ignore

        # Read the Pinecone API key from environment variables.
        api_key = os.getenv("PINECONE_API_KEY")

        # If the API key is not found, raise an error.
        # This prevents the program from running without proper authentication.
        if not api_key:
            raise EnvironmentError("PINECONE_API_KEY not set in environment")

        # Create a Pinecone client instance using the API key.
        self.pc = Pinecone(api_key=api_key)

        # Store the index name for later use.
        self.index_name = index_name

        # Store the vector dimension (must match the embedding model dimension).
        self.dim = dim

        # Retrieve all existing Pinecone indexes in the current account.
        # self.pc.list_indexes() returns objects representing indexes.
        # We extract their names into a list.
        existing = [i.name for i in self.pc.list_indexes()]

        # If the specified index does not exist, create it.
        if index_name not in existing:
            print(f"  Creating Pinecone index '{index_name}'...")

            # Create a new Pinecone index.
            self.pc.create_index(
                name=index_name,            # name of the index
                dimension=dim,              # vector embedding dimension
                metric=metric,              # similarity metric (cosine, dotproduct, euclidean)
                spec=ServerlessSpec(        # serverless deployment configuration
                    cloud=cloud,            # cloud provider (e.g., AWS)
                    region=region           # cloud region where index is hosted
                ),
            )

            # Wait until the index becomes ready before using it.
            # Pinecone may take a few seconds to finish provisioning resources.
            while not self.pc.describe_index(index_name).status["ready"]:
                time.sleep(1)

        # Connect to the Pinecone index so it can be used for vector operations
        # such as upsert (add vectors) and query (search).
        self.index = self.pc.Index(index_name)

    def add(
            self,
            embeddings: np.ndarray,
            metadata: list[dict],
            batch_size: int = 100,
            namespace: str = "",
    ) -> None:
        """
        Add (upsert) vectors into the Pinecone index.

        Parameters:
        embeddings : np.ndarray
            A 2D NumPy array of embedding vectors. Each row corresponds to one document embedding.

        metadata : list[dict]
            A list of metadata dictionaries associated with each embedding.
            Each metadata item usually contains fields like text, source, page, etc.

        batch_size : int
            Number of vectors uploaded in one API request. Uploading in batches improves
            efficiency and avoids sending extremely large requests.

        namespace : str
            Optional namespace inside the index. Namespaces allow separating different
            datasets within the same index (e.g., different companies or document types).
        """

        vectors = []
        # This list will store all vectors formatted according to Pinecone's required schema.

        for vec, meta in zip(embeddings, metadata):
            # Iterate through embeddings and metadata together.
            # zip() pairs each vector with its corresponding metadata.

            # Pinecone requires text to be stored in metadata under a specific key.
            # This block also removes metadata fields that Pinecone does not support.

            clean_meta = {
                k: v for k, v in meta.items()
                # Iterate through all metadata fields (key-value pairs).

                if isinstance(v, (str, int, float, bool, list))
                   # Pinecone only supports certain data types in metadata.
                   # Valid types: string, integer, float, boolean, or list.

                   and (not isinstance(v, str) or len(v) < 40_000)
                # If the value is a string, ensure it is not extremely large.
                # Pinecone limits metadata field sizes, so this prevents errors.
            }

            vectors.append({
                "id": str(uuid.uuid4()),
                # Generate a unique identifier for the vector.
                # uuid4() creates a random globally unique ID.

                "values": vec.tolist(),
                # Convert the NumPy embedding vector into a Python list,
                # because Pinecone's API expects standard JSON-compatible data.

                "metadata": clean_meta,
                # Attach the cleaned metadata to the vector.
            })

        print(f"  Upserting {len(vectors)} vectors to Pinecone...")
        # Print a message showing how many vectors will be uploaded.

        for i in range(0, len(vectors), batch_size):
            # Iterate through the vectors in batches.
            # Example:
            # if batch_size = 100 and vectors = 350
            # batches will be: [0-99], [100-199], [200-299], [300-349]

            self.index.upsert(
                vectors=vectors[i: i + batch_size],
                namespace=namespace,
            )
            # Upload the current batch of vectors to the Pinecone index.
            # "upsert" means:
            #   - insert if the vector ID does not exist
            #   - update if the vector ID already exists

            time.sleep(0.1)
            # Pause briefly to avoid hitting API rate limits.
            # Many cloud services limit how quickly requests can be sent.

    def search(
            self,
            query_vec: np.ndarray,
            k: int = 5,
            filter: dict | None = None,
            namespace: str = "",
    ) -> list[dict]:
        """
        Search the Pinecone index using a query embedding.

        Parameters
        ----------
        query_vec : np.ndarray
            The embedding vector representing the query text.
            This vector is compared against vectors stored in the index.

        k : int
            Number of top similar vectors to return (Top-K search).

        filter : dict | None
            Optional metadata filter used to restrict search results.

            Example filters:

            {"company": {"$in": ["AAPL", "MSFT"]}}
            → return results where company is AAPL or MSFT

            {"year": {"$gte": 2022}}
            → return results from year >= 2022

            {"$and": [{"company": "AAPL"}, {"year": {"$gte": 2022}}]}
            → return AAPL results from year >= 2022

        namespace : str
            Namespace inside the index used to separate datasets.
            If empty, the default namespace is used.

        Returns
        -------
        list[dict]
            A list of results where each result contains metadata,
            similarity score, and vector id.
        """

        kwargs = dict(
            vector=query_vec.tolist(),
            # Convert NumPy vector to Python list because the Pinecone API
            # expects JSON-compatible data.

            top_k=k,
            # Return the top-k most similar vectors.

            include_metadata=True,
            # Include stored metadata in the search results.

            namespace=namespace,
            # Limit the search to a specific namespace.
        )

        if filter:
            kwargs["filter"] = filter
            # If a metadata filter is provided, add it to the query arguments.
            # Pinecone will only return vectors whose metadata satisfies the filter.

        resp = self.index.query(**kwargs)
        # Execute the vector similarity search on the Pinecone index.
        # The **kwargs syntax expands the dictionary into keyword arguments.

        results = []
        # This list will store the formatted search results.

        for match in resp["matches"]:
            # Pinecone returns matches in the format:
            # resp = {
            #   "matches": [
            #       {
            #           "id": "...",
            #           "score": 0.92,
            #           "metadata": {...}
            #       }
            #   ]
            # }

            results.append({
                **match["metadata"],
                # Expand the metadata dictionary into the result dictionary.

                "score": match["score"],
                # Similarity score between query vector and stored vector.
                # Higher score usually means higher similarity.

                "id": match["id"],
                # Unique identifier of the matched vector.
            })

        return results
        # Return the formatted search results as a list of dictionaries.

    def delete_all(self, namespace: str = "") -> None:
        """Wipe all vectors (useful for re-indexing)."""
        self.index.delete(delete_all=True, namespace=namespace)
        print(f"  Deleted all vectors in namespace '{namespace}'")

    def save(self, path: str) -> None:
        print("  PineconeStore is cloud-managed — no local save needed.")

    def load(self, path: str) -> None:
        print("  PineconeStore is cloud-managed — no local load needed.")

    def __len__(self) -> int:
        stats = self.index.describe_index_stats()
        return stats.get("total_vector_count", 0)

    def stats(self) -> dict:
        return self.index.describe_index_stats()


# ── Hybrid search helper ───────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
    alpha: float = 0.5,
) -> list[dict]:
    """
    Combine dense retrieval results and sparse retrieval results using
    Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    dense_results : list[dict]
        Results returned from dense vector search (embedding similarity).

    sparse_results : list[dict]
        Results returned from sparse keyword search such as BM25.

    k : int
        RRF constant used to dampen the influence of rank.
        A larger k makes rank differences less significant.

    alpha : float
        Weight controlling the contribution of dense vs sparse retrieval.

        alpha = 1.0 → only dense results contribute
        alpha = 0.0 → only sparse results contribute
        alpha = 0.5 → equal contribution from both
    """

    scores: dict[str, float] = {}
    # Dictionary storing the accumulated RRF score for each document.

    docs: dict[str, dict] = {}
    # Dictionary storing the original document object for each key.

    for rank, doc in enumerate(dense_results):
        # Iterate through dense retrieval results.
        # enumerate() returns both the rank (index) and the document.

        key = doc.get("text", "")[:50]
        # Use the first 50 characters of the document text as a unique key.
        # This helps identify the same document appearing in multiple rankings.

        scores[key] = scores.get(key, 0) + alpha * (1 / (k + rank + 1))
        # Compute the RRF score contribution for this dense result.

        # RRF formula:
        # score = 1 / (k + rank)

        # Here:
        # - rank + 1 avoids division by zero
        # - alpha controls the dense retrieval weight
        # - scores.get(key, 0) ensures accumulation if document appears multiple times

        docs[key] = doc
        # Store the document object so it can be returned later.

    for rank, doc in enumerate(sparse_results):
        # Iterate through sparse retrieval results (e.g., BM25 keyword search).

        key = doc.get("text", "")[:50]
        # Generate the same key for matching documents across both rankings.

        scores[key] = scores.get(key, 0) + (1 - alpha) * (1 / (k + rank + 1))
        # Add the sparse retrieval contribution to the RRF score.

        # (1 - alpha) ensures dense + sparse weights sum to 1.

        docs[key] = doc
        # Store the document reference.

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    # Sort documents by their final RRF score (highest first).

    return [{**docs[k], "rrf_score": scores[k]} for k in sorted_keys]
    # Return a list of documents sorted by RRF score.
    # Each document dictionary is expanded (**docs[k])
    # and an additional field "rrf_score" is added.


# ── Latency benchmark ──────────────────────────────────────────────────────────

def benchmark_query_latency(
    store: FAISSStore | PineconeStore,
    query_vecs: np.ndarray,
    k: int = 5,
    n_queries: int = 50,
) -> dict:
    """
    Benchmark the latency of vector search queries.

    Parameters
    ----------
    store : FAISSStore | PineconeStore
        The vector store used for search. It can be either:
        - FAISSStore (local vector index)
        - PineconeStore (cloud vector database)

    query_vecs : np.ndarray
        A matrix of query embedding vectors.
        Each row represents one query vector.

    k : int
        Number of top results to retrieve in each search query.

    n_queries : int
        Number of queries used for benchmarking.
        If there are fewer vectors than n_queries, the function will use all available queries.

    Returns
    -------
    dict
        A dictionary containing latency statistics (in milliseconds):
        p50, p95, p99, mean, and total number of queries measured.
    """

    import statistics
    # Import Python's statistics module for computing median and mean values.

    latencies = []
    # List that stores latency (execution time) of each query in milliseconds.

    for i in range(min(n_queries, len(query_vecs))):
        # Run up to n_queries searches, but not more than the number of available query vectors.

        start = time.perf_counter()
        # Record the start time using a high-precision performance counter.
        # perf_counter() is preferred for benchmarking because it provides high-resolution timing.

        store.search(query_vecs[i], k=k)
        # Perform a vector similarity search using the query vector.
        # This calls the store's search function (either FAISS or Pinecone).

        latencies.append((time.perf_counter() - start) * 1000)
        # Calculate the elapsed time and convert it to milliseconds.
        # Store the latency in the latencies list.

    latencies.sort()
    # Sort latency values from smallest to largest.
    # Sorting is required for computing percentile statistics.

    return {
        "p50_ms":  round(statistics.median(latencies), 2),
        # p50 = median latency (50th percentile)
        # Half the queries are faster than this value.

        "p95_ms":  round(latencies[int(len(latencies) * 0.95)], 2),
        # p95 = 95th percentile latency
        # 95% of queries are faster than this value.

        "p99_ms":  round(latencies[int(len(latencies) * 0.99)], 2),
        # p99 = 99th percentile latency
        # Used to measure tail latency in production systems.

        "mean_ms": round(statistics.mean(latencies), 2),
        # Average latency across all queries.

        "n":       len(latencies),
        # Total number of queries measured.
    }


# ── Demo ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    print("=== FAISS demo ===")
    store = FAISSStore(dim=8, index_type="flat")

    # Fake embeddings for demonstration
    rng = np.random.default_rng(42)
    vecs = rng.random((20, 8)).astype("float32")
    meta = [{"text": f"document {i}", "company": "AAPL", "year": 2023} for i in range(20)]

    store.add(vecs, meta)
    print(f"  Index size: {len(store)}")
    print("  Stats:", store.stats())

    query = rng.random(8).astype("float32")
    results = store.search(query, k=3)
    for r in results:
        print(f"  score={r['score']:.4f} | {r['text']}")

    store.save("data/indices/test_index")
    store.load("data/indices/test_index")
    print(f"  Reloaded index size: {len(store)}")