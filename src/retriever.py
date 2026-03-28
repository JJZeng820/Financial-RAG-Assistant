"""
src/retriever.py
Retrieval layer: dense search, MMR deduplication, optional reranking.
"""

from __future__ import annotations
# Postpone evaluation of type hints (improves compatibility and avoids circular imports)

import numpy as np
# NumPy is used for vector math (cosine similarity, dot products, norms)

from embedder import get_embedder
# Factory function that returns the embedding model

from vectorstore import FAISSStore, PineconeStore
# Two supported vector databases:
# FAISS   → local vector index
# Pinecone → managed cloud vector database


class FinancialRetriever:
    """
    High-level retrieval interface on top of a vector store.

    Supports:
        - Basic dense vector search
        - MMR (Maximal Marginal Relevance) to reduce redundant results
        - Optional metadata filtering (ticker, year, section)
    """

    def __init__(
        self,
        store: FAISSStore | PineconeStore,
        embed_model: str = "openai-small",
    ):
        # Vector database instance (FAISS or Pinecone)
        self.store = store

        # Initialize the embedding model
        self.embedder = get_embedder(embed_model)

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = 5,
        method: str = "mmr",
        filter: dict | None = None,
    ) -> list[dict]:
        """
        Retrieve top-k chunks for a query.

        Args:
            query:  natural-language question
            k:      number of results to return
            method: 'dense' — pure cosine similarity search
                    'mmr'   — Maximal Marginal Relevance (reduces duplicate results)
            filter: metadata filter dictionary

        Returns:
            list of result dictionaries with 'text', 'score', and metadata
        """

        # Convert the query string into a vector embedding
        query_vec = self.embedder([query])[0]

        # Choose retrieval strategy
        if method == "dense":
            return self._dense_search(query_vec, k, filter)

        elif method == "mmr":
            return self._mmr_search(query_vec, k, filter)

        else:
            # Raise error if unsupported method is used
            raise ValueError(f"Unknown method: {method!r}. Use 'dense' or 'mmr'.")

    def retrieve_multi(
        self,
        queries: list[str],
        k: int = 5,
        method: str = "mmr",
        deduplicate: bool = True,
    ) -> list[dict]:
        """
        Retrieve documents for multiple queries and merge results.

        Useful for:
            - multi-hop questions
            - query decomposition
        """

        # Track already seen documents to avoid duplicates
        seen_texts: set[str] = set()

        # Store merged results
        all_results: list[dict] = []

        # Run retrieval for each query separately
        for query in queries:

            results = self.retrieve(query, k=k, method=method)

            for r in results:

                # Use the first 80 characters as a deduplication key
                key = r["text"][:80]

                # Skip duplicates if enabled
                if deduplicate and key in seen_texts:
                    continue

                seen_texts.add(key)
                all_results.append(r)

        # Sort results by score (descending)
        return sorted(
            all_results,
            key=lambda x: x.get("score", 0),
            reverse=True
        )[:k * len(queries)]

    # ── Internal methods ───────────────────────────────────────────────────────

    def _dense_search(
        self,
        query_vec: np.ndarray,
        k: int,
        filter: dict | None,
    ) -> list[dict]:
        """
        Perform basic dense vector search.
        """

        # Pinecone supports filtering directly inside the search API
        if isinstance(self.store, PineconeStore):
            return self.store.search(query_vec, k=k, filter=filter)

        # FAISS does not support metadata filtering natively
        # So we create a Python filter function
        filter_fn = _build_faiss_filter(filter) if filter else None

        return self.store.search(query_vec, k=k, filter_fn=filter_fn)


    def _mmr_search(
        self,
        query_vec: np.ndarray,
        k: int,
        filter: dict | None,
        fetch_k: int | None = None,
        lambda_mult: float = 0.6,
    ) -> list[dict]:
        """
        Maximal Marginal Relevance retrieval.

        Balances:
            relevance to the query
            diversity among retrieved chunks

        lambda_mult:
            1.0 → pure relevance
            0.0 → pure diversity
        """

        # Determine candidate pool size
        if fetch_k is None:
            fetch_k = min(k * 4, len(self.store) or k * 4)

        # 1️⃣ Fetch a larger candidate pool using dense search
        candidates = self._dense_search(query_vec, k=fetch_k, filter=filter)

        if not candidates:
            return []

        # 2️⃣ Re-embed candidate texts
        cand_texts = [c["text"] for c in candidates]
        cand_vecs = self.embedder(cand_texts)

        # 3️⃣ MMR selection loop
        selected_indices: list[int] = []

        # Remaining candidate indices
        remaining = list(range(len(candidates)))

        while len(selected_indices) < k and remaining:

            # First document: choose highest relevance to query
            if not selected_indices:

                scores = [
                    float(
                        np.dot(cand_vecs[i], query_vec)
                        / (
                            np.linalg.norm(cand_vecs[i])
                            * np.linalg.norm(query_vec)
                            + 1e-9
                        )
                    )
                    for i in remaining
                ]

                best = remaining[int(np.argmax(scores))]

            else:
                # Later selections balance relevance and diversity
                mmr_scores = []

                for i in remaining:

                    # Relevance score
                    rel = float(
                        np.dot(cand_vecs[i], query_vec)
                        / (
                            np.linalg.norm(cand_vecs[i])
                            * np.linalg.norm(query_vec)
                            + 1e-9
                        )
                    )

                    # Redundancy score (similarity to already selected docs)
                    redundancy = max(
                        float(
                            np.dot(cand_vecs[i], cand_vecs[j])
                            / (
                                np.linalg.norm(cand_vecs[i])
                                * np.linalg.norm(cand_vecs[j])
                                + 1e-9
                            )
                        )
                        for j in selected_indices
                    )

                    # MMR formula
                    mmr_scores.append(
                        lambda_mult * rel - (1 - lambda_mult) * redundancy
                    )

                best = remaining[int(np.argmax(mmr_scores))]

            # Move best candidate to selected list
            selected_indices.append(best)
            remaining.remove(best)

        # Return selected documents
        return [candidates[i] for i in selected_indices]


# ── Filter builder ─────────────────────────────────────────────────────────────

def _build_faiss_filter(filter_dict: dict):
    """
    Convert a Pinecone-style filter dictionary into a Python filtering function for FAISS search results.
    """

    def match(meta: dict) -> bool:

        for key, condition in filter_dict.items():

            # Logical AND
            if key == "$and":
                return all(_build_faiss_filter(c)(meta) for c in condition)

            # Logical OR
            if key == "$or":
                return any(_build_faiss_filter(c)(meta) for c in condition)

            value = meta.get(key)

            # Operator-based conditions
            if isinstance(condition, dict):

                for op, operand in condition.items():

                    if op == "$eq" and not (value == operand):
                        return False

                    if op == "$ne" and not (value != operand):
                        return False

                    if op == "$gt" and not (value > operand):
                        return False

                    if op == "$gte" and not (value >= operand):
                        return False

                    if op == "$lt" and not (value < operand):
                        return False

                    if op == "$lte" and not (value <= operand):
                        return False

                    if op == "$in" and value not in operand:
                        return False

                    if op == "$nin" and value in operand:
                        return False

            else:
                # Direct equality condition
                if value != condition:
                    return False

        return True

    return match


# ── Demo / test script ─────────────────────────────────────────────────────────

if __name__ == "__main__":

    import numpy as np

    # Create a small FAISS vector store
    store = FAISSStore(dim=8)

    # Random vectors for testing
    rng = np.random.default_rng(0)
    vecs = rng.random((30, 8)).astype("float32")

    # Metadata for each document
    meta = [
        {
            "text": f"This is document {i} about {'Apple' if i % 2 == 0 else 'Microsoft'}.",
            "company": "AAPL" if i % 2 == 0 else "MSFT",
            "year": 2022 + (i % 3),
        }
        for i in range(30)
    ]

    # Add vectors to the index
    store.add(vecs, meta)

    # Dummy embedder to avoid calling external APIs
    class DummyEmbedder:

        def __call__(self, texts):
            return (
                np.random.default_rng(hash(texts[0]) % 2**31)
                .random((len(texts), 8))
                .astype("float32")
            )

    # Create retriever
    retriever = FinancialRetriever(store, embed_model="openai-small")

    # Replace embedder with dummy
    retriever.embedder = DummyEmbedder()

    print("=== Dense retrieval ===")

    results = retriever.retrieve(
        "Apple revenue risk factors",
        k=3,
        method="dense",
        filter={"company": "AAPL"},
    )

    for r in results:
        print(f"  [{r['score']:.4f}] {r['company']} {r['year']} — {r['text']}")

    print("\n=== MMR retrieval ===")

    results = retriever.retrieve(
        "Apple revenue risk factors",
        k=3,
        method="mmr",
    )

    for r in results:
        print(f"  [{r.get('score', 0):.4f}] {r['company']} {r['year']} — {r['text']}")