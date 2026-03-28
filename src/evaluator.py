"""
src/evaluator.py
Retrieval and generation evaluation: RAGAS + classical IR metrics.
Week 8 core module.
"""
from __future__ import annotations
# Allows using list[str] instead of List[str] in type hints (Python 3.9 backport)

import json
# Used to read eval_dataset.json and write ragas_results.json

import statistics
# Provides statistics.mean() — used to average scores across queries

from pathlib import Path
# File path handling — Path("tests/eval_dataset.json").read_text()

import numpy as np
# Used in ndcg_at_k() for np.log2() in the DCG formula


# ── Classical IR metrics ───────────────────────────────────────────────────────

def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    # Slice the top-k results from the full retrieved list
    top_k = retrieved_ids[:k]

    # Count how many of the top-k are actually relevant
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)

    # hits / k = fraction of retrieved docs that were useful
    # Guard against k=0 to avoid ZeroDivisionError
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    # Edge case: if there are no relevant docs, recall is undefined → return 0
    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)

    # hits / total_relevant = fraction of ALL relevant docs we found
    # Denominator is the total number of relevant docs, not k
    return hits / len(relevant_ids)


def average_precision(retrieved_ids: list, relevant_ids: set) -> float:
    """Average precision (AP) for a single query."""
    if not relevant_ids:
        return 0.0
    hits = 0
    precisions = []
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            hits += 1
            precisions.append(hits / rank)
    return statistics.mean(precisions) if precisions else 0.0


def mean_average_precision(results: list[tuple[list, set]]) -> float:
    """MAP across multiple queries."""
    aps = [average_precision(retrieved, relevant) for retrieved, relevant in results]
    return statistics.mean(aps) if aps else 0.0


def mrr(results: list[tuple[list, set]]) -> float:
    """Mean Reciprocal Rank: average of 1/rank of first relevant result."""
    rr_scores = []
    for retrieved, relevant in results:
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                rr_scores.append(1 / rank)
                break
        else:
            rr_scores.append(0.0)
    return statistics.mean(rr_scores) if rr_scores else 0.0


def ndcg_at_k(
    retrieved_ids: list,
    relevance_scores: dict[str, float],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain with graded relevance.
    relevance_scores: {doc_id: score} where higher = more relevant.
    """
    gains = [relevance_scores.get(doc_id, 0.0) for doc_id in retrieved_ids[:k]]
    dcg   = sum(g / np.log2(i + 2) for i, g in enumerate(gains))

    ideal = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg  = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    retriever_fn,
    test_cases: list[dict],
    k_values: list[int] = [1, 3, 5],
) -> dict:
    """
    Evaluate a retriever function across multiple k values.

    Args:
        retriever_fn: callable(query: str) -> list[str]  (returns doc ids)
        test_cases:   list of {"query": str, "relevant_ids": list[str]}

    Returns:
        {k: {precision, recall, map, mrr, ndcg}}
    """
    all_results = []
    for tc in test_cases:
        retrieved = retriever_fn(tc["query"])
        relevant  = set(tc["relevant_ids"])
        all_results.append((retrieved, relevant))

    metrics = {}
    for k in k_values:
        precisions = [precision_at_k(r, rel, k) for r, rel in all_results]
        recalls    = [recall_at_k(r, rel, k)    for r, rel in all_results]
        metrics[f"k={k}"] = {
            f"precision@{k}": round(statistics.mean(precisions), 4),
            f"recall@{k}":    round(statistics.mean(recalls), 4),
        }

    metrics["map"]  = round(mean_average_precision(all_results), 4)
    metrics["mrr"]  = round(mrr(all_results), 4)
    return metrics


# ── RAGAS evaluation ───────────────────────────────────────────────────────────

def run_ragas(
    rag_chain,
    eval_dataset_path: str = "tests/eval_dataset.json",
    output_path: str = "tests/ragas_results.json",
) -> dict:
    """
    Run RAGAS evaluation suite on a FinancialRAGChain.

    eval_dataset.json format:
        [{"question": str, "ground_truth": str}, ...]

    Requires: pip install ragas datasets
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    # Load eval questions
    eval_data_raw = json.loads(Path(eval_dataset_path).read_text())
    print(f"Running RAGAS on {len(eval_data_raw)} questions...")

    rows = []
    for item in eval_data_raw:
        question = item["question"]
        ground_truth = item["ground_truth"]

        # Run RAG pipeline
        result = rag_chain.ask(question)
        retrieved_chunks = rag_chain.retrieve_only(question)

        rows.append({
            "question":   question,
            "answer":     result["answer"],
            "contexts":   [c["text"] for c in retrieved_chunks],
            "ground_truth": ground_truth,
        })

    dataset = Dataset.from_list(rows)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    df = result.to_pandas()
    summary = df[
        ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    ].mean().to_dict()

    # Save full results
    Path(output_path).parent.mkdir(exist_ok=True)
    df.to_json(output_path, orient="records", indent=2)
    print(f"\nRAGAS summary (mean over {len(rows)} questions):")
    for metric, score in summary.items():
        bar = "█" * int(score * 20)
        print(f"  {metric:<22} {score:.4f}  {bar}")
    print(f"\nFull results saved to {output_path}")

    return summary


# ── Chunking strategy comparison ───────────────────────────────────────────────

def compare_chunking_strategies(
    text: str,
    queries: list[dict],  # [{"query": str, "relevant_snippets": list[str]}]
    strategies: list[str] = ["fixed", "recursive", "token"],
) -> dict:
    """
    Build one index per chunking strategy and compare retrieval quality.
    Returns a report dict for notebook visualization.
    """
    import sys
    sys.path.insert(0, "src")
    from chunker import chunk_document
    from embedder import get_embedder
    from vectorstore import FAISSStore

    embedder = get_embedder("openai-small")
    report = {}

    for strategy in strategies:
        print(f"\n[{strategy}] chunking...")
        chunks = chunk_document(text, strategy=strategy)
        texts  = [c.text for c in chunks]

        vecs = embedder(texts)
        store = FAISSStore(dim=vecs.shape[1])
        store.add(vecs, [{"text": t} for t in texts])

        precision_scores = []
        for qitem in queries:
            q_vec = embedder([qitem["query"]])[0]
            results = store.search(q_vec, k=5)
            retrieved_texts = [r["text"] for r in results]

            # Compute precision: did any relevant snippet appear in top-5?
            hits = sum(
                1 for snippet in qitem["relevant_snippets"]
                if any(snippet[:50] in rt for rt in retrieved_texts)
            )
            precision_scores.append(hits / max(len(qitem["relevant_snippets"]), 1))

        report[strategy] = {
            "num_chunks":    len(chunks),
            "avg_chunk_len": round(sum(len(c.text) for c in chunks) / len(chunks)),
            "precision@5":   round(statistics.mean(precision_scores), 4),
        }
        print(f"  {len(chunks)} chunks, avg {report[strategy]['avg_chunk_len']} chars, P@5={report[strategy]['precision@5']:.3f}")

    return report


# ── Sample eval dataset ────────────────────────────────────────────────────────

SAMPLE_EVAL_DATASET = [
    {
        "question": "What was Apple's total net sales in fiscal year 2023?",
        "ground_truth": "Apple's total net sales in fiscal year 2023 were $383.3 billion."
    },
    {
        "question": "What were Apple's main product revenue segments?",
        "ground_truth": "Apple's revenue segments include iPhone, Mac, iPad, Wearables/Home/Accessories, and Services."
    },
    {
        "question": "What risk factors related to competition did Apple highlight?",
        "ground_truth": "Apple highlighted intense competition in all markets where it operates, with competitors having substantial resources and aggressive pricing strategies."
    },
    {
        "question": "What was Microsoft's total revenue in fiscal year 2023?",
        "ground_truth": "Microsoft reported total revenue of $211.9 billion for fiscal year 2023."
    },
    {
        "question": "How did Microsoft's Intelligent Cloud segment perform?",
        "ground_truth": "Microsoft's Intelligent Cloud segment revenue was $87.9 billion, growing 19% year-over-year, with Azure growing 29%."
    },
    {
        "question": "What was Nvidia's data center revenue growth?",
        "ground_truth": "Nvidia's data center revenue grew significantly, driven by AI and machine learning demand."
    },
    {
        "question": "What were Apple's gross margins in 2023?",
        "ground_truth": "Apple's gross margin was 44.1% in fiscal year 2023, up from 43.3% in 2022."
    },
    {
        "question": "What liquidity risks did Apple disclose?",
        "ground_truth": "Apple disclosed risks related to its ability to generate sufficient cash flows to meet obligations, and dependency on continued access to capital markets."
    },
]


def create_eval_dataset(output_path: str = "tests/eval_dataset.json") -> None:
    """Write the sample eval dataset to disk."""
    Path(output_path).parent.mkdir(exist_ok=True)
    Path(output_path).write_text(json.dumps(SAMPLE_EVAL_DATASET, indent=2))
    print(f"Saved {len(SAMPLE_EVAL_DATASET)} eval questions to {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Demo: classical metrics on dummy data
    print("=== Classical Metrics Demo ===")

    # Fake retrieval scenario: 5 relevant docs among 10
    retrieved = ["doc_1", "doc_3", "doc_5", "doc_2", "doc_8"]
    relevant  = {"doc_1", "doc_3", "doc_7"}

    for k in [1, 3, 5]:
        p = precision_at_k(retrieved, relevant, k)
        r = recall_at_k(retrieved, relevant, k)
        print(f"  P@{k}={p:.3f}  R@{k}={r:.3f}")

    # MRR across 3 queries
    queries_data = [
        (["doc_3", "doc_1", "doc_5"], {"doc_1", "doc_3"}),  # first relevant at rank 1
        (["doc_8", "doc_2", "doc_1"], {"doc_1"}),            # first relevant at rank 3
        (["doc_8", "doc_9", "doc_6"], {"doc_7"}),            # no relevant in top-3
    ]
    print(f"\n  MRR = {mrr(queries_data):.3f}")
    print(f"  MAP = {mean_average_precision(queries_data):.3f}")

    # NDCG with graded relevance
    graded = {"doc_1": 3.0, "doc_3": 2.0, "doc_5": 1.0}
    print(f"  NDCG@5 = {ndcg_at_k(retrieved, graded, 5):.3f}")

    create_eval_dataset()