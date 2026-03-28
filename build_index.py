"""
build_index.py
One-shot script: download filings → chunk → embed → save FAISS index.
Run this once before starting the app.

Usage:
    python build_index.py                     # full run
    python build_index.py --demo              # tiny demo (no API for filings)
    python build_index.py --strategy semantic # choose chunking strategy
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")


# ── Argument parsing ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Build Financial RAG index")
parser.add_argument("--demo",     action="store_true",  help="Use synthetic demo data (no downloads)")
parser.add_argument("--strategy", default="recursive",  help="Chunking strategy: fixed|recursive|token|semantic")
parser.add_argument("--tickers",  nargs="+", default=["AAPL", "MSFT", "NVDA"])
parser.add_argument("--chunk-size",  type=int, default=512)
parser.add_argument("--overlap",     type=int, default=64)
parser.add_argument("--embed-model", default="minilm")
parser.add_argument("--index-path",  default="data/indices/financial_index")
args = parser.parse_args()


# ── Demo data (used when --demo flag is set) ───────────────────────────────────

DEMO_SECTIONS = [
    {"company": "AAPL", "ticker": "AAPL", "year": 2023, "section": "Business",
     "text": "Apple designs, manufactures, and markets smartphones, personal computers, tablets, wearables, "
             "and accessories, and sells a variety of related services. The Company's products include iPhone, "
             "Mac, iPad, Apple Watch, AirPods, Apple TV, HomePod, Beats products and accessories."},
    {"company": "AAPL", "ticker": "AAPL", "year": 2023, "section": "Financial Statements",
     "text": "Apple's total net sales for fiscal year 2023 were $383.3 billion, compared to $394.3 billion "
             "in fiscal year 2022, a decrease of approximately 3 percent. iPhone net sales were $200.6 billion. "
             "Services net sales were $85.2 billion, growing 9 percent year over year."},
    {"company": "AAPL", "ticker": "AAPL", "year": 2023, "section": "MDA",
     "text": "Apple's gross margin was 44.1 percent in fiscal year 2023, compared to 43.3 percent in 2022. "
             "Products gross margin was 36.6 percent. Services gross margin was 70.8 percent, "
             "reflecting the high-margin nature of the software and services business."},
    {"company": "AAPL", "ticker": "AAPL", "year": 2023, "section": "Risk Factors",
     "text": "Global and regional economic conditions could materially adversely affect Apple's business. "
             "The Company is subject to intense competition in all markets in which it operates. "
             "Competitors have substantially greater financial, technical, and human resources. "
             "There can be no assurance that Apple will be able to continue to compete effectively."},
    {"company": "MSFT", "ticker": "MSFT", "year": 2023, "section": "Financial Statements",
     "text": "Microsoft reported total revenue of $211.9 billion for fiscal year 2023, an increase of "
             "7 percent compared to the prior year. Operating income was $88.5 billion. "
             "Diluted earnings per share was $9.81, compared with $9.65 in the prior year."},
    {"company": "MSFT", "ticker": "MSFT", "year": 2023, "section": "Business",
     "text": "Microsoft's Intelligent Cloud segment revenue was $87.9 billion, growing 19 percent. "
             "Azure and other cloud services revenue grew 29 percent. The Productivity and Business "
             "Processes segment revenue was $69.3 billion, increasing 9 percent."},
    {"company": "MSFT", "ticker": "MSFT", "year": 2023, "section": "MDA",
     "text": "Microsoft's gross margin increased to 69 percent in fiscal year 2023, up from 68 percent "
             "in 2022. The improvement reflects growth in the higher-margin cloud services business. "
             "Operating margin was 42 percent for the year."},
    {"company": "NVDA", "ticker": "NVDA", "year": 2023, "section": "Business",
     "text": "Nvidia's data center revenue for fiscal year 2024 was $47.5 billion, up from $15.0 billion "
             "in fiscal year 2023. Revenue growth was driven by strong demand for AI infrastructure "
             "and the H100 GPU architecture. Gaming revenue was $10.4 billion."},
    {"company": "NVDA", "ticker": "NVDA", "year": 2023, "section": "Risk Factors",
     "text": "Nvidia faces intense competition from AMD, Intel, and other semiconductor companies. "
             "The Company's business is subject to rapid technological change. Significant concentration "
             "of revenue from a small number of customers increases business risk. "
             "Export control regulations may restrict sales to certain markets."},
]


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    start_time = time.time()
    index_path = args.index_path

    print("=" * 60)
    print("Financial RAG — Index Builder")
    print(f"  Strategy:    {args.strategy}")
    print(f"  Chunk size:  {args.chunk_size} chars / {args.overlap} overlap")
    print(f"  Embed model: {args.embed_model}")
    print(f"  Output:      {index_path}")
    print("=" * 60)

    # 1. Load sections
    if args.demo:
        print("\n[1/4] Using demo data (--demo flag set)")
        sections = [s for s in DEMO_SECTIONS if s["ticker"] in args.tickers]
    else:
        from ingest import download_filings, load_all_filings
        print("\n[1/4] Downloading filings...")
        download_filings(args.tickers)
        print("\n[2/4] Parsing filings...")
        sections = load_all_filings()

    print(f"  {len(sections)} sections loaded")

    # 2. Chunk each section
    print(f"\n[2/4] Chunking with strategy='{args.strategy}'...")
    from chunker import chunk_document

    all_chunks = []
    for section in sections:
        chunks = chunk_document(
            section["text"],
            strategy=args.strategy,
            metadata={
                "company": section["company"],
                "ticker":  section["ticker"],
                "year":    section["year"],
                "section": section["section"],
            },
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        all_chunks.extend(chunks)

    print(f"  {len(all_chunks)} chunks created")
    avg_len = sum(len(c.text) for c in all_chunks) / max(len(all_chunks), 1)
    print(f"  Average chunk length: {avg_len:.0f} chars")

    # 3. Embed
    print(f"\n[3/4] Embedding {len(all_chunks)} chunks with {args.embed_model}...")
    from embedder import get_embedder

    embedder = get_embedder(args.embed_model)
    texts = [c.text for c in all_chunks]
    embeddings = embedder(texts)
    print(f"  Embedding shape: {embeddings.shape}")

    # 4. Build and save index
    print(f"\n[4/4] Building FAISS index...")
    from vectorstore import FAISSStore

    store = FAISSStore(dim=embeddings.shape[1])
    meta_list = [
        {**c.metadata, "text": c.text}
        for c in all_chunks
    ]
    store.add(embeddings, meta_list)
    store.save(index_path)

    elapsed = time.time() - start_time
    print(f"\n✓ Done in {elapsed:.1f}s — {len(store)} vectors indexed")
    print(f"  Index: {index_path}.faiss")
    print(f"  Meta:  {index_path}_meta.json")
    print("\nNext step: python src/app.py")


if __name__ == "__main__":
    main()
