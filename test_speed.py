"""
test_speed.py
Benchmark: embedding + retrieval + LLM response time on CPU.
Run: python test_speed.py
"""
import time
import sys
sys.path.insert(0, "src")

print("=" * 50)
print("Speed Benchmark — Local RAG on CPU")
print("=" * 50)

# ── Test 1: MiniLM Embedding ──────────────────────
print("\n[1/3] Testing MiniLM embedding speed...")
t0 = time.time()

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

test_texts = [
    "What was Apple's total revenue in fiscal year 2023?",
    "Apple's total net sales for fiscal year 2023 were $383.3 billion.",
    "Microsoft reported total revenue of $211.9 billion for fiscal year 2023.",
]

vecs = model.encode(test_texts)
t1 = time.time()
print(f"  ✓ Embedded {len(test_texts)} texts in {t1 - t0:.2f}s")
print(f"  Vector shape: {vecs.shape}")

# ── Test 2: FAISS Retrieval ───────────────────────
print("\n[2/3] Testing FAISS retrieval speed...")
from pathlib import Path

INDEX_PATH = "data/indices/financial_index"
if Path(f"{INDEX_PATH}.faiss").exists():
    t0 = time.time()
    import faiss, json
    import numpy as np

    index = faiss.read_index(f"{INDEX_PATH}.faiss")
    with open(f"{INDEX_PATH}_meta.json") as f:
        meta = json.load(f)

    query_vec = model.encode(["What was Apple's revenue?"])
    D, I = index.search(query_vec.astype("float32"), k=5)
    t1 = time.time()
    print(f"  ✓ Retrieved top-5 from {index.ntotal} vectors in {t1 - t0:.3f}s")
    for i, idx in enumerate(I[0]):
        m = meta[idx]
        print(f"    [{i+1}] {m.get('company','?')} {m.get('year','?')} — {m.get('section','?')}")
else:
    print("  ⚠️  Index not found, skipping FAISS test")

# ── Test 3: Ollama LLM ────────────────────────────
print("\n[3/3] Testing Ollama LLM speed...")
try:
    import requests
    t0 = time.time()

    resp = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": "In one sentence, what is Apple's main business?",
            "stream": False,
        },
        timeout=120,
    )

    t1 = time.time()
    result = resp.json()
    print(f"  ✓ LLM responded in {t1 - t0:.1f}s")
    print(f"  Response: {result.get('response', '')[:200]}")
    eval_count = result.get('eval_count', 0)
    eval_duration = result.get('eval_duration', 1) / 1e9
    if eval_count and eval_duration:
        print(f"  Speed: {eval_count / eval_duration:.1f} tokens/sec")

except requests.exceptions.ConnectionError:
    print("  ❌ Ollama not running — start it with: ollama serve")
except Exception as e:
    print(f"  ❌ Error: {e}")

print("\n" + "=" * 50)
print("Benchmark complete!")
print("=" * 50)