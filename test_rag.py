"""
test_rag.py
Interactive terminal interface for Financial RAG with tool calling.
Run: TRANSFORMERS_OFFLINE=1 python test_rag.py
     TRANSFORMERS_OFFLINE=1 python test_rag.py --auto
"""
from __future__ import annotations

import sys
import time
import argparse
sys.path.insert(0, "src")

parser = argparse.ArgumentParser()
parser.add_argument("--auto",       action="store_true")
parser.add_argument("--index-path", default="data/indices/financial_index")
parser.add_argument("--k",          type=int, default=3)
args = parser.parse_args()

AUTO_QUESTIONS = [
    "What was Apple's total revenue in fiscal year 2023?",
    "Calculate Apple's revenue growth: 2022 was $394.3B, 2023 was $383.3B.",
    "Compare Apple and Microsoft gross margins.",
    "What risk factors did Apple highlight in their 10-K?",
]

print("\n" + "=" * 60)
print("  Financial RAG — Tool Calling Mode")
print("=" * 60)
print(f"\nLoading index: {args.index_path}")

t0 = time.time()
from chain import FinancialRAGChain
rag = FinancialRAGChain(args.index_path, k=args.k)
print(f"✓ Ready in {time.time() - t0:.1f}s\n")


def ask(question: str):
    print(f"\n{'─'*60}")
    print(f"Q: {question}")
    print(f"{'─'*60}")

    result = rag.ask(question)

    # Show tool calls
    if result.get("tool_log"):
        print(f"\n🔧 Tools called ({len(result['tool_log'])}):")
        for t in result["tool_log"]:
            args_str = str(t["args"])[:60]
            result_str = t["result"][:100].replace("\n", " ")
            print(f"  • {t['tool']}({args_str})")
            print(f"    → {result_str}...")

    print(f"\n💬 Answer:\n{result['answer']}")
    print(f"\n⏱  {result['elapsed']}s total")


if args.auto:
    print("Running preset questions...\n")
    for q in AUTO_QUESTIONS:
        ask(q)
    print(f"\n{'='*60}\nDone.")
    sys.exit(0)

# Interactive mode
print("Type your question and press Enter. Type 'quit' to exit.\n")
print("Example questions:")
for q in AUTO_QUESTIONS:
    print(f"  • {q}")
print()

while True:
    try:
        question = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break
    if not question:
        continue
    if question.lower() in ("quit", "exit", "q"):
        print("Bye!")
        break
    ask(question)