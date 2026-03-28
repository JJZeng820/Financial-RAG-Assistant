"""
src/chain.py
RAG chain with tool calling.
Handles llama3.2 bug where it returns raw JSON instead of tool_calls.
"""
from __future__ import annotations

import sys
import json
import re
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "llama3.2"

# ── Tool definitions ───────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_filings",
            "description": (
                "Search SEC 10-K filings for financial data about a company. "
                "Use for revenue, margins, risk factors, segments, and any factual filing data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":   {"type": "string",  "description": "Search query e.g. 'Apple total revenue 2023'"},
                    "company": {"type": "string",  "description": "Ticker: AAPL, MSFT, NVDA, or any"},
                    "k":       {"type": "integer", "description": "Number of results, default 3"},
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform financial calculations: growth rates, margins, ratios.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["growth_rate", "margin", "ratio", "difference", "average"]},
                    "value_a":   {"type": "number", "description": "First value"},
                    "value_b":   {"type": "number", "description": "Second value"},
                    "label":     {"type": "string", "description": "Label for the result"},
                },
                "required": ["operation", "value_a", "value_b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_companies",
            "description": "Compare two companies on a financial metric side by side.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_a": {"type": "string", "description": "First ticker e.g. AAPL"},
                    "company_b": {"type": "string", "description": "Second ticker e.g. MSFT"},
                    "metric":    {"type": "string", "description": "Metric to compare e.g. gross margin"},
                },
                "required": ["company_a", "company_b", "metric"]
            }
        }
    },
]


# ── Tool implementations ───────────────────────────────────────

def _tool_search_filings(store, embed_model, query: str, company: str = "any", k: int = 3) -> str:
    vec = embed_model.encode([query], normalize_embeddings=True)[0]
    filter_fn = None
    if company and company not in ("any", ""):
        filter_fn = lambda m: m.get("ticker") == company or m.get("company") == company
    results = store.search(vec, k=int(k), filter_fn=filter_fn)
    if not results:
        return "No relevant filings found."
    parts = []
    for i, r in enumerate(results, 1):
        label = f"{r.get('company','?')} {r.get('year','?')} — {r.get('section','?')}"
        text  = r.get("text", "")[:400]
        score = r.get("score", 0)
        parts.append(f"[Result {i}] {label} (relevance={score:.3f})\n{text}")
    return "\n\n".join(parts)


def _to_float(x) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    return float(str(x).replace(",", "").replace("$", "").replace("B", "").strip())


def _tool_calculate(operation: str, value_a, value_b, label: str = "") -> str:
    try:
        a, b = _to_float(value_a), _to_float(value_b)
        if operation == "growth_rate":
            r = ((a - b) / abs(b)) * 100
            return f"{label or 'Growth rate'}: {r:+.2f}% (from {b} to {a})"
        elif operation == "margin":
            return f"{label or 'Margin'}: {(a/b)*100:.2f}%"
        elif operation == "ratio":
            return f"{label or 'Ratio'}: {a/b:.4f}"
        elif operation == "difference":
            return f"{label or 'Difference'}: {a-b:+.2f}"
        elif operation == "average":
            return f"{label or 'Average'}: {(a+b)/2:.2f}"
        else:
            return f"Unknown operation: {operation}"
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Calculation error: {e}"


def _tool_compare_companies(store, embed_model, company_a: str, company_b: str, metric: str) -> str:
    vec = embed_model.encode([metric], normalize_embeddings=True)[0]
    ra  = store.search(vec, k=3, filter_fn=lambda m: m.get("ticker") == company_a)
    rb  = store.search(vec, k=3, filter_fn=lambda m: m.get("ticker") == company_b)
    def fmt(results, ticker):
        if not results:
            return f"{ticker}: No data found."
        return f"{ticker}:\n" + "\n".join(
            f"  [{r.get('section','?')} {r.get('year','?')}] {r.get('text','')[:250]}"
            for r in results
        )
    return f"Comparison — {metric}:\n\n{fmt(ra, company_a)}\n\n{fmt(rb, company_b)}"


# ── llama3.2 raw-JSON fallback ─────────────────────────────────

def _extract_json_tool_call(text: str) -> dict | None:
    """
    llama3.2 sometimes returns raw JSON like:
        {"name": "search_filings", "parameters": {...}}
    instead of using the tool_calls field.
    This function detects and parses that.
    """
    text = text.strip()
    # Try to find JSON block in the response
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        if "name" in data and data["name"] in ("search_filings", "calculate", "compare_companies"):
            return data
    except Exception:
        pass
    return None


def _flatten_args(fn_name: str, raw_args: dict) -> dict:
    """
    Normalize tool arguments — handles cases where LLM returns
    schema objects instead of plain values:
        {"query": {"type": "string", "description": "..."}}
    vs correct:
        {"query": "Apple revenue 2023"}
    """
    clean = {}
    for k, v in raw_args.items():
        # Skip schema-only keys with no real value
        if isinstance(v, dict):
            # Try to find a real value inside
            if "value" in v:
                clean[k] = v["value"]
            elif "description" in v and len(v) == 1:
                continue  # pure schema, skip
            elif "enum" in v:
                # e.g. {"type": "any", "enum": ["AAPL"]} → take first enum
                enums = v.get("enum", [])
                clean[k] = enums[0] if enums else "any"
            else:
                clean[k] = str(list(v.values())[0])
        else:
            clean[k] = v

    # Defaults
    if fn_name == "search_filings":
        clean.setdefault("company", "any")
        clean.setdefault("k", 3)

    return clean


def _dispatch_tool(store, embed_model, fn_name: str, fn_args: dict) -> str:
    fn_args = _flatten_args(fn_name, fn_args)
    print(f"  🔧 {fn_name}({json.dumps(fn_args, ensure_ascii=False)[:100]})")
    if fn_name == "search_filings":
        return _tool_search_filings(store, embed_model, **fn_args)
    elif fn_name == "calculate":
        return _tool_calculate(**fn_args)
    elif fn_name == "compare_companies":
        return _tool_compare_companies(store, embed_model, **fn_args)
    else:
        return f"Unknown tool: {fn_name}"


# ── Agentic tool loop ──────────────────────────────────────────

def _run_tool_loop(store, embed_model, question: str, max_rounds: int = 6) -> tuple[str, list]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior financial analyst with access to SEC filing tools.\n"
                "Rules:\n"
                "- ALWAYS call search_filings first before answering any factual question.\n"
                "- Use calculate for any numeric computation.\n"
                "- Use compare_companies for side-by-side company comparisons.\n"
                "- After getting tool results, write a clear answer citing [Source N].\n"
                "- Never return raw JSON — always use tool_calls."
            )
        },
        {"role": "user", "content": question}
    ]

    tool_log = []

    for round_num in range(max_rounds):
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":   MODEL,
                "messages": messages,
                "tools":   TOOLS,
                "stream":  False,
                "options": {"temperature": 0.0},
            },
            timeout=120,
        )
        resp.raise_for_status()
        data    = resp.json()
        msg     = data.get("message", {})
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls", [])

        # ── Case 1: proper tool_calls returned ────────────────
        if tool_calls:
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            })
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"].get("arguments", {})
                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except Exception:
                        fn_args = {}
                result = _dispatch_tool(store, embed_model, fn_name, fn_args)
                tool_log.append({"tool": fn_name, "args": fn_args, "result": result[:300]})
                messages.append({"role": "tool", "content": result})
            continue

        # ── Case 2: llama3.2 bug — raw JSON in content ────────
        parsed = _extract_json_tool_call(content)
        if parsed:
            fn_name = parsed["name"]
            fn_args = parsed.get("parameters", parsed.get("arguments", {}))
            result  = _dispatch_tool(store, embed_model, fn_name, fn_args)
            tool_log.append({"tool": fn_name, "args": fn_args, "result": result[:300]})
            # Inject as if tool was called properly
            messages.append({"role": "assistant", "content": ""})
            messages.append({"role": "tool", "content": result})
            # Ask LLM to now answer based on the tool result
            messages.append({
                "role": "user",
                "content": "Now answer the original question using the tool results above. Do not call any more tools."
            })
            continue

        # ── Case 3: plain text final answer ───────────────────
        return content, tool_log

    return "Max rounds reached.", tool_log


# ── RAG Chain ──────────────────────────────────────────────────

class FinancialRAGChain:

    def __init__(
        self,
        index_path: str,
        embed_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama3.2",
        k: int = 3,
        temperature: float = 0.0,
    ):
        global MODEL
        MODEL = llm_model
        self.k = k

        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {embed_model}")
        self.embed_model = SentenceTransformer(embed_model)

        faiss_file = f"{index_path}.faiss"
        if not Path(faiss_file).exists():
            raise FileNotFoundError(f"Index not found: {faiss_file}")

        sys.path.insert(0, "src")
        from vectorstore import FAISSStore
        dim = self.embed_model.get_sentence_embedding_dimension()
        self.store = FAISSStore(dim=dim)
        self.store.load(index_path)
        print(f"Loaded {len(self.store)} vectors from {faiss_file}")

    def ask(self, question: str) -> dict:
        t0 = time.time()
        answer, tool_log = _run_tool_loop(self.store, self.embed_model, question)
        return {
            "question": question,
            "answer":   answer,
            "sources":  [e for e in tool_log if e["tool"] == "search_filings"],
            "tool_log": tool_log,
            "elapsed":  round(time.time() - t0, 1),
        }

    def retrieve_only(self, question: str, k: int | None = None) -> list[dict]:
        vec = self.embed_model.encode([question], normalize_embeddings=True)[0]
        return self.store.search(vec, k=k or self.k)


if __name__ == "__main__":
    INDEX_PATH = "data/indices/financial_index"
    rag = FinancialRAGChain(INDEX_PATH)
    for q in [
        "What was Apple's total revenue in FY2023?",
        "Calculate Apple's revenue growth: 2022 was $394.3B, 2023 was $383.3B.",
        "Compare Apple and Microsoft gross margins.",
    ]:
        print(f"\n{'='*60}\nQ: {q}")
        r = rag.ask(q)
        print(f"\nA: {r['answer']}")
        print("\n🔧 Tools used:")
        for t in r["tool_log"]:
            print(f"  • {t['tool']} → {t['result'][:80]}...")
        print(f"⏱  {r['elapsed']}s")