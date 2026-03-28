"""
src/app.py
Gradio chat interface for the Financial RAG Assistant.
Run: python src/app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

INDEX_PATH = "data/indices/financial_index"

# Lazy-load the chain so Gradio starts fast
_rag = None

def get_rag():
    global _rag
    if _rag is None:
        from chain import FinancialRAGChain
        _rag = FinancialRAGChain(INDEX_PATH)
    return _rag


# ── Response function ──────────────────────────────────────────────────────────

def answer(question: str, history: list) -> str:
    """Main chat callback."""
    if not question.strip():
        return "Please enter a question."

    try:
        rag = get_rag()
        result = rag.ask(question)

        # Format sources block
        sources_lines = []
        for i, s in enumerate(result["sources"], 1):
            sources_lines.append(
                f"**[{i}]** {s['company']} {s['year']} — *{s['section']}*\n"
                f"> {s['snippet'][:120]}…"
            )
        sources_block = "\n\n".join(sources_lines)

        return f"{result['answer']}\n\n---\n**📎 Retrieved Sources:**\n\n{sources_block}"

    except FileNotFoundError:
        return (
            "⚠️ Index not found. Run the following first:\n"
            "```bash\n"
            "python build_index.py --demo\n"
            "```"
        )
    except Exception as e:
        return f"❌ Error: {e}"


# ── Build Gradio UI ────────────────────────────────────────────────────────────

EXAMPLE_QUESTIONS = [
    "What was Apple's total revenue in fiscal year 2023?",
    "How did Microsoft's Azure cloud revenue grow in FY2023?",
    "What risk factors did Nvidia highlight in their latest 10-K?",
    "Compare Apple and Microsoft gross margins in 2023.",
    "What were Apple's main product revenue segments?",
    "How does Apple describe its competition risk?",
]

DESCRIPTION = """
## 📊 Financial RAG Assistant
**Phase 2 Project** — Ask questions grounded in SEC filings (10-K, 10-Q)

**Indexed companies:** Apple (AAPL) · Microsoft (MSFT) · Nvidia (NVDA)
**Data sources:** 10-K annual reports · Earnings call transcripts

Each answer cites the exact source document and section.
"""

with gr.Blocks(title="Financial RAG Assistant") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Financial Q&A",
                height=500,
            )
            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="Ask about SEC filings, revenue, risks, segments...",
                    show_label=False,
                    scale=5,
                )
                submit_btn = gr.Button("Ask →", variant="primary", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### 💡 Example Questions")
            for ex in EXAMPLE_QUESTIONS:
                gr.Button(ex, size="sm").click(
                    fn=lambda q=ex: q,
                    outputs=question_box,
                )

            gr.Markdown("### ⚙️ Settings")
            with gr.Accordion("Advanced", open=False):
                k_slider = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Retrieved chunks (k)",
                )
                gr.Markdown(
                    "Higher k = more context, slower response.\n"
                    "For financial Q&A, k=5 is usually optimal."
                )

    # Event handlers — tuple format (user, bot) for Gradio 6.10
    def chat(question, history):
        if not question.strip():
            return history, ""

        response = answer(question, history)
        history = history or []
        history.append((question, response))
        return history, ""

    submit_btn.click(
        chat,
        [question_box, chatbot],
        [chatbot, question_box],
    )

    question_box.submit(
        chat,
        [question_box, chatbot],
        [chatbot, question_box],
    )

    gr.Markdown(
        "---\n*Answers are grounded in indexed SEC filings. "
        "This is a learning project — not financial advice.*"
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_error=True,
    )