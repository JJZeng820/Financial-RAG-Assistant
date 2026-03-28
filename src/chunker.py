"""
src/chunker.py
Five chunking strategies for financial documents.
Week 7 core module.
"""
# Module description:
# This file defines the core data structures and types used for document chunking.
# Chunking is the process of splitting large documents into smaller pieces (chunks)
# so they can be embedded and retrieved efficiently in a RAG (Retrieval-Augmented Generation) pipeline.

from __future__ import annotations
# Enables postponed evaluation of type annotations.
# Instead of evaluating type hints immediately, Python stores them as strings
# and resolves them later. This helps:
# - avoid circular import issues
# - improve performance for type checking
# - allow referencing classes before they are defined

import re
# Import Python's regular expression module.
# This is commonly used in chunking tasks for:
# - sentence splitting
# - text cleaning
# - pattern matching
# Example: splitting text by punctuation marks.

from dataclasses import dataclass, field
# dataclass:
# Automatically generates methods such as:
# - __init__()
# - __repr__()
# - __eq__()
# which simplifies the creation of simple data containers.

# field:
# Allows customization of dataclass attributes,
# for example specifying default values or factories.

from typing import Literal
# Literal is a typing tool that restricts a variable
# to a fixed set of allowed values.

# Example:
# mode: Literal["fast", "slow"]
# mode can only be "fast" or "slow"


StrategyName = Literal[
    "fixed", "recursive", "token", "semantic", "sentence_window"
]
# Define a type alias for the allowed chunking strategies.
# This restricts the chunking method name to one of the following strings:
#
# "fixed"           -> fixed-length chunking
# "recursive"       -> recursive text splitting
# "token"           -> token-based chunking
# "semantic"        -> semantic similarity based chunking
# "sentence_window" -> sentence window chunking
#
# Using Literal improves:
# - IDE autocomplete
# - static type checking
# - code readability


@dataclass
class Chunk:
    """
    A single chunk of text extracted from a larger document.

    Each chunk contains:
    - the text content
    - optional metadata describing the source or context
    """

    text: str
    # The textual content of the chunk.
    # Example:
    # "Apple reported strong revenue growth in Q4."

    metadata: dict = field(default_factory=dict)
    # Metadata associated with this chunk.
    # Example metadata:
    # {
    #   "company": "AAPL",
    #   "year": 2023,
    #   "source": "10-K"
    # }
    #
    # default_factory=dict ensures that each Chunk instance
    # gets its own independent dictionary.
    #
    # Without default_factory, all Chunk objects might share
    # the same dictionary, which would cause bugs.

    def __len__(self):
        """
        Return the length of the chunk text.

        This allows using the built-in len() function on a Chunk object.
        Example:
            chunk = Chunk("Hello world")
            len(chunk) -> 11
        """
        return len(self.text)
        # Return the number of characters in the chunk text.


# ── Strategy dispatcher ────────────────────────────────────────────────────────

def chunk_document(
        text: str,
        strategy: StrategyName = "recursive",
        metadata: dict | None = None,
        **kwargs,
) -> list[Chunk]:
    """
    Split text into chunks using the chosen strategy.

    Args:
        text:     raw document text to be split into smaller pieces
        strategy: chunking strategy name, must be one of:
                  'fixed', 'recursive', 'token', 'semantic', 'sentence_window'
                  default is "recursive"
        metadata: optional metadata dictionary attached to every chunk
                  (e.g., {"company": "AAPL", "year": 2023})
        **kwargs: additional parameters passed to the specific chunking strategy
                  (e.g., chunk_size, overlap, window_size)

    Returns:
        list[Chunk]: a list of Chunk objects containing chunk text and metadata
    """

    # Ensure metadata is always a dictionary.
    # If metadata is None, use an empty dictionary instead.
    meta = metadata or {}

    # Select the corresponding chunking function based on the chosen strategy.
    # This dictionary maps strategy names to their internal implementation functions.
    fn = {
        "fixed": _fixed_size,  # fixed-length chunking
        "recursive": _recursive,  # recursive hierarchical splitting
        "token": _token_based,  # token-count-based splitting
        "semantic": _semantic,  # semantic similarity based chunking
        "sentence_window": _sentence_window,  # sliding window over sentences
    }[strategy]
    # fn becomes the selected chunking function.
    # Example:
    # strategy = "recursive"
    # fn = _recursive
    #
    # The function can then be called like:
    # raw_chunks = fn(text, **kwargs)


    # Execute the selected chunking function.
    # It returns a list of raw text chunks (strings).
    # Additional keyword arguments (**kwargs) are passed to the strategy function.
    raw_chunks = fn(text, **kwargs)

    # Convert raw text chunks into Chunk objects.
    # Each chunk keeps its text and inherits the provided metadata,
    # while also adding the chunking strategy used.
    return [
        Chunk(
            text=c,
            metadata={**meta, "strategy": strategy}
        )
        for c in raw_chunks
    ]


# ── Strategy 1: Fixed-size ─────────────────────────────────────────────────────

def _fixed_size(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """Split on character count with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap# Move the start pointer forward, but keep an overlap between consecutive chunks Without overlap, semantic breaks may occur between chunks.
    return [c for c in chunks if c.strip()]


# ── Strategy 2: Recursive character splitter ──────────────────────────────────

def _recursive(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """
    LangChain-style recursive splitter.
    Tries to split on paragraph, then newline, then sentence, then word.
    """

    # Import the RecursiveCharacterTextSplitter from LangChain.
    # This is a commonly used text splitter in RAG pipelines because it
    # tries to preserve semantic structure when splitting documents.
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Create a text splitter instance with the desired parameters.
    splitter = RecursiveCharacterTextSplitter(

        # Maximum size of each chunk (measured in characters).
        # The splitter will try to keep chunks under this limit.
        chunk_size=chunk_size,

        # Number of characters shared between adjacent chunks.
        # This overlap helps preserve context across chunk boundaries.
        chunk_overlap=overlap,

        # Ordered list of separators used for recursive splitting.
        # The splitter will try them one by one in this order:
        separators=[
            "\n\n",  # first try splitting by paragraph (two newlines)
            "\n",    # if still too large, split by single newline
            ". ",    # then try sentence boundary
            " ",     # then try splitting by word
            ""       # finally fallback to character-level split
        ],
    )

    # Apply the splitter to the input text.
    # The function returns a list of text chunks (strings).
    return splitter.split_text(text)


# ── Strategy 3: Token-based ────────────────────────────────────────────────────

def _token_based(
    text: str,
    chunk_size: int = 256,
    overlap: int = 32,
) -> list[str]:
    """
    Split by token count (respects LLM context limits).
    Uses tiktoken cl100k_base (GPT-4 tokenizer).
    """
    from langchain_text_splitters import TokenTextSplitter

    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        encoding_name="cl100k_base",
    )
    return splitter.split_text(text)


# ── Strategy 4: Semantic chunking ─────────────────────────────────────────────

def _semantic(
    text: str,
    breakpoint_threshold: int = 90,
    **kwargs,   # ✅ ADD THIS
) -> list[str]:
    """
    Split text based on semantic changes between sentences.

    This method computes embeddings for sentences and measures the similarity
    between consecutive sentences. When the similarity drops significantly,
    it assumes the topic has changed and creates a new chunk.

    Args:
        text: full document text to split
        breakpoint_threshold: percentile threshold (0–100) used to decide when
                              a similarity drop is large enough to trigger a split.
                              Example: 90 means only the top 10% largest similarity
                              drops will create chunk boundaries.

    Returns:
        list[str]: list of semantically coherent text chunks
    """

    # Import LangChain's experimental semantic text splitter.
    # This splitter groups sentences into chunks based on embedding similarity.
    from langchain_experimental.text_splitter import SemanticChunker

    # Import OpenAI embedding model wrapper used to compute sentence embeddings.
    from langchain_openai import OpenAIEmbeddings

    # Initialize the embedding model.
    # "text-embedding-3-small" is a lightweight and inexpensive embedding model
    # suitable for semantic similarity tasks.
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # Create a semantic chunker instance.
    splitter = SemanticChunker(

        # Embedding model used to compute sentence vectors.
        embeddings,

        # Method used to determine the threshold for splitting.
        # "percentile" means the threshold is computed from the distribution
        # of similarity drops between sentences.
        breakpoint_threshold_type="percentile",

        # The percentile value used as the splitting threshold.
        # Example:
        # 90 → only the largest 10% similarity drops will trigger splits.
        breakpoint_threshold_amount=breakpoint_threshold,
    )

    # Apply the semantic splitter to the text.
    # The splitter will:
    # 1. break text into sentences
    # 2. compute embeddings
    # 3. detect large similarity drops
    # 4. create chunks accordingly
    return splitter.split_text(text)


# ── Strategy 5: Sentence window ───────────────────────────────────────────────

def _sentence_window(
    text: str,
    window_size: int = 2,
) -> list[str]:
    """
    Sentence-window chunking strategy.

    Each chunk is centered on a single sentence but also includes its
    surrounding sentences as context (a sliding window). This helps
    preserve local semantic context for retrieval.

    Args:
        text: full document text to split
        window_size: number of sentences to include before and after
                     the central sentence

    Returns:
        list[str]: list of windowed sentence chunks
    """

    # Split the text into sentences using a regex that detects
    # punctuation marks (. ! ?) followed by whitespace.
    # Example:
    # "Hello world. How are you?" → ["Hello world.", "How are you?"]
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Clean each sentence by stripping whitespace and filtering out very short sentences (less than 20 characters).
    # This helps remove noise like headings or fragments.
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    # List that will store the final sentence windows.
    windows = []

    # Iterate through each sentence with its index.
    for i, sent in enumerate(sentences):

        # Determine the start of the context window.
        # Ensure it does not go below index 0.
        start = max(0, i - window_size)

        # Determine the end of the context window.
        # Ensure it does not exceed the total number of sentences.
        end = min(len(sentences), i + window_size + 1)

        # Combine the sentences in the window into one chunk.
        # Example with window_size=2:
        # [sent_{i-2}, sent_{i-1}, sent_i, sent_{i+1}, sent_{i+2}]
        window_text = " ".join(sentences[start:end])

        # Add the window chunk to the result list.
        windows.append(window_text)

    # Return all generated sentence windows.
    return windows


# ── Benchmark helper ───────────────────────────────────────────────────────────

def benchmark_strategies(
    text: str,
    strategies: list[StrategyName] | None = None,
) -> dict[str, dict]:
    """
    Run multiple chunking strategies on the same text and compute
    summary statistics for comparison.

    Args:
        text: input document text
        strategies: list of strategies to evaluate.
                    If None, default strategies are used.

    Returns:
        dict[str, dict]:
        Example:
        {
            "fixed": {
                "count": 20,
                "avg_len": 480,
                "min_len": 350,
                "max_len": 512,
                "stdev": 40
            }
        }
    """

    # Import statistics module for computing mean and standard deviation
    import statistics

    # If no strategies are specified, use a default subset.
    # Semantic and sentence_window are skipped because they are slower
    # (semantic requires embedding calls).
    if strategies is None:
        strategies = ["fixed", "recursive", "token"]

    # Dictionary to store benchmark results for each strategy
    results = {}

    # Iterate through each chunking strategy
    for strategy in strategies:
        try:
            # Run the chunking function using the selected strategy
            chunks = chunk_document(text, strategy=strategy)

            # Compute the length of each chunk
            # len(c.text) counts the number of characters in the chunk
            lengths = [len(c.text) for c in chunks]

            # Store summary statistics for the current strategy
            results[strategy] = {

                # Total number of chunks produced
                "count": len(chunks),

                # Average chunk length
                "avg_len": round(statistics.mean(lengths)),

                # Minimum chunk length
                "min_len": min(lengths),

                # Maximum chunk length
                "max_len": max(lengths),

                # Standard deviation of chunk lengths
                # (0 if only one chunk exists)
                "stdev": round(statistics.stdev(lengths)) if len(lengths) > 1 else 0,
            }

        # Catch errors (for example missing libraries or model failures)
        except Exception as e:

            # Record the error message instead of stats
            results[strategy] = {"error": str(e)}

    # Print formatted table header for comparison
    print(f"\n{'Strategy':<18} {'Count':>6} {'Avg':>6} {'Min':>6} {'Max':>6} {'Stdev':>6}")

    # Print a separator line
    print("-" * 54)

    # Print results for each strategy
    for name, stats in results.items():

        # If the strategy failed, print the error
        if "error" in stats:
            print(f"{name:<18}  ERROR: {stats['error']}")

        # Otherwise print the computed statistics
        else:
            print(
                f"{name:<18} {stats['count']:>6} {stats['avg_len']:>6} "
                f"{stats['min_len']:>6} {stats['max_len']:>6} {stats['stdev']:>6}"
            )

    # Return the full results dictionary for further analysis
    return results


# ── Demo ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SAMPLE_TEXT = """
Apple Inc. designs, manufactures, and markets smartphones, personal computers,
tablets, wearables, and accessories worldwide.

The Company's business strategy leverages its unique ability to design and develop
its own operating systems, hardware, application software, and services.

Risk Factors: The Company's business, reputation, results of operations,
financial condition, and stock price can be affected by a number of factors,
whether currently known or unknown, including those described below.

Competition: The markets for the Company's products and services are highly
competitive, and are characterized by aggressive price cutting, with resulted
in lower gross margins.
"""

    meta = {"company": "AAPL", "year": 2023, "section": "sample"}

    for strategy in ["fixed", "recursive", "token"]:
        chunks = chunk_document(SAMPLE_TEXT, strategy=strategy, metadata=meta)
        print(f"\n[{strategy}] {len(chunks)} chunks")
        for i, c in enumerate(chunks):
            print(f"  {i+1}. ({len(c.text):4d} chars) {c.text[:80].strip()!r}")

    benchmark_strategies(SAMPLE_TEXT)