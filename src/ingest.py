"""
src/ingest.py
Download and parse SEC 10-K filings from EDGAR.
Fixes:
  - Year parsed from filing date inside document, not folder name
  - HTML entities decoded (&#8217; → ', &#160; → space, etc.)
  - HTML tags stripped before chunking
"""
import html
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

TICKERS      = ["AAPL", "MSFT", "NVDA"]
FILING_TYPES = ["10-K"]
DATA_DIR     = Path("data/sec_filings")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_filings(tickers: list[str] = TICKERS, num_filings: int = 3) -> None:
    """Download recent 10-K filings for each ticker."""
    from sec_edgar_downloader import Downloader
    dl = Downloader("MyCompany", "myemail@example.com", DATA_DIR)
    for ticker in tickers:
        print(f"Downloading 10-K for {ticker}...")
        dl.get("10-K", ticker, limit=num_filings)
    print("Download complete.")


def strip_html(text: str) -> str:
    """Remove all HTML tags, decode entities, normalize whitespace."""
    # 1. Decode HTML entities first: &#8217; → ' , &#160; → space
    text = html.unescape(text)
    # 2. Remove <style>...</style> blocks
    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    # 3. Remove <script>...</script> blocks
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    # 4. Replace block-level tags with newlines
    text = re.sub(r'<(?:div|p|br|tr|li|h[1-6])[^>]*>', '\n', text, flags=re.IGNORECASE)
    # 5. Strip all remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # 6. Collapse excessive whitespace
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_year_from_text(text: str) -> int:
    """
    Try to extract the fiscal year from filing text.
    Looks for patterns like 'fiscal year 2023', 'September 28, 2024', etc.
    """
    # Pattern: "fiscal year ended ... 20XX"
    m = re.search(r'fiscal year (?:ended?|ending).*?(20\d{2})', text[:5000], re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Pattern: "for the year ended September XX, 20XX"
    m = re.search(r'for the year ended[^.]*?(20\d{2})', text[:5000], re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Pattern: "Annual Report ... 20XX"
    m = re.search(r'annual report[^.]*?(20\d{2})', text[:3000], re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Pattern: standalone 4-digit year near top of document
    matches = re.findall(r'\b(20\d{2})\b', text[:2000])
    if matches:
        # Return most common year near top
        from collections import Counter
        return int(Counter(matches).most_common(1)[0][0])

    return 0


def extract_text_from_pdf(pdf_path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            pages.append(t)
    return "\n\n".join(pages)


def extract_text_from_txt(txt_path: Path) -> str:
    return txt_path.read_text(encoding="utf-8", errors="ignore")


def clean_text(text: str) -> str:
    """Full cleaning pipeline: strip HTML → normalize whitespace."""
    # Strip HTML tags and decode entities
    text = strip_html(text)
    # Remove lone page numbers
    text = re.sub(r'(?m)^\s*\d+\s*$', '', text)
    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def parse_sections(text: str, ticker: str, year: int) -> list[dict]:
    """
    Split 10-K into major sections based on Item headings.
    Returns list of {company, ticker, year, section, text}.
    """
    section_patterns = {
        "Business":             r"(?i)item\s+1\b[^aA]",
        "Risk Factors":         r"(?i)item\s+1[aA]\b",
        "Properties":           r"(?i)item\s+2\b",
        "Legal":                r"(?i)item\s+3\b",
        "MDA":                  r"(?i)item\s+7\b[^aA]",
        "Quantitative Risk":    r"(?i)item\s+7[aA]\b",
        "Financial Statements": r"(?i)item\s+8\b",
        "Controls":             r"(?i)item\s+9\b",
    }

    positions = {}
    for name, pattern in section_patterns.items():
        match = re.search(pattern, text)
        if match:
            positions[name] = match.start()

    sorted_sections = sorted(positions.items(), key=lambda x: x[1])
    chunks = []

    for i, (name, start) in enumerate(sorted_sections):
        end = sorted_sections[i + 1][1] if i + 1 < len(sorted_sections) else len(text)
        section_text = text[start:end].strip()
        if len(section_text) > 100:
            chunks.append({
                "company": ticker,
                "ticker":  ticker,
                "year":    year,
                "section": name,
                "text":    section_text,
            })

    if not chunks:
        chunks.append({
            "company": ticker,
            "ticker":  ticker,
            "year":    year,
            "section": "Full Filing",
            "text":    text,
        })

    return chunks


def load_all_filings() -> list[dict]:
    """Walk downloaded filings and return all parsed sections."""
    all_sections = []

    for ticker in TICKERS:
        ticker_dir = DATA_DIR / "sec-edgar-filings" / ticker / "10-K"
        if not ticker_dir.exists():
            print(f"  [!] No filings for {ticker}")
            continue

        for filing_dir in sorted(ticker_dir.iterdir()):
            # Find primary document
            doc = None
            for ext in ["*.htm", "*.html", "*.txt"]:
                files = list(filing_dir.glob(ext))
                if files:
                    # Prefer the largest file (usually the main filing)
                    doc = max(files, key=lambda f: f.stat().st_size)
                    break
            if doc is None:
                pdfs = list(filing_dir.glob("*.pdf"))
                if pdfs:
                    doc = pdfs[0]
            if doc is None:
                continue

            print(f"  Parsing {ticker} from {doc.name}...")

            if doc.suffix == ".pdf":
                raw = extract_text_from_pdf(doc)
            else:
                raw = extract_text_from_txt(doc)

            # Extract year from document content (more reliable than folder name)
            year = extract_year_from_text(raw)

            cleaned = clean_text(raw)
            sections = parse_sections(cleaned, ticker, year)
            all_sections.extend(sections)
            print(f"    → year={year}, {len(sections)} sections, {len(cleaned):,} chars")

    print(f"\nTotal sections: {len(all_sections)}")
    return all_sections


if __name__ == "__main__":
    download_filings()
    sections = load_all_filings()
    for s in sections[:5]:
        print(f"{s['company']} {s['year']} — {s['section']}: {len(s['text'])} chars")
        print(f"  Preview: {s['text'][:100]!r}")