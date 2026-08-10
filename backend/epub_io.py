from __future__ import annotations

import warnings
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ebooklib import epub
from markdownify import markdownify as html_to_markdown

# EPUB chapter documents are XHTML (often with an <?xml?> prolog); we deliberately
# parse them with an HTML parser for lenient, namespace-agnostic content extraction.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def epub_to_markdown(path: Path) -> str:
    """Convert an EPUB's chapter content to a single Markdown string, in spine
    (reading) order."""
    book = epub.read_epub(str(path))
    parts = []
    for item_id, _linear in book.spine:
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_content(), "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        body = soup.body or soup
        text_md = html_to_markdown(str(body), heading_style="ATX", strip=["img"]).strip()
        if text_md:
            parts.append(text_md)
    return "\n\n".join(parts)
