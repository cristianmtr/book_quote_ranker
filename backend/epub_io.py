from __future__ import annotations

import uuid
import warnings
from pathlib import Path

import ebooklib
import markdown as md_lib
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


def markdown_to_epub(title: str, markdown_text: str, output_path: Path) -> Path:
    """Assemble a single-chapter EPUB from Markdown text (used for the merged
    Samples output)."""
    html_body = md_lib.markdown(markdown_text, extensions=["extra"])

    book = epub.EpubBook()
    book.set_identifier(str(uuid.uuid4()))
    book.set_title(title)
    book.set_language("en")

    chapter = epub.EpubHtml(title=title, file_name="content.xhtml", lang="en")
    chapter.content = f"<html><body>{html_body}</body></html>"
    book.add_item(chapter)

    book.toc = [epub.Link("content.xhtml", title, "content")]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(output_path), book)
    return output_path
