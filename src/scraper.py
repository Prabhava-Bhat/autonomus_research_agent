import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document


# Tags that reliably carry article / document content.
# Targeting these instead of calling soup.get_text() avoids ingesting menus,
# cookie banners, footer legal text, and sidebar ads.
CONTENT_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "blockquote"]

# Tags whose entire subtree (including text) should be discarded before
# content extraction.
NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]


class WebScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }

    def scrape_url(self, url: str) -> Document | None:
        """Scrape a URL and return a clean Document object.

        Changes vs. original:
        - Noise tags (`nav`, `footer`, `header`, `aside`, `form`, `noscript`,
          `script`, `style`) are removed before any text extraction.
        - Content is assembled from semantic tags only (p, headings, li …)
          rather than dumping the entire DOM with get_text(). This produces
          significantly cleaner chunks and better retrieval quality.
        """
        try:
            print(f"Scraping URL: {url}")
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # 1. Remove noisy structural elements entirely.
            for tag in soup(NOISE_TAGS):
                tag.decompose()

            # 2. Extract text only from meaningful content tags.
            content_elements = soup.find_all(CONTENT_TAGS)
            lines = [el.get_text(separator=" ", strip=True) for el in content_elements]
            # Drop empty strings and very short fragments (e.g. stray bullets).
            lines = [line for line in lines if len(line) > 20]
            text = "\n".join(lines)

            if not text.strip():
                # Fallback: if targeted extraction yields nothing (e.g. JS-heavy
                # pages where content is injected at runtime), take the full body.
                text = soup.get_text(separator=" ", strip=True)

            title = soup.title.string.strip() if soup.title and soup.title.string else "Unknown Title"

            return Document(
                page_content=text,
                metadata={
                    "source": url,
                    "title": title,
                    "type": "web_scrape",
                },
            )

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None