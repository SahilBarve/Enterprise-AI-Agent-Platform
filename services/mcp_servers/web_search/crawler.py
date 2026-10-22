"""Web page crawler with SSRF protection, robots.txt politeness, and HTML extraction (FR-WEB-2, FR-WEB-3, FR-WEB-8).

================================================================================
EDUCATIONAL ARCHITECTURE NOTES:
Why Secure Web Crawling is Non-Trivial in Enterprise Multi-Agent Systems
--------------------------------------------------------------------------------
1. Defense-in-Depth against Server-Side Request Forgery (SSRF):
   - Traditional naive HTTP clients fetch whatever URL an agent requests.
   - An adversary (or a confused agent) can probe internal VPC hosts (`10.0.0.1`,
     `172.16.0.1`), Kubernetes API servers (`10.96.0.1:443`), or cloud metadata
     (`169.254.169.254`).
   - We validate URLs using DNS resolution and IP address range checks before
     ever opening a socket connection.

2. Web Politeness & robots.txt Compliance (FR-WEB-2):
   - Web crawlers must respect `robots.txt` disallow rules and identify themselves
     with a responsible `User-Agent`.
   - We cache parsed `robots.txt` rules per domain origin to minimize redundant
     requests.

3. Content Distillation (HTML -> Clean Markdown/Text) (FR-WEB-3):
   - Modern web pages are 80-90% noise: navigation headers, footers, cookie banners,
     tracking scripts, and styling.
   - Feeding raw HTML to an LLM burns 10x-50x more tokens and causes hallucination.
   - We strip boilerplate tags (<script>, <style>, <nav>, <footer>) and extract the
     semantic text body (<article>, <main>, or <p> tags).

4. Indirect Prompt Injection Isolation (FR-WEB-8):
   - Third-party web pages cannot be trusted.
   - All extracted text is sanitized and wrapped inside `<untrusted_web_content>`
     boundary tags so the downstream LLM treats it as informational data, not commands.
================================================================================
"""

import urllib.robotparser
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from libs.common.logging import get_logger
from services.mcp_servers.web_search.models import WebPageFetchResponse
from services.mcp_servers.web_search.security import is_safe_url, sanitize_web_content

logger = get_logger("mcp.web_search.crawler")

# Standard bot identification header
USER_AGENT = "EnterpriseAIBot/1.0 (+https://enterprise.ai/bot)"


class WebCrawler:
    """Safe, polite web crawler for fetching and extracting webpage content."""

    def __init__(
        self,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        respect_robots: bool = True,
        timeout_seconds: float = 10.0,
    ) -> None:
        """Initialize the crawler with security policies.

        Args:
            allowed_domains: Optional domain allowlist.
            blocked_domains: Optional domain denylist.
            respect_robots: Whether to check and honor robots.txt rules.
            timeout_seconds: Default request timeout in seconds.
        """
        self.allowed_domains = allowed_domains
        self.blocked_domains = blocked_domains
        self.respect_robots = respect_robots
        self.timeout_seconds = timeout_seconds
        self._robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}

    def fetch_page(self, url: str) -> WebPageFetchResponse:
        """Safely fetch and distill a target webpage.

        Args:
            url: Target HTTP/HTTPS URL.

        Returns:
            WebPageFetchResponse with extracted title, text, and security status.
        """
        # 1. SSRF and Domain Security Validation (FR-WEB-2)
        is_safe, reason = is_safe_url(
            url,
            allowed_domains=self.allowed_domains,
            blocked_domains=self.blocked_domains,
        )
        if not is_safe:
            logger.warning("Web fetch rejected by security policy", url=url, reason=reason)
            return WebPageFetchResponse(
                url=url,
                title="Access Blocked by Security Policy",
                content=f"Security check failed: {reason}",
                raw_length=len(reason),
                is_safe=False,
                metadata={"blocked_reason": reason},
            )

        # 2. robots.txt Politeness Check (FR-WEB-2)
        if self.respect_robots and not self._is_allowed_by_robots(url):
            logger.info("Web fetch skipped due to robots.txt disallow", url=url)
            return WebPageFetchResponse(
                url=url,
                title="Disallowed by robots.txt",
                content="Target URL is disallowed by the website's robots.txt policy.",
                raw_length=0,
                is_safe=True,
                metadata={"robots_disallowed": True},
            )

        # 3. HTTP Request Execution with Politeness Headers
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        try:
            with httpx.Client(
                timeout=self.timeout_seconds,
                follow_redirects=True,
                headers=headers,
            ) as client:
                response = client.get(url)
                response.raise_for_status()
                html_content = response.text
                status_code = response.status_code
        except Exception as e:
            logger.error("HTTP fetch failed", url=url, error=str(e))
            return WebPageFetchResponse(
                url=url,
                title="Fetch Error",
                content=f"Failed to fetch webpage: {e!s}",
                raw_length=0,
                is_safe=True,
                metadata={"error": str(e)},
            )

        # 4. Content Distillation (HTML -> Clean Text) (FR-WEB-3)
        title, clean_text, extracted_meta = self._extract_clean_content(html_content, url)

        # 5. Indirect Prompt-Injection Sanitization (FR-WEB-8)
        sanitized_content = sanitize_web_content(clean_text)

        metadata: dict[str, Any] = {
            "status_code": status_code,
            "char_count": len(clean_text),
            **extracted_meta,
        }

        return WebPageFetchResponse(
            url=url,
            title=title,
            content=sanitized_content,
            raw_length=len(clean_text),
            fetched_at=datetime.now(UTC).isoformat(),
            is_safe=True,
            metadata=metadata,
        )

    def _is_allowed_by_robots(self, url: str) -> bool:
        """Check if robots.txt permits crawling the URL.

        Args:
            url: Target URL.

        Returns:
            True if permitted or if robots.txt is unavailable; False if disallowed.
        """
        try:
            parsed = urlparse(url)
            origin = f"{parsed.scheme}://{parsed.netloc}"
            robots_url = f"{origin}/robots.txt"

            if origin not in self._robots_cache:
                rp = urllib.robotparser.RobotFileParser()
                rp.set_url(robots_url)
                # Attempt to read robots.txt with quick timeout
                try:
                    with httpx.Client(timeout=3.0, follow_redirects=True) as client:
                        resp = client.get(robots_url)
                        if resp.status_code == 200:
                            rp.parse(resp.text.splitlines())
                        else:
                            rp.parse(["User-agent: *", "Allow: /"])
                except Exception:
                    # If robots.txt cannot be reached, default to permissive
                    rp.parse(["User-agent: *", "Allow: /"])
                self._robots_cache[origin] = rp

            return self._robots_cache[origin].can_fetch(USER_AGENT, url)
        except Exception as e:
            logger.debug("Failed robots.txt check, defaulting to allow", url=url, error=str(e))
            return True

    def _extract_clean_content(
        self, html_content: str, url: str
    ) -> tuple[str, str, dict[str, Any]]:
        """Parse raw HTML and extract title, semantic text body, and meta tags.

        Args:
            html_content: Raw HTML string.
            url: Original URL for fallback title.

        Returns:
            Tuple of (title, clean_text, metadata_dict).
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        if not title:
            h1 = soup.find("h1")
            title = h1.get_text(strip=True) if h1 else urlparse(url).netloc

        # Extract meta description
        meta_desc = ""
        meta_desc_tag = soup.find("meta", attrs={"name": "description"})
        if meta_desc_tag and "content" in meta_desc_tag.attrs:
            meta_desc = str(meta_desc_tag.attrs["content"]).strip()

        # Remove boilerplates, scripts, styles, and UI chrome
        for unwanted in soup(
            ["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript", "svg"]
        ):
            unwanted.decompose()

        # Prioritize semantic article/main container if present
        content_container = soup.find("article") or soup.find("main") or soup.body or soup

        # Extract visible paragraph and header text
        blocks: list[str] = []
        for elem in content_container.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
            text = elem.get_text(separator=" ", strip=True)
            if text and len(text) > 10:
                blocks.append(text)

        cleaned_text = "\n\n".join(blocks)
        if not cleaned_text:
            # Fallback to general text extraction
            cleaned_text = content_container.get_text(separator="\n", strip=True)

        return title, cleaned_text, {"meta_description": meta_desc}
