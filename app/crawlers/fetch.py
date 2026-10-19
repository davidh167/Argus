from abc import ABC, abstractmethod

import httpx


class FetchStrategy(ABC):
    @abstractmethod
    def fetch(self, url: str) -> str:
        """Return raw HTML for url."""


class HttpxFetch(FetchStrategy):
    def fetch(self, url: str) -> str:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        return resp.text
