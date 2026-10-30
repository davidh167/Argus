"""Fetching of untrusted URLs (scraped hrefs, user-entered careers pages).

SSRF protection (private/loopback/link-local/metadata addresses, redirect re-validation, connecting
to the validated IP so DNS rebinding cannot swap it) is done by `requests-hardened` (Saleor). This
module only adds what no maintained library provides: a response size cap and a content-type
allowlist. Failures raise: requests_hardened.ip_filter.InvalidIPAddress (blocked address),
requests.exceptions.InvalidSchema / InvalidURL (non-http(s) or hostless URL),
requests.exceptions.TooManyRedirects, requests.exceptions.HTTPError, and the two errors below.
"""

import logging
from abc import ABC, abstractmethod

from requests_hardened import Config, HTTPSession

logger = logging.getLogger(__name__)

MAX_REDIRECTS = 5
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
ALLOWED_CONTENT_TYPES = (
    "text/html",
    "application/xhtml+xml",
    "application/json",
    "application/ld+json",
)

DEFAULT_CONFIG = Config(
    ip_filter_enable=True,
    ip_filter_allow_loopback_ips=False,
    never_redirect=False,
    default_timeout=(10, 30),
)


class ResponseTooLargeError(RuntimeError):
    pass


class UnsupportedContentTypeError(RuntimeError):
    pass


class FetchStrategy(ABC):
    @abstractmethod
    def fetch(self, url: str) -> str:
        """Return raw HTML for url."""


class SafeFetch(FetchStrategy):
    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        self._config = config  # tests pass a loopback-allowing config to reach a local server

    def fetch(self, url: str) -> str:
        with HTTPSession(self._config) as session:
            session.max_redirects = MAX_REDIRECTS
            session.trust_env = False  # a proxy env var would make the IP filter check the proxy
            try:
                with session.get(url, stream=True) as resp:
                    resp.raise_for_status()
                    return _read_capped_text(resp, url)
            except Exception:
                logger.warning("Blocked or failed fetch of %r", url, exc_info=True)
                raise


def _read_capped_text(resp, url: str) -> str:
    content_type_header = resp.headers.get("content-type", "")
    content_type = content_type_header.split(";")[0].strip().lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise UnsupportedContentTypeError(f"Unsupported content-type {content_type!r} for {url!r}")
    body = bytearray()
    for chunk in resp.iter_content(chunk_size=65536):
        body.extend(chunk)
        if len(body) > MAX_RESPONSE_BYTES:
            raise ResponseTooLargeError(f"Response for {url!r} exceeds {MAX_RESPONSE_BYTES} bytes")
    # requests defaults text/* without a charset to ISO-8859-1; the web default is UTF-8.
    encoding = resp.encoding if "charset=" in content_type_header.lower() else "utf-8"
    return body.decode(encoding or "utf-8", errors="replace")
