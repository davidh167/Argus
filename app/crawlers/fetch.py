import ipaddress
import logging
import socket
from abc import ABC, abstractmethod
from urllib.parse import urljoin, urlsplit

import httpx

logger = logging.getLogger(__name__)

MAX_REDIRECTS = 5
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
ALLOWED_CONTENT_TYPES = (
    "text/html",
    "application/xhtml+xml",
    "application/json",
    "application/ld+json",
)
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class UnsafeUrlError(ValueError):
    """URL is not an http(s) URL to a public host (SSRF guard)."""


class TooManyRedirectsError(RuntimeError):
    pass


class ResponseTooLargeError(RuntimeError):
    pass


class UnsupportedContentTypeError(RuntimeError):
    pass


def validate_public_url(url: str) -> None:
    """Reject anything that is not http(s) to a host resolving only to public addresses.

    Scraped hrefs and user-entered careers URLs are untrusted; without this a page can point us
    at loopback services or cloud metadata (169.254.169.254) and have the body stored as a JD.
    Raises UnsafeUrlError (and logs) rather than falling back."""
    parts = urlsplit(url)
    host = parts.hostname
    if parts.scheme not in ("http", "https") or not host:
        logger.warning("Blocked fetch of non-http(s) or hostless URL: %r", url)
        raise UnsafeUrlError(f"Only http(s) URLs with a host are allowed: {url!r}")
    try:
        infos = socket.getaddrinfo(host, parts.port or None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        logger.warning("Blocked fetch, host did not resolve: %r", url)
        raise UnsafeUrlError(f"Could not resolve host {host!r}") from exc
    for info in infos:
        ip = ipaddress.ip_address(str(info[4][0]).split("%")[0])
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if not ip.is_global or ip.is_multicast:
            logger.warning("Blocked fetch of %r: host %r resolves to non-public %s", url, host, ip)
            raise UnsafeUrlError(f"Host {host!r} resolves to a non-public address ({ip})")


class FetchStrategy(ABC):
    @abstractmethod
    def fetch(self, url: str) -> str:
        """Return raw HTML for url."""


class HttpxFetch(FetchStrategy):
    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        self._transport = transport  # tests inject a mock transport; None means real network

    def fetch(self, url: str) -> str:
        with httpx.Client(timeout=30, follow_redirects=False, transport=self._transport) as client:
            for _ in range(MAX_REDIRECTS + 1):
                validate_public_url(url)  # every hop, not just the first
                with client.stream("GET", url) as resp:
                    if resp.status_code in _REDIRECT_STATUSES and "location" in resp.headers:
                        url = urljoin(url, resp.headers["location"])
                        continue
                    resp.raise_for_status()
                    return _read_capped_text(resp, url)
        logger.warning("Too many redirects fetching %r", url)
        raise TooManyRedirectsError(f"More than {MAX_REDIRECTS} redirects, last: {url!r}")


def _read_capped_text(resp: httpx.Response, url: str) -> str:
    content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning("Rejected %r: content-type %r", url, content_type)
        raise UnsupportedContentTypeError(f"Unsupported content-type {content_type!r} for {url!r}")
    body = bytearray()
    for chunk in resp.iter_bytes():
        body.extend(chunk)
        if len(body) > MAX_RESPONSE_BYTES:
            logger.warning("Rejected %r: body over %d bytes", url, MAX_RESPONSE_BYTES)
            raise ResponseTooLargeError(f"Response for {url!r} exceeds {MAX_RESPONSE_BYTES} bytes")
    return body.decode(resp.encoding or "utf-8", errors="replace")
