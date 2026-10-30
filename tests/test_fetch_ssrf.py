"""SSRF guard for HttpxFetch. DNS and HTTP are mocked; no real network."""

import socket

import httpx
import pytest

from app.crawlers import fetch as fetch_mod
from app.crawlers.fetch import (
    HttpxFetch,
    ResponseTooLargeError,
    TooManyRedirectsError,
    UnsafeUrlError,
    UnsupportedContentTypeError,
    validate_public_url,
)

HOSTS = {
    "jobs.example.com": ["93.184.216.34"],
    "evil.example.com": ["10.0.0.5"],
    "loopback.example.com": ["127.0.0.1"],
    "meta.example.com": ["169.254.169.254"],
    "v6.example.com": ["::1"],
    "mapped.example.com": ["::ffff:127.0.0.1"],
    "multi.example.com": ["93.184.216.34", "192.168.1.1"],
}


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    def getaddrinfo(host, port, *args, **kwargs):
        if host in HOSTS:
            addrs = HOSTS[host]
        else:
            try:
                socket.inet_pton(socket.AF_INET, host)
            except OSError:
                raise socket.gaierror("no such host") from None
            addrs = [host]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, 0)) for a in addrs]

    monkeypatch.setattr(fetch_mod.socket, "getaddrinfo", getaddrinfo)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://jobs.example.com/x",
        "http:///nohost",
        "http://127.0.0.1:8000/docs",
        "http://169.254.169.254/latest/meta-data/",
        "http://0.0.0.0/",
        "http://224.0.0.1/",
        "http://192.168.0.1/",
        "http://evil.example.com/",
        "http://loopback.example.com/",
        "http://meta.example.com/",
        "http://v6.example.com/",
        "http://mapped.example.com/",
        "http://multi.example.com/",
        "http://unresolvable.invalid/",
    ],
)
def test_validate_rejects_unsafe_urls(url):
    with pytest.raises(UnsafeUrlError):
        validate_public_url(url)


def test_validate_allows_public_host():
    validate_public_url("https://jobs.example.com/careers")


def _fetcher(handler):
    return HttpxFetch(transport=httpx.MockTransport(handler))


def _html(request):
    return httpx.Response(
        200, headers={"content-type": "text/html; charset=utf-8"}, text="<p>ok</p>"
    )


def test_fetch_returns_body():
    assert _fetcher(_html).fetch("https://jobs.example.com/a") == "<p>ok</p>"


def test_fetch_json_allowed():
    def handler(request):
        return httpx.Response(200, json={"a": 1})

    assert _fetcher(handler).fetch("https://jobs.example.com/a") == '{"a":1}'


def test_fetch_blocks_internal_target_without_requesting():
    called = []

    def handler(request):
        called.append(request)
        return _html(request)

    with pytest.raises(UnsafeUrlError):
        _fetcher(handler).fetch("http://127.0.0.1:8000/docs")
    assert called == []


def test_fetch_revalidates_redirect_hops():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://meta.example.com/secret"})

    with pytest.raises(UnsafeUrlError):
        _fetcher(handler).fetch("https://jobs.example.com/a")
    assert seen == ["https://jobs.example.com/a"]


def test_fetch_follows_safe_relative_redirect():
    def handler(request):
        if request.url.path == "/a":
            return httpx.Response(301, headers={"location": "/b"})
        return _html(request)

    assert _fetcher(handler).fetch("https://jobs.example.com/a") == "<p>ok</p>"


def test_fetch_redirect_cap():
    def handler(request):
        return httpx.Response(302, headers={"location": "/loop"})

    with pytest.raises(TooManyRedirectsError):
        _fetcher(handler).fetch("https://jobs.example.com/loop")


def test_fetch_rejects_wrong_content_type():
    def handler(request):
        return httpx.Response(200, headers={"content-type": "application/zip"}, content=b"PK")

    with pytest.raises(UnsupportedContentTypeError):
        _fetcher(handler).fetch("https://jobs.example.com/a")


def test_fetch_rejects_oversized_body(monkeypatch):
    monkeypatch.setattr(fetch_mod, "MAX_RESPONSE_BYTES", 100)

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"x" * 101)

    with pytest.raises(ResponseTooLargeError):
        _fetcher(handler).fetch("https://jobs.example.com/a")


def test_fetch_http_error_raises():
    def handler(request):
        return httpx.Response(404, headers={"content-type": "text/html"}, text="no")

    with pytest.raises(httpx.HTTPStatusError):
        _fetcher(handler).fetch("https://jobs.example.com/a")
