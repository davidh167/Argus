"""SSRF guard (requests-hardened) and size/content-type caps in SafeFetch.

Blocking cases use the production config with DNS mocked: the filter rejects before any socket is
opened. Success/redirect/size cases talk to a throwaway server on 127.0.0.1 through a config that
allows loopback (the only difference from production), so no external network is touched.
"""

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
from requests_hardened import Config
from requests_hardened.ip_filter import InvalidIPAddress

from app.crawlers import fetch as fetch_mod
from app.crawlers.fetch import (
    DEFAULT_CONFIG,
    ResponseTooLargeError,
    SafeFetch,
    UnsupportedContentTypeError,
)

HOSTS = {
    "jobs.example.com": "93.184.216.34",
    "evil.example.com": "10.0.0.5",
    "loopback.example.com": "127.0.0.1",
    "meta.example.com": "169.254.169.254",
    "v6.example.com": "::1",
    "mapped.example.com": "::ffff:127.0.0.1",
}


@pytest.fixture
def fake_dns(monkeypatch):
    real = socket.getaddrinfo

    def getaddrinfo(host, port, *args, **kwargs):
        if host in HOSTS:
            addr = HOSTS[host]
            fam = socket.AF_INET6 if ":" in addr else socket.AF_INET
            return [(fam, socket.SOCK_STREAM, 6, "", (addr, port or 0))]
        return real(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000/docs",
        "http://169.254.169.254/latest/meta-data/",
        "http://0.0.0.0/",
        "http://192.168.0.1/",
        "http://[::1]/",
        "http://evil.example.com/",
        "http://loopback.example.com/",
        "http://meta.example.com/",
        "http://v6.example.com/",
        "http://mapped.example.com/",
    ],
)
def test_blocks_non_public_targets(fake_dns, url):
    with pytest.raises(InvalidIPAddress):
        SafeFetch().fetch(url)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://jobs.example.com/x"])
def test_rejects_non_http_schemes(fake_dns, url):
    with pytest.raises(requests.exceptions.InvalidSchema):
        SafeFetch().fetch(url)


def test_rejects_hostless_url(fake_dns):
    with pytest.raises(requests.exceptions.InvalidURL):
        SafeFetch().fetch("http:///nohost")


LOCAL_CONFIG = Config(
    ip_filter_enable=True,
    ip_filter_allow_loopback_ips=True,
    never_redirect=False,
    default_timeout=(5, 5),
)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path
        if path == "/ok":
            self._send(200, "text/html; charset=utf-8", b"<p>ok</p>")
        elif path == "/json":
            self._send(200, "application/json", b'{"a": 1}')
        elif path == "/utf8-no-charset":
            self._send(200, "text/html", "café".encode())
        elif path == "/zip":
            self._send(200, "application/zip", b"PK")
        elif path == "/big":
            self._send(200, "text/html", b"x" * 101)
        elif path == "/missing":
            self._send(404, "text/html", b"no")
        elif path == "/relative":
            self._redirect("/ok")
        elif path == "/to-metadata":
            self._redirect("http://169.254.169.254/latest/meta-data/")
        elif path == "/loop":
            self._redirect("/loop")

    def _send(self, status, ctype, body):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, location):
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture(scope="module")
def base_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _fetch(base_url, path):
    return SafeFetch(LOCAL_CONFIG).fetch(base_url + path)


def test_production_config_blocks_the_same_local_server(base_url):
    assert DEFAULT_CONFIG.ip_filter_allow_loopback_ips is False
    with pytest.raises(InvalidIPAddress):
        SafeFetch().fetch(base_url + "/ok")


def test_returns_body(base_url):
    assert _fetch(base_url, "/ok") == "<p>ok</p>"


def test_json_allowed(base_url):
    assert _fetch(base_url, "/json") == '{"a": 1}'


def test_defaults_to_utf8_without_charset(base_url):
    assert _fetch(base_url, "/utf8-no-charset") == "café"


def test_follows_safe_relative_redirect(base_url):
    assert _fetch(base_url, "/relative") == "<p>ok</p>"


def test_redirect_to_metadata_host_is_blocked(base_url):
    with pytest.raises(InvalidIPAddress):
        _fetch(base_url, "/to-metadata")


def test_redirect_cap(base_url):
    with pytest.raises(requests.exceptions.TooManyRedirects):
        _fetch(base_url, "/loop")


def test_rejects_wrong_content_type(base_url):
    with pytest.raises(UnsupportedContentTypeError):
        _fetch(base_url, "/zip")


def test_rejects_oversized_body(base_url, monkeypatch):
    monkeypatch.setattr(fetch_mod, "MAX_RESPONSE_BYTES", 100)
    with pytest.raises(ResponseTooLargeError):
        _fetch(base_url, "/big")


def test_http_error_raises(base_url):
    with pytest.raises(requests.exceptions.HTTPError):
        _fetch(base_url, "/missing")
