"""Unit tests for the proxy-aware rate-limit key in api/main.py.

Behind the HF Space proxy every client shares one socket IP, so keying the
slowapi limiter on it turns "20/minute" into a global cap. _client_key()
prefers the leftmost X-Forwarded-For hop when FILGOAL_TRUST_PROXY=1.
"""

from __future__ import annotations

from fastapi import Request

from api.main import _client_key


def _request(xff: str | None = None, client_host: str = "9.9.9.9") -> Request:
    headers = [(b"x-forwarded-for", xff.encode())] if xff is not None else []
    return Request({
        "type": "http",
        "method": "GET",
        "path": "/ask",
        "headers": headers,
        "client": (client_host, 5000),
        "server": ("testserver", 80),
        "scheme": "http",
    })


def test_xff_honored_when_trust_proxy_on(monkeypatch):
    monkeypatch.setenv("FILGOAL_TRUST_PROXY", "1")
    req = _request(xff="1.2.3.4, 5.6.7.8", client_host="9.9.9.9")
    assert _client_key(req) == "1.2.3.4"


def test_xff_ignored_when_trust_proxy_off(monkeypatch):
    monkeypatch.setenv("FILGOAL_TRUST_PROXY", "0")
    req = _request(xff="1.2.3.4, 5.6.7.8", client_host="9.9.9.9")
    assert _client_key(req) == "9.9.9.9"


def test_socket_ip_used_when_no_xff(monkeypatch):
    monkeypatch.setenv("FILGOAL_TRUST_PROXY", "1")
    req = _request(xff=None, client_host="9.9.9.9")
    assert _client_key(req) == "9.9.9.9"


def test_distinct_clients_get_distinct_keys(monkeypatch):
    monkeypatch.setenv("FILGOAL_TRUST_PROXY", "1")
    assert _client_key(_request(xff="1.1.1.1")) != _client_key(_request(xff="2.2.2.2"))
