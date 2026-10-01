"""Public-origin routing must not weaken the default localhost boundary."""
import json
import threading
from contextlib import contextmanager
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from odin.server import Workbench, make_server

PUBLIC_ORIGIN = 'https://odinpocket.stocksuite.app'


@contextmanager
def running(tmp_path, public_origin=None):
    app = Workbench(tmp_path / 'missing.pt', tmp_path)
    server = make_server(app, port=0, public_origin=public_origin)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def request(base, path='/api/status', host='odinpocket.stocksuite.app', origin=None, payload=None):
    headers = {'Host': host}
    if origin is not None:
        headers['Origin'] = origin
    data = None if payload is None else json.dumps(payload).encode()
    if data is not None:
        headers['Content-Type'] = 'application/json'
    return urlopen(Request(base + path, data=data, headers=headers), timeout=5)


def test_public_origin_accepts_tunnel_host_and_browser_origin(tmp_path):
    # Removing the explicit Host/Origin allowance breaks an actual browser POST.
    with running(tmp_path, PUBLIC_ORIGIN) as base:
        with request(base, origin=PUBLIC_ORIGIN) as response:
            status = json.load(response)
        assert status['hosted'] is True
        assert status['reload_allowed'] is False
        with pytest.raises(HTTPError) as exc:
            request(base, '/api/generate', origin=PUBLIC_ORIGIN, payload={'prompt': ''})
        assert exc.value.code == 400  # reaches validation, not rejected as cross-origin
        with urlopen(base + '/api/status') as response:
            assert json.load(response)['hosted'] is True


@pytest.mark.parametrize('host,origin', [
    ('evil.example', PUBLIC_ORIGIN),
    ('odinpocket.stocksuite.app.evil.example', PUBLIC_ORIGIN),
    ('odinpocket.stocksuite.app', 'https://evil.example'),
    ('odinpocket.stocksuite.app', 'http://odinpocket.stocksuite.app'),
    ('odinpocket.stocksuite.app', 'null'),
])
def test_public_origin_rejects_unlisted_hosts_and_origins(tmp_path, host, origin):
    with running(tmp_path, PUBLIC_ORIGIN) as base:
        with pytest.raises(HTTPError) as exc:
            request(base, host=host, origin=origin)
        assert exc.value.code == 403


def test_public_mode_blocks_reload_even_via_loopback(tmp_path):
    with running(tmp_path, PUBLIC_ORIGIN) as base:
        for host in ['odinpocket.stocksuite.app', base.removeprefix('http://')]:
            with pytest.raises(HTTPError) as exc:
                request(base, '/api/reload', host=host, payload={})
            assert exc.value.code == 403
            assert 'disabled' in json.load(exc.value)['error'].lower()


def test_local_mode_stays_local_and_allows_reload(tmp_path):
    with running(tmp_path) as base:
        with pytest.raises(HTTPError) as exc:
            request(base)
        assert exc.value.code == 403
        host = base.removeprefix('http://')
        with request(base, '/api/reload', host=host, payload={}) as response:
            assert json.load(response)['ready'] is False
        with urlopen(base + '/api/status') as response:
            status = json.load(response)
        assert status['hosted'] is False and status['reload_allowed'] is True


@pytest.mark.parametrize('origin', [
    'http://odinpocket.stocksuite.app', 'https://', 'https://*.stocksuite.app',
    'https://user:password@odinpocket.stocksuite.app',
    'https://odinpocket.stocksuite.app/path',
    'https://odinpocket.stocksuite.app?x=1',
    'https://odinpocket.stocksuite.app#fragment',
    'https://odinpocket.stocksuite.app:bad',
])
def test_public_origin_requires_an_exact_https_origin(tmp_path, origin):
    app = Workbench(tmp_path / 'missing.pt', tmp_path)
    with pytest.raises(ValueError):
        make_server(app, port=0, public_origin=origin)
