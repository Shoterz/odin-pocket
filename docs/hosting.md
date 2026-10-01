# Host the demo

Complete the Python environment and checkpoint setup in the [README](../README.md), then run:

```bash
bash scripts/serve-public.sh
```

The launcher uses CPU inference with four threads, binds to `127.0.0.1:8788`, and allows the public origin `https://odinpocket.stocksuite.app`. Keep the process running; stopping it or sleeping the computer takes the demo offline.

In Cloudflare Tunnel, add a published application route:

| Setting | Value |
| --- | --- |
| Hostname | `odinpocket.stocksuite.app` |
| Service type | HTTP |
| Service URL | `127.0.0.1:8788` |

This assumes `cloudflared` runs on the same host as ODIN. No router port forwarding is needed. Leave the HTTP Host Header override unset so the configured public hostname reaches the server. Cloudflare terminates HTTPS; the local tunnel service uses HTTP. See [Cloudflare's routing documentation](https://developers.cloudflare.com/tunnel/concepts/routing/).

After adding the route, open `https://odinpocket.stocksuite.app`, generate a passage, and check Compare endings and Model evidence before adding the URL to Devpost.

The hosted interface says that passages are sent to Axiom AI's computer for inference. The application does not write passages to request logs. Cloudflare carries the requests. This is a shared demo: one inference runs at a time, with a busy response to concurrent requests and a maximum of 256 generated tokens. It is not an authenticated multi-user service. Remote checkpoint reload is disabled; restart the process to load different weights.

To customize the interpreter, port or hostname:

```bash
ODIN_PYTHON=/path/to/venv/bin/python \
ODIN_PORT=8788 \
ODIN_PUBLIC_ORIGIN=https://odinpocket.stocksuite.app \
bash scripts/serve-public.sh
```

`ODIN_CHECKPOINT` optionally selects another checkpoint path. The public origin must be an exact HTTPS origin without a path or trailing slash. Only that hostname and localhost are accepted; wildcard hosts and cross-origin browser requests are rejected. Host and Origin checks are routing restrictions, not authentication.

For ordinary local use, `bash run.sh` retains its localhost-only behavior on port 8766.
