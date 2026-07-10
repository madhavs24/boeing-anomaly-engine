"""Web server — serves the dashboard + a live /api/now endpoint. Runs locally AND on a host
(Render/Railway/Fly free tier) for a PUBLIC, LIVE site. Stdlib only.

Local:   python -m src.serve            -> http://localhost:8000
Hosted:  gunicorn not needed; `python -m src.serve` binds 0.0.0.0:$PORT and self-heals on boot
         (fetches data + trains models if missing, then serves).

Endpoints:  /  (dashboard)   /api/now  (live prediction JSON)   /health  (readiness)
"""
from __future__ import annotations
import http.server, socketserver, json, os, threading
from .util import ROOT, RESULTS
from . import dashboard

PORT = int(os.environ.get("PORT", "8000"))
_READY = {"ok": False, "msg": "starting"}


def _ensure():
    """On boot: make sure a data panel, trained models, and dashboard.html all exist."""
    try:
        from . import data, cache
        if not (RESULTS / "models.joblib").exists():
            # prefer real data if reachable (host has internet), else cached, else synthetic
            try:
                data.get_panel("live")
            except Exception:
                pass
            cache.train_and_cache()
        if not (ROOT / "dashboard.html").exists():
            dashboard.build()
        _READY.update(ok=True, msg="ready")
    except Exception as e:
        _READY.update(ok=False, msg=f"boot error: {e}")


def _now():
    try:
        from . import cache
        return cache.predict_now()
    except Exception as e:
        return {"error": str(e)[:160]}


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, body, ctype, code=200):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(json.dumps(_READY).encode(), "application/json",
                       200 if _READY["ok"] else 503); return
        if self.path.startswith("/api/now"):
            self._send(json.dumps(_now()).encode(), "application/json"); return
        if self.path in ("/", "/index.html", "/dashboard.html"):
            p = ROOT / "dashboard.html"
            if not p.exists():
                dashboard.build()
            self._send(p.read_bytes(), "text/html; charset=utf-8"); return
        self._send(b"not found", "text/plain", 404)

    def log_message(self, *a): pass


def main():
    threading.Thread(target=_ensure, daemon=True).start()   # self-heal without blocking bind
    print(f"Boeing monitor on http://0.0.0.0:{PORT}  (/, /api/now, /health)")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
