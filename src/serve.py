"""Web server — serves the dashboard + a live /api/now endpoint. Runs locally AND on a host
(Render/Railway/Fly free tier) for a PUBLIC, LIVE site. Stdlib only.

Local:   python -m src.serve            -> http://localhost:8000
Hosted:  gunicorn not needed; `python -m src.serve` binds 0.0.0.0:$PORT and self-heals on boot
         (uses bundled panel/models when present; never blocks requests on heavy work).

Endpoints:  /  (dashboard)   /api/now  (live prediction JSON)   /health  (readiness)
"""
from __future__ import annotations
import http.server, json, os, threading, time
from .util import ROOT, RESULTS
from . import dashboard

PORT = int(os.environ.get("PORT", "8000"))
_READY = {"ok": False, "msg": "starting"}
# /api/now is expensive (full feature rebuild). Serve a cached result and refresh it in a
# background thread so the web server never blocks — recomputing per request starved the
# single-threaded server and got the instance killed by the host's health checks.
NOW_TTL = 300
_NOW = {"ts": 0.0, "data": None, "computing": False}
_NOW_LOCK = threading.Lock()
_TRAIN_LOCK = threading.Lock()
_TRAINING = False


def _ensure():
    """On boot: serve dashboard immediately; train models in the background only if missing."""
    global _TRAINING
    dash = ROOT / "dashboard.html"
    if dash.exists():
        _READY.update(ok=True, msg="ready")
    try:
        from . import cache
        if not (RESULTS / "models.joblib").exists():
            with _TRAIN_LOCK:
                _TRAINING = True
                _READY.update(msg="training models")
                try:
                    # Use bundled/cached panel only — live fetch on boot OOMs the free tier.
                    cache.train_and_cache()
                finally:
                    _TRAINING = False
        _READY.update(ok=True, msg="ready")
        _compute_now()   # warm the /api/now cache so the first visitor gets live data
    except Exception as e:
        if dash.exists():
            _READY.update(ok=True, msg=f"ready (api warming: {e})")
        else:
            _READY.update(ok=False, msg=f"boot error: {e}")


def _compute_now():
    if _TRAINING:
        return
    try:
        from . import cache
        out = cache.predict_now()
    except Exception as e:
        out = {"error": str(e)[:160]}
    with _NOW_LOCK:
        _NOW.update(ts=time.time(), data=out, computing=False)


def _now():
    """Return the cached live prediction; kick off a background refresh if stale."""
    if _TRAINING:
        return {"error": "warming up — models still loading, try again shortly"}
    with _NOW_LOCK:
        # retry errors quickly (e.g. models still training on boot); good results last NOW_TTL
        ttl = 20 if (_NOW["data"] is None or "error" in (_NOW["data"] or {})) else NOW_TTL
        stale = _NOW["data"] is None or (time.time() - _NOW["ts"]) > ttl
        if stale and not _NOW["computing"]:
            _NOW["computing"] = True
            threading.Thread(target=_compute_now, daemon=True).start()
        return _NOW["data"] or {"error": "warming up — models still loading, try again shortly"}


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, body, ctype, code=200):
        try:
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        except BrokenPipeError:
            pass

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(json.dumps(_READY).encode(), "application/json",
                       200 if _READY["ok"] else 503); return
        if self.path.startswith("/api/now"):
            self._send(json.dumps(_now()).encode(), "application/json"); return
        if self.path in ("/", "/index.html", "/dashboard.html"):
            p = ROOT / "dashboard.html"
            if not p.exists():
                self._send(b"dashboard not ready", "text/plain", 503); return
            self._send(p.read_bytes(), "text/html; charset=utf-8"); return
        self._send(b"not found", "text/plain", 404)

    def log_message(self, *a): pass


def main():
    if (ROOT / "dashboard.html").exists():
        _READY.update(ok=True, msg="ready")
    threading.Thread(target=_ensure, daemon=True).start()   # self-heal without blocking bind
    print(f"Boeing monitor on http://0.0.0.0:{PORT}  (/, /api/now, /health)")
    # threaded server: slow requests must never block /health, or the host kills the instance
    with http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler) as httpd:
        httpd.daemon_threads = True
        httpd.serve_forever()


if __name__ == "__main__":
    main()
