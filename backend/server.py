#!/usr/bin/env python3
"""PathQA local server.

A zero-extra-dependency backend (Python stdlib only, beyond torch) chosen for an
on-prem / air-gapped lab: nothing to install, nothing phones home, slide data
never leaves the machine. Serves the JSON API and the single-file React UI.

Run:
    PYTHONPATH=. python backend/server.py            # demo model, synthetic data
    PATHQA_MODEL=model.pt PYTHONPATH=. python backend/server.py --port 8000

Then open http://127.0.0.1:8000  (binds to localhost only by design).

To swap in FastAPI+uvicorn later, the QASession API in qa_service.py maps 1:1
onto path operations; the routing here is deliberately thin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from qa_service import QASession

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
SESSION: QASession = None  # set in main()

# Optional free password gate. Set env PATHQA_PASSWORD to require a passcode
# (protects the app without a paid load balancer). Leave unset to disable
# (e.g. when a real auth layer like IAP sits in front).
_PW = os.environ.get("PATHQA_PASSWORD")
AUTH_TOKEN = hashlib.sha256(_PW.encode()).hexdigest()[:32] if _PW else None

LOGIN_HTML = b"""<!doctype html><meta charset="utf-8"><title>Pathify sign in</title>
<style>body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f7f8fa;color:#111827;display:flex;min-height:100vh;align-items:center;justify-content:center;margin:0}
.box{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:26px;width:320px;text-align:center}
h1{font-size:18px;margin:0 0 4px}.s{color:#6b7280;font-size:13px;margin-bottom:16px}
input{width:100%;padding:10px 12px;border:1px solid #e5e7eb;border-radius:10px;font-size:15px;margin-bottom:10px}
button{width:100%;padding:10px;border:0;border-radius:10px;background:#0f766e;color:#fff;font-weight:600;font-size:15px;cursor:pointer}
.e{color:#b91c1c;font-size:13px;min-height:18px;margin-top:8px}</style>
<div class="box"><h1>Pathify</h1><div class="s">Enter the access code to continue.</div>
<input id="p" type="password" placeholder="Access code" autofocus>
<button onclick="go()">Sign in</button><div class="e" id="e"></div></div>
<script>async function go(){const r=await fetch("/api/login",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({password:document.getElementById("p").value})});
if(r.ok){location.reload();}else{document.getElementById("e").textContent="Wrong code, try again.";}}
document.getElementById("p").addEventListener("keydown",e=>{if(e.key==="Enter")go();});</script>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "PathQA/0.1"

    # -- helpers --
    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _file(self, path: Path, ctype):
        if not path.exists():
            return self._json({"error": f"not found: {path.name}"}, 404)
        self._send(200, path.read_bytes(), ctype)

    def log_message(self, *a):  # quieter logs
        pass

    def _user(self):
        # IAP injects the signed-in user's email; use it for audit provenance.
        raw = self.headers.get("X-Goog-Authenticated-User-Email", "")
        return raw.replace("accounts.google.com:", "") or "anonymous"

    def _authed(self):
        if not AUTH_TOKEN:
            return True
        return ("pathify_auth=" + AUTH_TOKEN) in (self.headers.get("Cookie") or "")

    # -- routes --
    def do_GET(self):
        p = urlparse(self.path).path
        if AUTH_TOKEN and not self._authed():
            return self._send(200, LOGIN_HTML, "text/html; charset=utf-8")
        if p in ("/", "/index.html"):
            return self._file(FRONTEND / "index.html", "text/html; charset=utf-8")
        if p == "/app.jsx":
            return self._file(FRONTEND / "app.jsx", "text/babel; charset=utf-8")
        if p == "/api/summary":
            return self._json(SESSION.summary())
        if p == "/api/worklist":
            return self._json({"items": SESSION.worklist()})
        if p.startswith("/api/patch/") and p.endswith("/cam.png"):
            cid = p[len("/api/patch/"):-len("/cam.png")]
            png = SESSION.patch_cam_png(cid)
            return self._send(200, png, "image/png") if png else self._json({"error": "no patch"}, 404)
        if p.startswith("/api/patch/"):
            cid = p.rsplit("/", 1)[-1].replace(".png", "")
            png = SESSION.patch_png(cid)
            return self._send(200, png, "image/png") if png else self._json({"error": "no patch"}, 404)
        if p == "/api/feedback/stats":
            return self._json(SESSION.feedback_stats())
        if p == "/api/model-info":
            mi = dict(getattr(SESSION, "model_info", {"source": "demo", "metrics": {}}))
            base = getattr(SESSION, "BASE_IMAGES", 0)
            contrib = SESSION.feedback_stats().get("total", 0)
            mi.update(base_images=base, contributions=contrib, total_images=base + contrib)
            return self._json(mi)
        if p == "/api/slides/summary":
            return self._json(SESSION.slide_summary())
        if p == "/api/slides/worklist":
            return self._json({"items": SESSION.slide_worklist()})
        if p.startswith("/api/slide/") and p.endswith("/heatmap.png"):
            sid = p[len("/api/slide/"):-len("/heatmap.png")]
            png = SESSION.slide_heatmap_png(sid)
            return self._send(200, png, "image/png") if png else self._json({"error": "no slide"}, 404)
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        p = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        try:
            if p == "/api/login":
                pw = str(payload.get("password", ""))
                if AUTH_TOKEN and hashlib.sha256(pw.encode()).hexdigest()[:32] == AUTH_TOKEN:
                    body = b'{"ok":true}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Set-Cookie", f"pathify_auth={AUTH_TOKEN}; Path=/; HttpOnly; SameSite=Lax; Max-Age=604800")
                    self.end_headers(); self.wfile.write(body); return
                return self._json({"error": "wrong password"}, 401)
            if AUTH_TOKEN and not self._authed():
                return self._json({"error": "auth required"}, 401)
            if p == "/api/decision":
                entry = SESSION.record_decision(
                    payload["case_id"], payload["action"], payload.get("note", ""), user=self._user())
                return self._json({"ok": True, "entry": entry})
            if p == "/api/settings":
                SESSION.set_threshold(tau=payload.get("tau"), budget=payload.get("budget"))
                return self._json({"ok": True, "summary": SESSION.summary()})
            if p == "/api/analyze-image":
                return self._json(SESSION.analyze_image(payload["image"]))
            if p == "/api/load-model":
                return self._json(SESSION.load_model_from_upload(payload["model"]))
            if p == "/api/retrain":
                return self._json(SESSION.retrain_on_feedback(int(payload.get("epochs", 3))))
            if p == "/api/feedback":
                res = SESSION.save_feedback(payload["image"], payload["label"],
                                            auto_learn=payload.get("auto_learn", False), user=self._user())
                return self._json({"ok": True, **res})
            if p == "/api/slides/decision":
                entry = SESSION.record_slide_decision(
                    payload["slide_id"], payload["action"], payload.get("note", ""), user=self._user())
                return self._json({"ok": True, "entry": entry})
            if p == "/api/slides/settings":
                SESSION.set_slide_tau(payload["tau"])
                return self._json({"ok": True, "summary": SESSION.slide_summary()})
        except (KeyError, ValueError) as e:
            return self._json({"error": str(e)}, 400)
        return self._json({"error": "not found"}, 404)


def serve(host="127.0.0.1", port=8000, open_browser=False, **session_kwargs):
    """Build the QA session and serve until interrupted."""
    global SESSION
    print("[PathQA] preparing QA session (loading/fine-tuning model, scoring slides)...")
    SESSION = QASession(model_path=os.environ.get("PATHQA_MODEL"), **session_kwargs)
    s = SESSION.summary()
    print(f"[PathQA] ready: {s['n_flagged']}/{s['n_cases']} patch cases flagged "
          f"({s['flag_rate']:.1%}) at tau={s['tau']:.2f}")
    url = f"http://{host}:{port}"
    print(f"[PathQA] serving on {url}  (localhost only — Ctrl+C to stop)")
    if open_browser:
        import threading, webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        ThreadingHTTPServer((host, port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\n[PathQA] stopped.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--slides", type=int, default=6)
    ap.add_argument("--slide-grid", type=int, default=14)
    ap.add_argument("--demo-train-steps", type=int, default=3)
    ap.add_argument("--open", action="store_true", help="open the browser")
    args = ap.parse_args()
    # Cloud hosts (Cloud Run, etc.) tell the app which port to use via $PORT.
    port = int(os.environ.get("PORT") or args.port)
    serve(host=args.host, port=port, open_browser=args.open,
          tau=args.tau, budget=args.budget, n_cases=args.cases,
          demo_train_steps=args.demo_train_steps,
          n_slides=args.slides, slide_grid=args.slide_grid)


if __name__ == "__main__":
    main()
