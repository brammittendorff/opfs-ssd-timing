#!/usr/bin/env python3
"""Static server for the FROST OPFS PoC.

Sets COOP/COEP so the page is cross-origin isolated, which unlocks the
high-resolution performance.now() timer the attack needs. Also disables
caching so edits to the JS/HTML always reload.

Usage:  python3 serve.py [port]   (default port 8000)
Open:   http://localhost:8000  in Google Chrome
"""
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class FrostHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Cross-origin isolation -> crossOriginIsolated === true -> high-res timers.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        # No caching while developing/testing.
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("[serve] " + (fmt % args) + "\n")


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    root = Path(__file__).resolve().parent
    handler = partial(FrostHandler, directory=str(root))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"[serve] FROST PoC serving {root}")
    print(f"[serve] Open http://localhost:{port}  in Google Chrome")
    print("[serve] Ctrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] bye")
        httpd.shutdown()


if __name__ == "__main__":
    main()
