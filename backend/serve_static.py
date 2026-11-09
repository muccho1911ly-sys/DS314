"""Fast threaded static file server for AIC/CAM data.
Serves D:/new_ERS/event-retrieval-system/data on port 8001.
Separate from the FastAPI backend so static file I/O doesn't block the API.

Usage:
    python serve_static.py
"""
import http.server
import socketserver
import os

PORT = 8001
DIRECTORY = "D:/new_ERS/event-retrieval-system/data"


class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "public, max-age=3600")
        super().end_headers()

    def log_message(self, format, *args):
        pass  # suppress per-request logs


socketserver.TCPServer.allow_reuse_address = True
with socketserver.ThreadingTCPServer(("0.0.0.0", PORT), CORSHandler) as httpd:
    print(f"Serving {DIRECTORY} on http://0.0.0.0:{PORT}")
    print("  Keyframes : http://localhost:8001/AIC/keyframes/")
    print("  Videos    : http://localhost:8001/AIC/videos/")
    httpd.serve_forever()
