"""Minimal internal Alertmanager webhook receiver for local audit evidence."""

import json
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


LOG_PATH = Path(os.getenv("NOTIFICATION_LOG_PATH", "/data/notifications.jsonl"))
PORT = int(os.getenv("NOTIFICATION_PORT", "8080"))
WRITE_LOCK = threading.Lock()


class NotificationHandler(BaseHTTPRequestHandler):
    """Accept Alertmanager payloads without exposing the receiver publicly."""

    def _respond(self, status, body):
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "healthy"})
            return
        self._respond(404, {"detail": "not found"})

    def do_POST(self):
        if self.path != "/alerts":
            self._respond(404, {"detail": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or not isinstance(payload.get("alerts"), list):
                raise ValueError("expected an Alertmanager object containing an alerts list")
        except (ValueError, json.JSONDecodeError) as error:
            self._respond(400, {"detail": str(error)})
            return

        record = {
            "received_at": datetime.now(timezone.utc).isoformat(),
            "source": "alertmanager",
            "payload": payload,
        }
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with WRITE_LOCK, LOG_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        self._respond(202, {"accepted": len(payload["alerts"])})

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), NotificationHandler).serve_forever()
