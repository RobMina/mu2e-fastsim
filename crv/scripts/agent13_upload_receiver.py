"""Short-lived TLS upload endpoint for an allocated node's private tmpfs."""
import argparse
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import ssl
import subprocess

FILES = {"code.tgz", "gen.001400_00000011.root", "lib_prod_as_20files.root",
         "crvcalib_sim_best_v1_5_run1400.csv", "crvphoton_sim_best_v1_5_run1400.csv",
         "mu2e_extracted_v04.gdml", "manifest.sha256"}


def serve(root, output):
    root, output = Path(root), Path(output)
    token = secrets.token_hex(32)
    key, cert = root / "upload.key", output / "upload.crt"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                    "-keyout", str(key), "-out", str(cert), "-subj", "/CN=agent13-upload",
                    "-addext", "subjectAltName=DNS:agent13-upload"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            self.request.settimeout(120)
            super().setup()

        def reply(self, data):
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

        def authorized(self):
            if hmac.compare_digest(self.headers.get("X-Agent13-Token", ""), token):
                return True
            self.send_error(403)
            return False

        def do_PUT(self):
            if not self.authorized():
                return
            name = self.path.removeprefix("/")
            length = int(self.headers.get("Content-Length", "-1"))
            if name not in FILES or not 0 <= length <= 2 * 1024**3 or (root / name).exists():
                self.send_error(400)
                return
            remaining, digest = length, hashlib.sha256()
            target = root / (name + ".partial")
            self.connection.settimeout(120)
            with target.open("xb") as stream:
                while remaining:
                    data = self.rfile.read(min(1024**2, remaining))
                    if not data:
                        raise ConnectionError("Incomplete upload")
                    stream.write(data)
                    digest.update(data)
                    remaining -= len(data)
            target.rename(root / name)
            self.reply((digest.hexdigest() + "\n").encode())

        def do_POST(self):
            if not self.authorized():
                return
            if self.path != "/ready":
                self.send_error(404)
                return
            entries = {}
            for line in (root / "manifest.sha256").read_text().splitlines():
                digest, name = line.split()
                if name not in FILES - {"manifest.sha256"} or name in entries:
                    self.send_error(400)
                    return
                with (root / name).open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if not hmac.compare_digest(actual, digest):
                    self.send_error(409)
                    return
                entries[name] = digest
            if entries.keys() != FILES - {"manifest.sha256"}:
                self.send_error(400)
                return
            (root / "ready").touch(exist_ok=False)
            self.reply(b"ready\n")

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)

    class Server(ThreadingHTTPServer):
        def get_request(self):
            sock, address = super().get_request()
            sock.settimeout(10)
            return context.wrap_socket(sock, server_side=True, do_handshake_on_connect=False), address

    server = Server(("0.0.0.0", 0), Handler)
    (output / "upload.token").write_text(token)
    (output / "upload.headers").write_text(f"X-Agent13-Token: {token}\n")
    (output / "upload.port").write_text(str(server.server_port))
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("output")
    args = parser.parse_args()
    serve(args.root, args.output)
