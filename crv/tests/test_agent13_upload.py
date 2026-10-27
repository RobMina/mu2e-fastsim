"""Exercise authenticated TLS uploads without detector inputs."""
import hashlib
import http.client
import importlib.util
from pathlib import Path
import subprocess
import sys
import time


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "agent13_upload_receiver.py"


def test_upload_contract(tmp_path):
    root, output = tmp_path / "private", tmp_path / "summary"
    root.mkdir()
    output.mkdir()
    spec = importlib.util.spec_from_file_location("receiver", SCRIPT)
    receiver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(receiver)
    process = subprocess.Popen([sys.executable, str(SCRIPT), str(root), str(output)])
    try:
        for _ in range(100):
            if (output / "upload.port").exists():
                break
            assert process.poll() is None
            time.sleep(0.05)
        port = int((output / "upload.port").read_text())
        token = (output / "upload.token").read_text()
        import ssl
        import socket
        context = ssl.create_default_context(cafile=str(output / "upload.crt"))

        def request(method, path, data=b"", secret=token):
            connection = http.client.HTTPSConnection("agent13-upload", port, context=context, timeout=5)
            connection.sock = context.wrap_socket(socket.create_connection(("127.0.0.1", port)), server_hostname="agent13-upload")
            connection.request(method, path, body=data, headers={"X-Agent13-Token": secret})
            response = connection.getresponse()
            status, body = response.status, response.read()
            connection.close()
            return status, body

        curl = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--noproxy", "*",
                               "--cacert", str(output / "upload.crt"), "--connect-to",
                               f"agent13-upload:{port}:127.0.0.1:{port}",
                               "-H", "@" + str(output / "upload.headers"), "-H", "Content-Length: 0",
                               "-H", "Transfer-Encoding:", "--upload-file", "-",
                               f"https://agent13-upload:{port}/code.tgz"], input=b"", capture_output=True)
        assert curl.returncode == 0, curl.stderr
        assert curl.stdout == (hashlib.sha256(b"").hexdigest() + "\n").encode()
        (root / "code.tgz").unlink()
        assert request("PUT", "/code.tgz", secret="wrong")[0] == 403
        assert request("PUT", "/../escape")[0] == 400
        entries = []
        for name in sorted(receiver.FILES - {"manifest.sha256"}):
            data = name.encode()
            digest = hashlib.sha256(data).hexdigest()
            assert request("PUT", "/" + name, data) == (200, (digest + "\n").encode())
            assert request("PUT", "/" + name, data)[0] == 400
            entries.append(f"{digest}  {name}\n")
        manifest = "".join(entries).encode()
        assert request("PUT", "/manifest.sha256", manifest)[0] == 200
        assert request("POST", "/ready") == (200, b"ready\n")
        assert (root / "ready").exists()
        assert not (tmp_path / "escape").exists()
    finally:
        process.terminate()
        process.wait(timeout=10)
