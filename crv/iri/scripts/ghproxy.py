#!/usr/bin/env python3
"""ssh ProxyCommand for Polaris compute nodes: tunnel host:port through the ALCF HTTP proxy (no nc/socat there).

usage (ssh): -o ProxyCommand="python3 ~/.ssh/ghproxy.py %h %p"
"""
import os
import select
import socket
import sys

PROXY = ("proxy.alcf.anl.gov", 3128)


def main():
    host, port = sys.argv[1], sys.argv[2]
    s = socket.create_connection(PROXY, timeout=20)
    s.sendall(f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode())
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = s.recv(4096)
        if not chunk:
            sys.exit("proxy closed the connection")
        head += chunk
    status = head.split(b"\r\n", 1)[0]
    if b" 200" not in status:
        sys.exit(f"proxy refused: {status.decode(errors='replace')}")
    rest = head.split(b"\r\n\r\n", 1)[1]
    if rest:
        os.write(1, rest)
    s.settimeout(None)
    fin, fout = 0, 1
    while True:
        r, _, _ = select.select([s, fin], [], [])
        if s in r:
            data = s.recv(65536)
            if not data:
                break
            os.write(fout, data)
        if fin in r:
            data = os.read(fin, 65536)
            if not data:
                s.shutdown(socket.SHUT_WR)
                while True:
                    data = s.recv(65536)
                    if not data:
                        return
                    os.write(fout, data)
            s.sendall(data)


if __name__ == "__main__":
    main()
