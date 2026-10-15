"""Serve WeaveSim Web on this machine and to other devices on the same network.

    python web/serve.py                 # http on port 8090 → http://localhost:8090/web/
    python web/serve.py --https         # https on port 8453 with a self-signed certificate (made once with openssl)
    python web/serve.py --port 9000     # a busy default port moves on to the next free one
    python web/serve.py --verbose       # log every request and every dropped connection

The server's root is the WeaveSim folder (the parent of web/), so the app reaches ../examples/scenes,
../examples/meshes and ../examples/calibration by relative URL — the same files the Python program uses.

Why --https: browsers expose WebGPU only in a "secure context". http://localhost counts as secure, but
http://192.168.x.x does not, so another device on the Wi-Fi gets the CPU solver over plain http. Over https it gets
WebGPU after you accept the certificate warning once (Safari: "Show details → visit this website"; Chrome:
"Advanced → proceed"). macOS may ask once to allow incoming connections for Python — allow it.
"""
from __future__ import annotations

import argparse
import errno
import http.server
import os
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                        # the WeaveSim folder
APP = 'WeaveSim Web'
PEM = 'weavesim-selfsigned.pem'


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm',
                      '.stl': 'application/octet-stream', '.obj': 'text/plain', '.csv': 'text/csv'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def do_GET(self):
        if self.path in ('/', '/index.html'):                       # the app lives in web/
            self.send_response(302)
            self.send_header('Location', '/web/')
            self.end_headers()
            return
        return super().do_GET()

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')              # always pick up edited modules and scenes
        super().end_headers()

    def log_message(self, fmt, *args):
        line = fmt % args
        if not VERBOSE and ('"GET /web/js/' in line or '"GET /web/vendor/' in line or 'favicon' in line):
            return
        super().log_message(fmt, *args)


VERBOSE = False
QUIET_ERRNO = {errno.EBADF, errno.ECONNRESET, errno.EPIPE, errno.ECONNABORTED, errno.ETIMEDOUT, errno.ENOTCONN, errno.ESHUTDOWN}


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    tls = None

    def finish_request(self, request, client_address):
        """https: a plain-http request on the https port (Safari, for a typed address) is redirected to https"""
        if self.tls is None:
            return super().finish_request(request, client_address)
        request.settimeout(15)
        first = request.recv(1, socket.MSG_PEEK)
        if not first:
            return
        if first[0] != 0x16:
            head = request.recv(8192).decode('latin-1', 'replace')
            lines = head.split('\r\n')
            path = lines[0].split(' ')[1] if len(lines[0].split(' ')) > 1 else '/'
            host = next((l.split(':', 1)[1].strip() for l in lines[1:] if l.lower().startswith('host:')), '')
            if not host:
                host = f'{(lan_addresses() or ["localhost"])[0]}:{self.server_address[1]}'
            location = f'https://{host}{path}'
            body = f'<a href="{location}">{location}</a>\n'.encode()
            request.sendall(f'HTTP/1.1 301 Moved Permanently\r\nLocation: {location}\r\nContent-Type: text/html\r\n'
                            f'Content-Length: {len(body)}\r\nConnection: close\r\n\r\n'.encode() + body)
            return
        tls = self.tls.wrap_socket(request, server_side=True)
        tls.settimeout(None)
        try:
            self.RequestHandlerClass(tls, client_address, self)
        finally:
            tls.close()

    def handle_error(self, request, client_address):
        e = sys.exc_info()[1]
        if isinstance(e, (ConnectionError, TimeoutError, ssl.SSLError)) or (isinstance(e, OSError) and e.errno in QUIET_ERRNO):
            if VERBOSE:
                print(f'{client_address[0]} - connection dropped: {type(e).__name__} {e}')
            return
        super().handle_error(request, client_address)


def bind(port, explicit):
    for p in range(port, port + (1 if explicit else 20)):
        try:
            return Server(('0.0.0.0', p), Handler), p
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
            print(f'port {p} is in use (another server still running? LocalSH uses 8080, Joint FEA its own)'
                  + ('' if explicit else ', trying the next one'))
    sys.exit('no free port: stop the other server (Ctrl-C in its terminal) or pass --port')


def lan_addresses():
    addrs = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith('127.'):
                addrs.add(ip)
    except socket.gaierror:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('10.255.255.255', 1)); addrs.add(s.getsockname()[0]); s.close()
    except OSError:
        pass
    return sorted(addrs)


def lan_selftest(scheme, port, ips):
    time.sleep(0.5)
    ctx = ssl._create_unverified_context() if scheme == 'https' else None
    for ip in ips:
        url = f'{scheme}://{ip}:{port}/web/index.html'
        try:
            with urllib.request.urlopen(url, timeout=4, context=ctx) as r:
                ok, why = r.status == 200, ''
        except Exception as e:                                      # noqa: BLE001 - any failure is the answer
            ok, why = False, f'{type(e).__name__}: {e}'
        if ok:
            print(f'  network check:  {ip} answers')
        else:
            print(f'  network check:  {ip} does NOT answer from this Mac ({why}).\n'
                  '    Check: System Settings > Privacy & Security > Local Network (the terminal app), the firewall\n'
                  f'    (allow incoming connections for {sys.executable}), or a VPN that blocks the local network.')


def ensure_cert(path_pem):
    if os.path.exists(path_pem):
        return path_pem
    names = ','.join(['DNS:localhost', 'DNS:' + socket.gethostname()] + ['IP:' + ip for ip in lan_addresses()] + ['IP:127.0.0.1'])
    cmd = ['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-keyout', path_pem, '-out', path_pem, '-days', '825',
           '-subj', f'/CN={APP}', '-addext', 'subjectAltName=' + names]
    print('creating a self-signed certificate:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    return path_pem


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--port', type=int, default=None)
    ap.add_argument('--https', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    global VERBOSE
    VERBOSE = args.verbose
    httpd, port = bind(args.port or (8453 if args.https else 8090), explicit=args.port is not None)
    scheme = 'http'
    if args.https:
        pem = ensure_cert(os.path.join(HERE, PEM))
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.load_cert_chain(pem)
        httpd.tls = ctx; scheme = 'https'
    print(f'{APP} — serving {ROOT}')
    print(f'  this machine:   {scheme}://localhost:{port}/web/   <- open this one here (WebGPU needs localhost or https)')
    for ip in lan_addresses():
        print(f'  other devices:  {scheme}://{ip}:{port}/web/   (same Wi-Fi{"" if args.https else "; CPU solver only over plain http: use --https for WebGPU"})')
    print('Ctrl-C stops the server.')
    ips = lan_addresses()
    if ips:
        threading.Thread(target=lan_selftest, args=(scheme, port, ips), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nstopped')


if __name__ == '__main__':
    sys.exit(main())
