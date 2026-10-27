"""Tiny token-protected HTTP server (standard library only) for the mumax+ GUI."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
import json
import os
import secrets
import threading
from urllib.parse import parse_qs, urlparse
import webbrowser

from .core import Controller, _clean


def _make_handler(ctrl: Controller, token: str):
    page = resources.files(__package__).joinpath("static/index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silence request logging
            pass

        def _send(self, code, body=b"", ctype="application/json", headers=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(_clean(obj), allow_nan=False).encode())

        def _authorized(self, query):
            supplied = self.headers.get("X-Token") or (query.get("token") or [""])[0]
            return secrets.compare_digest(supplied, token)

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            arg = lambda k, d="": (q.get(k) or [d])[0]
            if not self._authorized(q):
                return self._send(
                    403, b"forbidden: missing or wrong token", "text/plain"
                )
            if url.path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if url.path == "/api/tick":  # what mumax3 refreshes on every update
                return self._json(ctrl.snapshot())
            if url.path == "/api/page":  # parameters, mesh, solver settings, ...
                with ctrl.lock:
                    return self._json(ctrl.desc)
            if url.path == "/api/frame":  # the image data of one display (?view=id)
                vid = arg("view")
                with ctrl.lock:
                    frame = (
                        ctrl.frames.get(int(vid))
                        if vid
                        else next(iter(ctrl.frames.values()), None)
                    )
                if frame is None:
                    return self._send(204)
                meta, data = frame
                return self._send(
                    200,
                    data,
                    "application/octet-stream",
                    {"X-Frame-Meta": json.dumps(meta)},
                )
            if url.path == "/api/log":
                since = int(arg("since", "0"))
                with ctrl.lock:
                    return self._json([m for m in ctrl.log if m["id"] > since])
            if url.path == "/api/file":  # a movie made by the GUI
                name = arg("name")
                with ctrl.lock:
                    ok = name in ctrl.outputs
                if not ok:
                    return self._send(404, b"not found", "text/plain")
                with open(name, "rb") as f:
                    body = f.read()
                kind = {".mp4": "video/mp4"}.get(
                    os.path.splitext(name)[1], "application/octet-stream"
                )
                return self._send(200, body, kind)
            if url.path == "/api/table.txt":  # the table rows in memory
                with ctrl.lock:
                    text = ctrl.table.text()
                return self._send(200, text.encode(), "text/plain; charset=utf-8")
            if url.path == "/api/table":  # new rows for the plots (drawn in the page)
                ver, since = int(arg("version", "-1")), int(arg("since", "0"))
                return self._json(ctrl.table.since(ver, since))
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            url = urlparse(self.path)
            if not self._authorized(parse_qs(url.query)):
                return self._send(403, b"forbidden", "text/plain")
            if url.path != "/api/cmd":
                return self._send(404, b"not found", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", 0))
                cmd = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json({"ok": False, "error": "bad json"}, 400)
            self._json(ctrl.submit(cmd))

    return Handler


class GUI:
    """Web GUI for a mumax+ world, modelled on the mumax3 GUI.

    Parameters
    ----------
    world : mumaxplus.World
    namespace : dict, optional
        Namespace of the console, typically ``globals()`` of your script, so that you
        can type ``magnet.alpha = 0.1`` (and mumax3-style ``Run(1e-9)``) in the GUI
        console.
    title : str, optional
        Page title (default: the name of the script).
    host, port : where to listen. The default only accepts local connections; the first
        free port from ``port`` on is used (mumax3 also starts at 35367).
    open_browser : bool
        Open the page in the default browser.
    """

    def __init__(
        self,
        world,
        namespace=None,
        title=None,
        host="127.0.0.1",
        port=35367,
        open_browser=False,
    ):
        self.ctrl = Controller(world, namespace, title)
        self.token = secrets.token_urlsafe(16)
        self.host, self.port, self.open_browser = host, port, open_browser
        self._httpd = None
        self._thread = None
        self._loop = None  # the simulation thread of run(block=False)
        self._script_plots = self._script_views = (
            False  # replaced the default plot / display yet?
        )

    # --------------------------------------------------------- set up from the script
    def table_add(self, quantity, name=None, unit=None):
        """Add a column to the table (mumax3: TableAdd), to plot and save it.

        The table is saved in table.txt. ``quantity`` is a FieldQuantity (its average
        is added, one column per component), a ScalarQuantity, the name of a quantity
        of the selected magnet, or any function without arguments that returns a
        number, e.g. ``lambda: magnet.magnetization.average()[0] ** 2``.
        ``name`` and ``unit`` label the column (needed to name a function's column
        nicely).
        """
        self._do(lambda: self.ctrl.table_add(quantity, name, unit))

    def table_autosave(self, period):
        """Save a table row in table.txt every ``period`` s of simulation time.

        mumax3: TableAutosave. 0 = keep only a live preview in the page.
        """
        self._do(lambda: self.ctrl._table_autosave(period))

    def add_plot(self, x="t", y=None, logx=False, logy=False):
        """Show a plot of table column(s) ``y`` against column ``x``.

        Columns are names or 1-based numbers; ``y`` can be several, as a list or
        comma-separated (``"mx,my,mz"``). ``logx`` / ``logy`` use logarithmic axes.
        The first call replaces the default plot (mx against t); later calls add plots.
        """

        def add():
            if not self._script_plots:
                self.ctrl._plot_set(self.ctrl.plots[0], x, y, logx, logy)
                self._script_plots = True
            else:
                self.ctrl.add_plot(x, y, logx, logy)

        self._do(add)

    def add_display(self, quantity, comp="", layer=0, all=False):
        """Show a display of ``quantity``.

        ``quantity`` is a quantity name of the selected magnet, e.g.
        ``"effective_field"``; ``comp`` is "" (vector), "x", "y", "z" or a component
        number. The first call replaces the default display (magnetization); later
        calls add displays.
        """
        cfg = {"quant": quantity, "comp": comp, "layer": layer, "all": all}

        def add():
            if not self._script_views:
                self.ctrl.views[0].update(cfg)
                self.ctrl.views[0].pop("_err", None)
                self.ctrl._dirty = True
                self._script_views = True
            else:
                self.ctrl.add_view(**cfg)

        self._do(add)

    def _do(self, fn):
        """Run ``fn`` now, or on the simulation thread if run(block=False) drives it."""
        if self._loop is None or not self._loop.is_alive():
            return fn()
        r = self.ctrl.submit({"cmd": "call", "fn": fn}, wait=60.0)
        if r.get("ok") is False:
            raise RuntimeError(r.get("error"))
        return r.get("value")

    @property
    def url(self):
        """URL of the page, including the access token."""
        return f"http://{self.host}:{self.port}/?token={self.token}"

    def start_server(self):
        """Start the web server on the first free port from ``port`` on."""
        handler = _make_handler(self.ctrl, self.token)
        for port in range(self.port, self.port + 50):
            try:
                self._httpd = ThreadingHTTPServer((self.host, port), handler)
                self.port = port
                break
            except OSError:
                continue
        else:
            raise OSError("no free port found")
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        print(f"//entering interactive mode: open {self.url}")
        if self.open_browser:
            webbrowser.open(self.url)

    def run(self, block=True):
        """Serve the GUI and drive the simulation (like ``mumax3 -i``).

        With ``block=True`` (default) the simulation loop runs on the calling thread
        until Ctrl-C or :meth:`stop`. With ``block=False`` it runs in a background
        thread (handy in Jupyter).
        """
        if self._httpd is None:
            self.start_server()
        if not block:
            t = self._loop = threading.Thread(target=self.ctrl.loop, daemon=True)
            t.start()
            return t
        try:
            self.ctrl.loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        """Stop the simulation loop and the web server."""
        self.ctrl.stop()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


def serve(world, **kwargs):
    """Shortcut for ``GUI(world, **kwargs).run()``."""
    GUI(world, **kwargs).run()
