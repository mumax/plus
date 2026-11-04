"""Tiny token-protected HTTP server (standard library only) for the mumax⁺ GUI."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
import json
import secrets
import threading
import time
from urllib.parse import parse_qs, urlparse
import webbrowser

from .common import _clean
from .core import Controller
from .display import _check_view

FORBIDDEN = b"forbidden: missing or wrong token"
EVENT_GAP = 0.15  # minimum time (s) between two pushed updates of the page
KEEPALIVE = 15.0  # time (s) without updates after which a keepalive is sent


def _make_handler(ctrl: Controller, token: str):
    page = resources.files(__package__).joinpath("static/index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silence request logging to not flood terminal
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
            # as bytes: comparing str raises TypeError on a non-ASCII token
            return secrets.compare_digest(supplied.encode(), token.encode())

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            arg = lambda k, d="": (q.get(k) or [d])[0]
            if not self._authorized(q):
                return self._send(403, FORBIDDEN, "text/plain")
            if url.path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if url.path == "/api/events":  # the snapshot, pushed whenever it changes
                return self._events()
            if url.path == "/api/page":  # parameters, mesh, solver settings, ...
                return self._json(ctrl.desc)  # replaced whole, never changed in place
            if url.path == "/api/frame":  # the image data of one display (?view=id)
                with ctrl.lock:
                    frame = ctrl.frames.get(int(arg("view")))
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
                    lines = [m for m in ctrl.log if m["id"] > since]
                return self._json(lines)  # not holding the lock: a slow page
            if url.path == "/api/file":  # a movie made by the GUI
                name = arg("name")
                with ctrl.lock:
                    ok = name in ctrl.outputs
                if not ok:
                    return self._send(404, b"not found", "text/plain")
                with open(name, "rb") as f:
                    body = f.read()
                return self._send(200, body, "video/mp4")
            if url.path == "/api/table.txt":  # the table rows in memory
                text = ctrl.table.text()  # the table has its own lock
                return self._send(200, text.encode(), "text/plain; charset=utf-8")
            if url.path == "/api/table":  # new rows for the plots (drawn in the page)
                ver, since = int(arg("version", "-1")), int(arg("since", "0"))
                return self._json(ctrl.table.since(ver, since))
            self._send(404, b"not found", "text/plain")

        def _events(self):
            """Server-sent events: the snapshot whenever it changes.

            At most one event per EVENT_GAP (a run changes it all the time), always
            the newest snapshot, and a comment now and then to keep the connection.
            """
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")  # the stream ends with it
            self.end_headers()
            sid = 0  # the snapshot the page has: none yet
            try:
                while (new := ctrl.wait_snapshot(sid, KEEPALIVE)) is not None:
                    if new[0] == sid:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    sid, snap = new
                    data = json.dumps(snap, allow_nan=False).encode()
                    self.wfile.write(b"data: " + data + b"\n\n")
                    self.wfile.flush()
                    time.sleep(EVENT_GAP)
            except OSError:  # the page went away
                pass

        def do_POST(self):
            url = urlparse(self.path)
            if not self._authorized(parse_qs(url.query)):
                return self._send(403, FORBIDDEN, "text/plain")
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
    """Web GUI for a mumax⁺ world.

    Parameters
    ----------
    world : mumaxplus.World
    namespace : dict, optional
        Namespace of the console, typically ``globals()`` of your script, so that you
        can type ``magnet.alpha = 0.1`` (and commands like ``Run(1e-9)``) in the GUI
        console.
    title : str, optional
        Page title (default: the name of the script).
    host, port : where to listen. The default only accepts local connections; the first
        free port from ``port`` on is used.
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
        self._loop = None  # the simulation thread of run(block=False)
        # replaced the default plot / display yet?
        self._script_plots = self._script_views = False

    # --------------------------------------------------------- set up from the script
    def table_add(self, quantity, name=None, unit=None):
        """Add a column to the table, to plot and save it.

        The table is saved in table.txt. ``quantity`` is a FieldQuantity (its average
        is added, one column per component), a ScalarQuantity, the name of a quantity
        of the selected magnet, or any function without arguments that returns a
        number, e.g. ``lambda: magnet.magnetization.average()[0] ** 2``.
        ``name`` and ``unit`` label the column (needed to name a function's column
        nicely).
        """
        self.ctrl.call(lambda: self.ctrl.table_add(quantity, name, unit))

    def table_autosave(self, period):
        """Save a table row in table.txt every ``period`` s of simulation time.

        0 = keep only a live preview in the page.
        """
        self.ctrl.call(lambda: self.ctrl._table_autosave(period))

    def add_plot(self, x="t", y=None, logx=False, logy=False):
        """Show a plot of table column(s) ``y`` against column ``x``.

        Columns are names or 1-based numbers; ``y`` can be several, as a list or
        comma-separated (``"mx,my,mz"``). ``logx`` / ``logy`` use logarithmic axes.
        The first call replaces the default (empty) plot; later calls add plots.
        """

        def add():
            if not self._script_plots:
                self.ctrl._plot_set(self.ctrl.plots[0], x, y, logx, logy)
                self._script_plots = True
            else:
                self.ctrl.add_plot(x, y, logx, logy)

        self.ctrl.call(add)

    def add_display(
        self, quantity, comp="", layer=None, all=False, plane="xy", cmap="gray"
    ):
        """Show a display of ``quantity``.

        ``quantity`` is a quantity name of the selected magnet, e.g.
        ``"effective_field"``; ``comp`` is "" (vector), "x", "y", "z" or a component
        number. ``plane`` is "xy", or "xz" or "yz" for a cross-section; ``layer`` is
        the slice along the axis normal to it (default: the bottom layer of "xy", the
        middle of the others). ``cmap`` is the colormap of a scalar image, e.g.
        "viridis", or "coolwarm" (these diverging maps have 0 in the middle); see the
        display's Colormap menu for the list. The first call replaces the default
        display (magnetization); later calls add displays.
        """
        cfg = {
            "quant": quantity,
            "comp": comp,
            "plane": plane,
            "layer": layer,
            "all": all,
            "cmap": cmap,
        }
        _check_view(cfg)

        def add():
            if not self._script_views:
                self.ctrl.views[0].update(cfg)
                self.ctrl.views[0].pop("_err", None)
                self.ctrl._dirty = True
                self._script_views = True
            else:
                self.ctrl.add_view(**cfg)

        self.ctrl.call(add)

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
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        print(f"//entering interactive mode: open {self.url}")
        if self.open_browser:
            webbrowser.open(self.url)

    def run(self, block=True):
        """Serve the GUI and drive the simulation.

        With ``block=True`` (default) the simulation loop runs on the calling thread
        until Ctrl-C or :meth:`stop`. With ``block=False`` it runs in a background
        thread (handy in Jupyter). Only one loop can run: with ``block=False``, a loop
        already in the background is returned. After :meth:`stop` it can run again.
        """
        if self._loop is not None and self._loop.is_alive():
            if block:
                raise RuntimeError("the simulation already runs: call stop() first")
            return self._loop
        self.ctrl._stop = False
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
        if self._loop is not None and self._loop is not threading.current_thread():
            self._loop.join()  # it may still make a movie
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
