"""Time the skyrmion racetrack simulation without and with the GUI.

    python mumaxplus/gui/examples/benchmark_gui.py
    python mumaxplus/gui/examples/benchmark_gui.py --duration 1e-9 --repeats 5 --client

Each repeat builds a fresh world (same setup as examples/skyrmionracetrack.py),
minimizes it (not timed) and then times only the time evolution:

  * plain: world.timesolver.run(duration)
  * GUI:   the same run, started with the GUI's Run button command and driven by the
           GUI loop (chunked steps, display rendering, torque updates, publishing)

With --client a background thread polls the HTTP server like an open browser tab
(/api/tick, /api/frame, /api/log), so serving the page is included in the cost.
"""

import argparse
import json
import statistics
import threading
import time
import urllib.request

from mumaxplus import Ferromagnet, Grid, World
from mumaxplus.gui import GUI
from mumaxplus.util import neelskyrmion


def make_racetrack():
    """Create the world and magnet that are timed."""
    world = World((1e-9, 1e-9, 0.4e-9))
    magnet = Ferromagnet(world, Grid(size=(128, 64, 1)))
    magnet.enable_demag = False
    magnet.msat = 580e3
    magnet.aex = 15e-12
    magnet.ku1 = 0.8e6
    magnet.anisU = (0, 0, 1)
    magnet.alpha = 0.2
    magnet.dmi_tensor.set_interfacial_dmi(3.2e-3)
    magnet.magnetization = neelskyrmion(
        position=(64e-9, 32e-9, 0), radius=5e-9, charge=-1, polarization=1
    )
    magnet.minimize()

    # add a current
    magnet.xi = 0.3
    magnet.jcur = (1e12, 0, 0)
    magnet.pol = 0.4
    return world, magnet


def sync(magnet):
    """Wait for the GPU: copying a value to the host forces queued kernels to finish."""
    return magnet.magnetization.average()


def time_plain(duration):
    """Time a run without the GUI."""
    world, magnet = make_racetrack()
    sync(magnet)
    t0 = time.perf_counter()
    world.timesolver.run(duration)
    sync(magnet)
    return time.perf_counter() - t0, None, magnet.magnetization.average()


def browser_client(gui, stop):
    """Poll the server the way index.html does (150 ms while running)."""
    def get(path):
        req = urllib.request.Request(
            f"http://{gui.host}:{gui.port}{path}", headers={"X-Token": gui.token}
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.read()

    get("/")
    get("/api/page")
    last_frames, last_log = {}, None
    while not stop.is_set():
        tick = json.loads(get("/api/tick"))
        for v in tick.get("views", []):  # one image per display panel
            if last_frames.get(v["id"]) != v["frame_id"]:
                last_frames[v["id"]] = v["frame_id"]
                get(f"/api/frame?view={v['id']}")
        if tick.get("log_id") != last_log:
            last_log = tick.get("log_id")
            get("/api/log?since=0")
        stop.wait(0.15 if tick.get("running") else 0.3)


def time_gui(duration, client):
    """Time the same run driven by the GUI, optionally with a polling client."""
    world, magnet = make_racetrack()
    gui = GUI(world, namespace={"world": world, "magnet": magnet})
    gui.start_server()
    loop = gui.run(block=False)

    stop = threading.Event()
    poller = None
    if client:
        poller = threading.Thread(target=browser_client, args=(gui, stop), daemon=True)
        poller.start()
        time.sleep(0.5)  # let the "page" load before starting

    ctrl = gui.ctrl
    sync(magnet)
    t0 = time.perf_counter()
    # returns once the run has started
    ctrl.submit({"cmd": "run", "text": repr(duration)})
    while ctrl.running:
        time.sleep(0.001)
    elapsed = time.perf_counter() - t0
    # the GUI loop syncs implicitly (it reads the time and renders), so stop it before
    # touching the magnet from this thread
    stop.set()
    gui.stop()
    loop.join()
    if poller is not None:
        poller.join()
    return elapsed, ctrl.steps_done, magnet.magnetization.average()


def main():
    """Compare the run time with and without the GUI."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--duration", type=float, default=6e-10, help="simulated time per run (s)"
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--client",
        action="store_true",
        help="also poll the server like an open browser tab",
    )
    args = p.parse_args()

    print("warming up (CUDA init, first kernel launches)...")
    time_plain(1e-11)

    plain, gui = [], []
    # alternate, so drifts (GPU clocks, ...) hit both equally
    for i in range(args.repeats):
        t, _, m_plain = time_plain(args.duration)
        plain.append(t)
        print(f"[{i + 1}/{args.repeats}] plain: {t:8.3f} s   <m> = {m_plain}")
        t, steps, m_gui = time_gui(args.duration, args.client)
        gui.append(t)
        print(
            f"[{i + 1}/{args.repeats}] GUI:   {t:8.3f} s   <m> = {m_gui}   "
            f"({steps} steps)"
        )

    mp, mg = statistics.median(plain), statistics.median(gui)
    print(f"\nsimulated time {args.duration:g} s, {args.repeats} repeats, "
          f"browser client {'on' if args.client else 'off'}")
    print(f"plain median: {mp:.3f} s")
    print(f"GUI   median: {mg:.3f} s")
    print(f"GUI overhead: {mg - mp:+.3f} s ({100 * (mg / mp - 1):+.1f} %)")


if __name__ == "__main__":
    main()
