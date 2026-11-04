"""Movies of a display, recorded by the simulation (with or without the page)."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

import numpy as np

from .common import _next_due
from .plots import movie_frame, write_movie


class MovieMixin:
    """The Controller's movie recording (one movie per display at most)."""

    def _init_movies(self):
        self.outputs = set()  # files the page may download (the MP4 movies)

    def _movie_start(self, cfg, c):
        """Record a movie of a display: a frame every ``period`` s of simulation time.

        Frames are drawn here (plots.movie_frame) and kept as PNG files until
        _movie_stop makes the movie, so it records with or without the page.
        """
        if cfg.get("_movie"):
            raise ValueError("this display is already recording")
        period = float(self._eval(c["period"]))
        fps = float(self._eval(c["fps"]))
        fmt = c["format"]
        if period <= 0 or fps <= 0:
            raise ValueError("the period and the frame rate must be positive")
        if fmt not in ("mp4", "png"):
            raise ValueError(f"unknown movie format {fmt!r}")
        if fmt == "mp4" and not shutil.which("ffmpeg"):
            raise ValueError("MP4 needs ffmpeg, which is not installed: use PNG frames")
        base = cfg["quant"] + (f"_{cfg['comp']}" if cfg["comp"] != "" else "")
        stem = next(
            s
            for s in (f"{base}_movie_{i:03d}" for i in range(1000))
            if not Path(s).exists() and not Path(f"{s}.{fmt}").exists()
        )
        if fmt == "png":  # the frames are the result
            out = folder = Path(stem).resolve()
            out.mkdir()
        else:
            out = Path(f"{stem}.{fmt}").resolve()
            folder = Path(tempfile.mkdtemp(prefix="mumaxplus_movie_"))
        cfg["_movie"] = {
            "period": period,
            "fps": fps,
            "format": fmt,
            "arrows": int(c["arrows"]),
            "folder": folder,
            "out": out,
            "n": 0,
            "next": None,
        }
        self.say(f"recording {base}: a frame every {period:g} s, into {out}")
        self._movie_tick()  # the first frame: now

    def _movie_tick(self):
        """Add a frame to the movies that are due (at the step nearest their time)."""
        t, dt = self._now(), float(self._ts().timestep or 0.0)
        for cfg in self.views:
            mv = cfg.get("_movie")
            if not mv:
                continue
            tol = 0.5 * min(dt, mv["period"])  # t has rounding errors
            if mv["next"] is not None and t < mv["next"] - tol:
                continue
            try:
                self._movie_frame(cfg, mv)
            except Exception as e:  # stop recording, keep what was recorded
                self.say(f"movie of {cfg['quant']}: {type(e).__name__}: {e}", "err")
                self._movie_end(cfg)
                continue
            due = t if mv["next"] is None else mv["next"]
            mv["next"] = _next_due(due, t + tol, mv["period"])

    def _movie_frame(self, cfg, mv):
        self._render_view(cfg)  # the display as it is now
        with self.lock:
            meta, buf = self.frames[cfg["id"]]
        n = meta["ncomp"] * meta["w"] * meta["h"]
        data = np.frombuffer(buf[: 4 * n], "<f4").reshape(
            meta["ncomp"], meta["h"], meta["w"]
        )
        rgba = np.frombuffer(buf[4 * n :], np.uint8) if meta["rgba"] else None
        img = movie_frame(meta, data, rgba, mv["arrows"])
        img.save(mv["folder"] / f"{mv['n']:06d}.png")
        mv["n"] += 1

    def _movie_stop(self, cfg):
        """Stop recording and make the movie; return its file (or folder)."""
        mv = cfg.pop("_movie", None)
        if mv is None:
            raise ValueError("this display is not recording")
        if mv["format"] == "png":
            self.say(f"saved {mv['n']} frames in {mv['out']}")
            return str(mv["out"])
        if mv["n"] == 0:
            shutil.rmtree(mv["folder"], ignore_errors=True)
            raise ValueError("no frames were recorded")
        try:
            self.busy = "making the movie"
            self._publish()
            write_movie(mv["folder"], mv["out"], mv["fps"])
        except Exception as e:  # the frames took a whole run to make: keep them
            why = (getattr(e, "stderr", b"") or b"").decode(errors="replace").strip()
            raise ValueError(
                f"could not make {mv['out']} ({why or e}); "
                f"its frames are kept in {mv['folder']}"
            ) from e
        finally:
            self.busy = ""
        shutil.rmtree(mv["folder"], ignore_errors=True)
        self.outputs.add(str(mv["out"]))
        self.say(f"saved {mv['out']} ({mv['n']} frames at {mv['fps']:g} fps)")
        return str(mv["out"])

    def _movie_end(self, cfg):
        """Stop recording; a failure is reported, and the simulation goes on."""
        try:
            self._movie_stop(cfg)
        except Exception as e:
            self.say(f"movie of {cfg['quant']}: {e}", "err")

    def _finish_movies(self):
        """Make the movies still recording when the loop ends (the page is gone)."""
        for cfg in self.views:
            if cfg.get("_movie"):
                try:
                    print(f"//saved {self._movie_stop(cfg)}")
                except Exception as e:
                    print(f"//movie of {cfg['quant']}: {e}")

    # ------------------------------------------------------------------- commands
    def _cmd_movie_start(self, c):
        self._movie_start(self._view(c["view"]), c)

    def _cmd_movie_stop(self, c):
        return {"ok": True, "file": self._movie_stop(self._view(c["view"]))}
