"""Images made on the server.

* The display image's vector colours come from ``mumaxplus.util.get_rgb`` (reused, not
  re-implemented), so they match ``mumaxplus.util.plot_field`` and the rest of mumax+.
* ``mumaxplus.util`` has no time-series plot, so :func:`table_png` is ours. mumax3 draws
  this plot with gnuplot (``plot "table.txt" using x:y with lines``); we use matplotlib.

Figures use ``matplotlib.figure.Figure`` rather than ``pyplot``: pyplot has global
state and a global backend and may open windows, which we must not do inside the
user's process.
"""

from __future__ import annotations

import io
import threading

import numpy as np

lock = threading.RLock()  # matplotlib is not thread-safe

THEMES = {
    "light": dict(
        bg="#ffffff",
        face="#ffffff",
        ink="#1b1f27",
        mute="#667085",
        grid="#d9dde5",
        line="#2557d6",
    ),
    "dark": dict(
        bg="#171a21",
        face="#171a21",
        ink="#e6e9ef",
        mute="#98a2b3",
        grid="#2c323d",
        line="#6c93ff",
    ),
}


def vector_rgba(sl):
    """(3, h, w) float array -> (h, w, 4) uint8 using ``mumaxplus.util.get_rgb``.

    Like mumax3, the vectors are normalised by the largest norm, and cells with a zero
    vector (outside the geometry) come out gray.
    """
    from mumaxplus.util import get_rgb

    field = np.ascontiguousarray(sl[:, None, :, :], dtype=float)  # (3, 1, h, w)
    rgb = get_rgb(field, OoP_axis_idx=2, layer=0)  # (h, w, 3) in [0, 1]
    out = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
    out[..., :3] = np.round(np.clip(rgb, 0, 1) * 255)
    out[..., 3] = 255
    return out


def _fig(w, h, dpr, theme):
    from matplotlib.figure import Figure

    dpi = 100 * dpr
    t = THEMES[theme]
    return (
        Figure(
            figsize=(max(w, 80) / dpi, max(h, 60) / dpi), dpi=dpi, facecolor=t["bg"]
        ),
        t,
    )


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    return buf.getvalue()


def message_png(text, w, h, dpr=1, theme="light"):
    """Render ``text`` as a PNG image of w x h (CSS) pixels."""
    with lock:
        fig, t = _fig(w, h, dpr, theme)
        fig.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            color=t["mute"],
            fontsize=9,
            wrap=True,
        )
        return _png(fig)


def table_png(names, units, rows, xi, yi, w, h, dpr=1, theme="light"):
    """``plot "table.txt" using xi:yi with lines`` for the in-memory table."""
    from matplotlib.ticker import FormatStrFormatter

    if len(rows) < 2:
        return message_png("no data yet: run the simulation", w, h, dpr, theme)
    a = np.asarray(rows, dtype=float)
    with lock:
        fig, t = _fig(w, h, dpr, theme)
        ax = fig.add_subplot()
        ax.set_facecolor(t["face"])
        for s in ax.spines.values():
            s.set_color(t["grid"])
        ax.tick_params(colors=t["mute"], labelsize=8)
        ax.plot(a[:, xi], a[:, yi], color=t["line"], lw=1.4)
        ax.xaxis.set_major_formatter(
            FormatStrFormatter("%g")
        )  # gnuplot: set format x "%g"
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.grid(True, color=t["grid"], lw=0.6)
        label = lambda i: f"{names[i]} ({units[i]})" if units[i] else names[i]
        ax.set_xlabel(label(xi), fontsize=9, color=t["mute"])
        ax.set_ylabel(label(yi), fontsize=9, color=t["mute"])
        fig.tight_layout()
        return _png(fig)
