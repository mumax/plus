"""Images made on the server.

The display image's vector colours come from ``mumaxplus.util.get_rgb`` (reused, not
re-implemented), so they match ``mumaxplus.util.plot_field`` and the rest of mumax⁺;
a map of the regions gets a colour per region (region_rgba), and any other scalar
image the colours of its colormap (scalar_rgba).
The table plots are drawn in the page. Movie frames are drawn here, like the page draws
a display, so a recording does not need the page to be open.
"""

from __future__ import annotations

import colorsys
from functools import lru_cache
from pathlib import Path

import numpy as np

# the colormaps a display can use (matplotlib's names); the first is the default
COLORMAPS = (
    "gray",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "turbo",
    "coolwarm",
    "bwr",
    "RdBu_r",
    "seismic",
)
# these have a meaning at their middle: their range is symmetric, with 0 in the middle
DIVERGING = {"coolwarm", "bwr", "RdBu_r", "seismic"}


@lru_cache(maxsize=None)
def _lut(cmap):
    """Return the (256, 4) uint8 colours of a colormap."""
    import matplotlib

    return matplotlib.colormaps[cmap](np.linspace(0, 1, 256), bytes=True)


def colormap_stops(n=33):
    """Colours along each colormap, as "#rrggbb", for the page's colorbars."""
    at = np.linspace(0, 255, n).round().astype(int)
    return {
        name: ["#%02x%02x%02x" % tuple(c[:3]) for c in _lut(name)[at].tolist()]
        for name in COLORMAPS
    }


def vector_rgba(sl):
    """(3, h, w) float array -> (h, w, 4) uint8 using ``mumaxplus.util.get_rgb``.

    The vectors are normalised by the largest norm, and cells with a zero
    vector (outside the geometry) come out gray.
    """
    from mumaxplus.util import get_rgb

    field = np.ascontiguousarray(sl[:, None, :, :], dtype=float)  # (3, 1, h, w)
    if field.any():
        rgb = get_rgb(field, OoP_axis_idx=2, layer=0)  # (h, w, 3) in [0, 1]
    else:  # get_rgb's normalisation would divide 0 by 0
        rgb = np.full((*sl.shape[1:], 3), 0.5)
    out = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
    out[..., :3] = np.round(np.clip(rgb, 0, 1) * 255)
    out[..., 3] = 255
    return out


def region_rgba(regions):
    """(h, w) region indices -> (h, w, 4) uint8, a colour per region.

    The hues of successive indices are a golden angle apart, so neighbouring regions
    get clearly different colours. NaN (no cell there) is transparent.
    """
    out = np.zeros((*regions.shape, 4), dtype=np.uint8)
    for r in np.unique(regions[~np.isnan(regions)]):
        rgb = colorsys.hsv_to_rgb((r * 0.618034) % 1.0, 0.55, 0.9)
        out[regions == r] = [*(round(255 * c) for c in rgb), 255]
    return out


def scalar_rgba(v, lo, hi, cmap):
    """(h, w) floats -> (h, w, 4) uint8 in a colormap from ``lo`` to ``hi``.

    If ``lo`` equals ``hi``, everything gets the middle colour. NaN (no cell there) is
    transparent.
    """
    v = np.asarray(v, dtype=float)
    x = (v - lo) / (hi - lo) if hi > lo else np.full(v.shape, 0.5)
    out = _lut(cmap)[np.round(255 * np.clip(np.nan_to_num(x), 0, 1)).astype(np.uint8)]
    out[np.isnan(v), 3] = 0
    return out


# ------------------------------------------------------------------------- movies
MOVIE_BG = (232, 234, 239)  # outside the magnets, as the page's light theme


def _si_time(t):
    """Format a time with a unit: 1.234e-9 -> "1.234 ns"."""
    for unit, f in (("ps", 1e-12), ("ns", 1e-9), ("µs", 1e-6), ("ms", 1e-3)):
        if abs(t) < 1000 * f:
            return f"{t / f:.4g} {unit}"
    return f"{t:.4g} s"


def movie_frame(meta, data, rgba, arrow_cells=8, size=512):
    """One movie frame as a PIL image: the display's image, arrows, colorbar, and time.

    ``meta``, ``data`` and ``rgba`` are a display frame as sent to the page; the
    image is drawn like the page draws it, scaled up by a whole factor to about
    ``size`` pixels, and with pixels the shape of the cells.
    """
    from PIL import Image, ImageDraw, ImageFont

    w, h = meta["w"], meta["h"]
    img = np.array(rgba, dtype=np.uint8).reshape(h, w, 4)
    img = img[::-1]  # y up
    rgb = np.where(img[..., 3:] > 0, img[..., :3], np.array(MOVIE_BG, np.uint8))
    kx, ky = pixel_scale(meta, size, size)
    im = Image.fromarray(rgb).resize((w * kx, h * ky), Image.NEAREST)
    im = im.convert("RGBA")
    if meta["vector"] and arrow_cells:
        arrows = _arrows(data, meta["axes"][:2], w, h, kx, ky, arrow_cells)
        im = Image.alpha_composite(im, arrows)
    d = ImageDraw.Draw(im)
    font_size = max(12, im.width // 32)
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    text = "t = " + _si_time(meta["time"])
    box = d.textbbox((8, 6), text, font=font)
    d.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(255, 255, 255))
    d.text((8, 6), text, fill=(20, 20, 20), font=font)
    if meta["cmap"]:
        im = _with_colorbar(im, meta, font, font_size)
    return im.convert("RGB")


def _with_colorbar(im, meta, font, size):
    """Add the page's colorbar (drawColorbar) to the right of an image.

    ``size`` is the font's size. The width only depends on the font and the unit, so
    all frames of a movie have the same size.
    """
    from PIL import Image, ImageDraw

    ink = (20, 20, 20)
    bar, gap = max(14, size), size // 2
    d = ImageDraw.Draw(im)
    text_w = max(d.textlength(t, font=font) for t in ("-8.888e+88", meta["unit"]))
    width = im.width + gap + bar + gap + round(text_w) + gap
    height = max(im.height, 140)
    out = Image.new("RGBA", (width, height), (*MOVIE_BG, 255))
    out.paste(im, (0, 0))
    d = ImageDraw.Draw(out)
    x, top, bottom = im.width + gap, 2 * size, height - size
    at = np.linspace(255, 0, bottom - top).round().astype(int)
    colors = Image.fromarray(_lut(meta["cmap"])[at][:, None, :3])
    out.paste(colors.resize((bar, bottom - top), Image.NEAREST), (x, top))
    d.rectangle((x, top, x + bar, bottom), outline=ink)
    d.text((x, 0), meta["unit"], fill=ink, font=font)
    lo, hi = meta["vmin"], meta["vmax"]
    for f in (0.5,) if lo == hi else (0, 0.25, 0.5, 0.75, 1):  # one value: the middle
        y = bottom - f * (bottom - top)
        d.line((x + bar, y, x + bar + gap // 2, y), fill=ink)
        label = f"{lo + f * (hi - lo):.3e}"
        d.text((x + bar + gap, y - size // 2), label, fill=ink, font=font)
    return out


def pixel_scale(meta, max_w, max_h):
    """Whole scale factors (kx, ky) of an image's cells, as the page's drawFrame.

    The pixels get the shape of the cells (e.g. thin layers in an xz cut), and the
    image fits in about max_w × max_h.
    """
    w, h, (cx, cy) = meta["w"], meta["h"], meta["cell"]
    aspect = cy / cx if cx > 0 and cy > 0 else 1.0
    k = max(1, min(16, round(min(max_w / w, max_h / (h * aspect)))))
    return k, max(1, round(k * aspect))


def _arrows(d, comps, w, h, fx, fy, cells):
    """Draw the page's arrows (drawArrows: plot_field's quiver) on an RGBA layer.

    ``comps`` are the components of ``d`` along the image's horizontal and vertical.
    """
    from PIL import Image, ImageDraw

    layer = Image.new("RGBA", (w * fx, h * fy), (0, 0, 0, 0))
    S = max(cells, 6 / fx)
    nax, nay = max(int(w // S), 1), max(int(h // S), 1)
    sx, sy = w / nax, h / nay
    inside = ~np.isnan(d[0]) & np.any(np.nan_to_num(d) != 0, axis=0)
    arrows, maxip = [], 0.0
    for iy in range(nay):
        for ix in range(nax):
            cx, cy = int((ix + 0.5) * sx), int((iy + 0.5) * sy)
            if not inside[cy, cx]:
                continue
            ys = slice(int(iy * sy), int((iy + 1) * sy))
            xs = slice(int(ix * sx), int((ix + 1) * sx))
            on = inside[ys, xs]
            if not on.any():
                continue
            mx, my = (d[i][ys, xs][on].mean() for i in comps)
            ip = float(np.hypot(mx, my))
            maxip = max(maxip, ip)
            X, Y = (ix + 0.5) * sx * fx, (h - (iy + 0.5) * sy) * fy
            arrows.append((X, Y, mx, my, ip))
    if maxip == 0:
        return layer
    draw = ImageDraw.Draw(layer)
    sw, lmax = 0.005 * w * fx, 0.75 * min(sx * fx, sy * fy)
    for X, Y, mx, my, ip in arrows:
        L = ip / maxip * lmax
        if L == 0:
            continue
        u = sw * min(1, L / (5 * sw))  # shorter than its head: shrink the glyph
        c, s = mx / ip, -my / ip
        P = [(0, -0.5), (L / u - 4.5, -0.5), (L / u - 5, -1.5), (L / u, 0)]
        P += [(L / u - 5, 1.5), (L / u - 4.5, 0.5), (0, 0.5)]
        pts = []
        for px, py in P:
            px, py = px * u - L / 2, py * u
            pts.append((X + px * c - py * s, Y + px * s + py * c))
        draw.polygon(pts, fill=(0, 0, 0, 102))
    return layer


def write_movie(folder, path, fps):
    """Encode the frames 000000.png, 000001.png, ... in ``folder`` as an MP4 (ffmpeg).

    The MP4 is lossless (RGB, libx264rgb at crf 0): its pixels are the frames'.
    libx264rgb, not a YUV codec, so the colours of neighbouring cells do not bleed.
    """
    import shutil
    import subprocess

    frames = str(Path(folder) / "%06d.png")
    codec = ["-c:v", "libx264rgb", "-crf", "0", "-pix_fmt", "rgb24"]
    cmd = [shutil.which("ffmpeg"), "-y", "-loglevel", "error"]
    cmd += ["-framerate", f"{fps:g}", "-i", frames, *codec, str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
