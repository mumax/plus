"""Images made on the server.

The display image's vector colours come from ``mumaxplus.util.get_rgb`` (reused, not
re-implemented), so they match ``mumaxplus.util.plot_field`` and the rest of mumax+.
The table plots are drawn in the page. Movie frames are drawn here, like the page draws
a display, so a recording does not need the page to be open.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


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


# ------------------------------------------------------------------------- movies
MOVIE_BG = (232, 234, 239)  # outside the magnets, as the page's light theme


def _si_time(t):
    """Format a time with a unit: 1.234e-9 -> "1.234 ns"."""
    for unit, f in (("ps", 1e-12), ("ns", 1e-9), ("µs", 1e-6), ("ms", 1e-3)):
        if abs(t) < 1000 * f:
            return f"{t / f:.4g} {unit}"
    return f"{t:.4g} s"


def movie_frame(meta, data, rgba, arrow_cells=8, size=512):
    """One movie frame as a PIL image: the display's image, arrows, and the time.

    ``meta``, ``data`` and ``rgba`` are a display frame as sent to the page; the
    image is drawn like the page draws it (vector colours, or gray from the
    minimum to the maximum), scaled up by a whole factor to about ``size`` pixels.
    """
    from PIL import Image, ImageDraw, ImageFont

    w, h = meta["w"], meta["h"]
    if meta["vector"]:
        img = np.array(rgba, dtype=np.uint8).reshape(h, w, 4)
    else:
        lo, hi = meta["vmin"], meta["vmax"]
        if lo == hi:  # mumax3: gray instead of black
            lo, hi = lo - 1, hi + 1
        v = data[0]
        g = np.round(255 * np.clip((np.nan_to_num(v) - lo) / (hi - lo), 0, 1))
        img = np.empty((h, w, 4), np.uint8)
        img[..., :3] = g[..., None]
        img[..., 3] = np.where(np.isnan(v), 0, 255)
    img = img[::-1]  # y up
    rgb = np.where(img[..., 3:] > 0, img[..., :3], np.array(MOVIE_BG, np.uint8))
    k = max(1, min(16, round(size / max(w, h))))
    im = Image.fromarray(rgb.astype(np.uint8)).resize((w * k, h * k), Image.NEAREST)
    im = im.convert("RGBA")
    if meta["vector"] and arrow_cells:
        im = Image.alpha_composite(im, _arrows(data, w, h, k, arrow_cells))
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=max(12, im.width // 32))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    text = "t = " + _si_time(meta["time"])
    box = d.textbbox((8, 6), text, font=font)
    d.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(255, 255, 255))
    d.text((8, 6), text, fill=(20, 20, 20), font=font)
    return im.convert("RGB")


def _arrows(d, w, h, f, cells):
    """Draw the page's arrows (drawArrows: plot_field's quiver) on an RGBA layer."""
    from PIL import Image, ImageDraw

    layer = Image.new("RGBA", (w * f, h * f), (0, 0, 0, 0))
    S = max(cells, 6 / f)
    nax, nay = max(int(w // S), 1), max(int(h // S), 1)
    sx, sy = w / nax, h / nay
    inside = ~np.isnan(d[0]) & np.any(np.nan_to_num(d) != 0, axis=0)
    arrows, maxip = [], 0.0
    for iy in range(nay):
        for ix in range(nax):
            cx, cy = int((ix + 0.5) * sx), int((iy + 0.5) * sy)
            if not inside[cy, cx]:
                continue
            ys, xs = slice(int(iy * sy), int((iy + 1) * sy)), slice(
                int(ix * sx), int((ix + 1) * sx)
            )
            on = inside[ys, xs]
            if not on.any():
                continue
            mx, my = d[0][ys, xs][on].mean(), d[1][ys, xs][on].mean()
            ip = float(np.hypot(mx, my))
            maxip = max(maxip, ip)
            arrows.append(((ix + 0.5) * sx * f, (h - (iy + 0.5) * sy) * f, mx, my, ip))
    if maxip == 0:
        return layer
    draw = ImageDraw.Draw(layer)
    sw, lmax = 0.005 * w * f, 0.75 * min(sx, sy) * f
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


def write_movie(frames, path, fps, fmt):
    """Encode the PNG files ``frames`` as a GIF (Pillow) or an MP4 (ffmpeg).

    The MP4 is lossless (RGB, libx264rgb at crf 0): its pixels are the frames'. A
    GIF has at most 256 colours, so its frames share one palette (made from all of
    them) without dithering: a cell keeps its colour from frame to frame instead of
    flickering between nearby palette colours.
    """
    if fmt == "gif":
        from PIL import Image

        ims = [Image.open(f).convert("RGB") for f in frames]
        step = max(1, len(ims) // 64)  # the palette from up to 64 frames
        sample = ims[::step]
        mosaic = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
        for i, im in enumerate(sample):
            mosaic.paste(im, (0, i * im.height))
        pal = mosaic.quantize(256, method=Image.Quantize.MEDIANCUT)
        first, *rest = (
            im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims
        )
        first.save(
            path,
            save_all=True,
            append_images=rest,
            duration=round(1000 / fps),
            loop=0,
            optimize=False,  # keep the shared palette
        )
    else:
        import shutil
        import subprocess

        folder = Path(frames[0]).parent
        subprocess.run(
            [
                shutil.which("ffmpeg"),
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                f"{fps:g}",
                "-i",
                str(folder / "%06d.png"),
                "-c:v",
                "libx264rgb",  # RGB, not YUV: no colour bleeding between cells
                "-crf",
                "0",  # lossless
                "-pix_fmt",
                "rgb24",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
