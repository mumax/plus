"""Images made on the server.

The display image's vector colours come from ``mumaxplus.util.get_rgb`` (reused, not
re-implemented), so they match ``mumaxplus.util.plot_field`` and the rest of mumax+.
The table plots are drawn in the page.
"""

from __future__ import annotations

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
