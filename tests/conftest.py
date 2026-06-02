"""Shared simulator helpers for warpSLAM tests.

Provides:
    - load_map(yaml_path)              : load a PGM/PNG occupancy grid + metadata
    - extract_centerline(free)         : skeleton-based loop extraction (scipy/skimage)
    - resample_centerline(xy, step)    : arc-length resampling
    - simulate_scan(occ, pose, ...)    : vectorised lidar ray-cast
    - wrap(a)                          : angle wrap to [-pi, pi]
    - SCAN config constants matching a 270 deg / 1080 beam UST-10LX

Heavy dependencies (scipy, skimage, pyyaml, PIL) are imported lazily so the
synthetic test in test_e2e.py keeps running even when only numpy + warp are
installed.
"""
from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

# Make the package importable from tests/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# UST-10LX-like config (matches the F1Tenth lidar the SLAM is tuned for).
N_BEAMS = 1080
ANGLE_MIN = float(-np.radians(135.0))
ANGLE_INC = float(np.radians(270.0) / (N_BEAMS - 1))
SCAN_MAX_RANGE = 30.0


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def maps_dir() -> Path | None:
    """Locate the sibling warporacer/maps directory or honour $WARPSLAM_MAPS."""
    env = os.environ.get("WARPSLAM_MAPS")
    if env:
        p = Path(env)
        return p if p.is_dir() else None
    here = Path(__file__).resolve()
    cand = here.parent.parent.parent / "warporacer" / "maps"
    return cand if cand.is_dir() else None


def load_map(yaml_path):
    """Return dict with keys: occ (HxW bool walls), free (HxW bool), res, origin, h, w."""
    import yaml  # type: ignore
    from PIL import Image  # type: ignore

    yp = Path(yaml_path)
    meta = yaml.safe_load(yp.read_text())
    img = np.asarray(Image.open(yp.parent / meta["image"]).convert("L"))
    free = img >= 230  # matches warporacer's free-space threshold
    occ = ~free
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
    return dict(occ=occ, free=free, res=float(meta["resolution"]),
                origin=(ox, oy), h=img.shape[0], w=img.shape[1])


def _neighbours(skel, r, c):
    h, w = skel.shape
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and skel[nr, nc]:
                out.append((nr, nc))
    return out


def extract_centerline(free, smooth_window=51):
    """Largest closed loop in the skeleton of the free space.

    Returns world-frame (x, y) waypoints in walking order. Mirrors the
    centerline pipeline in warporacer/main.py (skeletonize -> prune endpoints
    -> keep largest CC -> BFS for longest path between two neighbours of an
    interior point -> Savitzky-Golay smoothing).
    """
    from scipy.ndimage import convolve, label  # type: ignore
    from scipy.signal import savgol_filter  # type: ignore
    from skimage.morphology import skeletonize  # type: ignore

    skel = skeletonize(free)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    while True:
        counts = convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
        ends = skel & (counts <= 1)
        if not ends.any():
            break
        skel &= ~ends
    labs, n = label(skel, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        raise RuntimeError("empty skeleton")
    sizes = np.bincount(labs.ravel())
    sizes[0] = 0
    skel = labs == int(np.argmax(sizes))

    pts = np.argwhere(skel)
    if len(pts) == 0:
        raise RuntimeError("empty largest component")
    centroid = pts.mean(0)
    order = np.argsort(((pts - centroid) ** 2).sum(1))

    best = None
    for k in range(min(16, len(order))):
        start = tuple(int(v) for v in pts[order[k]])
        nbrs = _neighbours(skel, *start)
        if len(nbrs) < 2:
            continue
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                src, tgt = nbrs[i], nbrs[j]
                parent = {src: src}
                q = deque([src])
                while q:
                    u = q.popleft()
                    if u == tgt:
                        break
                    for v in _neighbours(skel, *u):
                        if v in parent or v == start:
                            continue
                        parent[v] = u
                        q.append(v)
                if tgt not in parent:
                    continue
                path = [start]
                cur = tgt
                while cur != src:
                    path.append(cur)
                    cur = parent[cur]
                path.append(src)
                path.reverse()
                if best is None or len(path) > len(best):
                    best = path
    if best is None:
        raise RuntimeError("no loop found")

    rc = np.array(best)
    win = min(smooth_window, len(rc) - (1 - len(rc) % 2))
    if win >= 5 and win % 2 == 1:
        rc = savgol_filter(rc.astype(float), win, 3, axis=0, mode="wrap")
    return rc  # (n, 2) in row, col (image) coords


def rc_to_world(rc, m):
    x = m["origin"][0] + rc[:, 1] * m["res"]
    y = m["origin"][1] + (m["h"] - 1 - rc[:, 0]) * m["res"]
    return np.column_stack([x, y]).astype(np.float32)


def recenter_map(m, cl_xy, half_extent):
    """Shift the map + centerline so cl_xy fits inside [-half_extent, +half_extent].

    The SLAM occupancy grid is fixed (2048 cells x 5 cm = 102.4 m square centered
    at the SLAM origin). Real F1Tenth maps are stored in arbitrary world frames
    that often fall outside that window. Translating both the map origin and the
    centerline by the same vector is the cleanest way to test on any map without
    growing the SLAM grid.
    """
    cx = float((cl_xy[:, 0].min() + cl_xy[:, 0].max()) * 0.5)
    cy = float((cl_xy[:, 1].min() + cl_xy[:, 1].max()) * 0.5)
    if max(cl_xy[:, 0].max() - cx, cx - cl_xy[:, 0].min(),
           cl_xy[:, 1].max() - cy, cy - cl_xy[:, 1].min()) > half_extent - 5.0:
        return None, None  # too big even after centering
    m2 = dict(m, origin=(m["origin"][0] - cx, m["origin"][1] - cy))
    cl2 = cl_xy.copy()
    cl2[:, 0] -= cx
    cl2[:, 1] -= cy
    return m2, cl2


def resample_centerline(xy, step=0.10):
    """Resample a closed polyline at uniform arc-length spacing."""
    diff = np.diff(xy, axis=0, append=xy[:1])
    seg = np.linalg.norm(diff, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    n = max(8, int(L / step))
    targets = np.linspace(0.0, L, n, endpoint=False)
    xy_loop = np.vstack([xy, xy[:1]])
    x = np.interp(targets, s, xy_loop[:, 0])
    y = np.interp(targets, s, xy_loop[:, 1])
    out = np.column_stack([x, y]).astype(np.float32)
    tan = np.diff(out, axis=0, append=out[:1])
    th = np.arctan2(tan[:, 1], tan[:, 0]).astype(np.float32)
    return np.column_stack([out, th]).astype(np.float32)


def simulate_scan(occ, pose, n_beams=N_BEAMS, a_min=ANGLE_MIN, a_inc=ANGLE_INC,
                  max_range=SCAN_MAX_RANGE, res=0.05, origin=(0.0, 0.0)):
    """Brute-force ray-cast: per-beam stepping until wall hit or max range."""
    angles = a_min + a_inc * np.arange(n_beams, dtype=np.float32) + pose[2]
    cs, sn = np.cos(angles), np.sin(angles)
    h, w = occ.shape
    ox, oy = origin
    step = res * 0.5
    n_steps = int(max_range / step)
    ranges = np.zeros(n_beams, dtype=np.float32)
    done = np.zeros(n_beams, dtype=bool)
    for k in range(1, n_steps + 1):
        if done.all():
            break
        d = k * step
        ix = ((pose[0] + d * cs - ox) / res).astype(int)
        iy = ((pose[1] + d * sn - oy) / res).astype(int)
        ir = h - 1 - iy  # image rows count from top; world y from bottom
        ok = ~done & (ix >= 0) & (ix < w) & (ir >= 0) & (ir < h)
        done |= ~done & ~ok
        hits = np.zeros_like(done)
        hits[ok] = occ[ir[ok], ix[ok]]
        ranges[hits] = d
        done |= hits
    return ranges


def find_centerline_start(cl_xyz, m, margin_px=10):
    """Roll the centerline so it starts at a point comfortably inside free space."""
    from scipy.ndimage import distance_transform_edt  # type: ignore
    dt = distance_transform_edt(m["free"]) * m["res"]
    ic = ((cl_xyz[:, 0] - m["origin"][0]) / m["res"]).astype(int)
    ir = m["h"] - 1 - ((cl_xyz[:, 1] - m["origin"][1]) / m["res"]).astype(int)
    ir = np.clip(ir, 0, m["h"] - 1)
    ic = np.clip(ic, 0, m["w"] - 1)
    clearance = dt[ir, ic]
    return int(np.argmax(clearance))
