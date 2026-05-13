"""
Quick benchmark: polar-in-kernel vs precomputed-cartesian scan matching.

Run on a CUDA box:  pip install warp-lang && python bench_scan_match.py

Both kernels use the standard rotation convention (cos on x, sin on y) so the
scores match to FP rounding. The original kernel 1 had sin/cos swapped on the
output coords — that's a reflection, not a rotation. Not benchmark-relevant
but worth flagging for the actual SLAM code.
"""

import time

import numpy as np
import warp as wp

wp.init()

# --- Config (tweak to match your setup) ---
LIDAR_POINTS = 1080
NUM_CANDIDATES = 1024
MAP_W, MAP_H = 1000, 1000
RESOLUTION = 0.05  # m/cell
ORIGIN_X, ORIGIN_Y = -25.0, -25.0
LIDAR_MIN_VAL = -2.356  # -135 deg
LIDAR_INC_VAL = 0.00436  # 0.25 deg
ITERS, WARMUP = 200, 20

LIDAR_POINTS_C = wp.constant(LIDAR_POINTS)
LIDAR_MIN_C = wp.constant(LIDAR_MIN_VAL)
LIDAR_INC_C = wp.constant(LIDAR_INC_VAL)


# --- Kernel 1: polar in-kernel, 2P sin/cos per thread ---
@wp.kernel
def score_polar(
    ranges: wp.array(dtype=wp.float32),
    occupancy: wp.array2d(dtype=wp.float32),
    candidates: wp.array2d(dtype=wp.float32),
    scores: wp.array(dtype=wp.float32),
    origin_x: float,
    origin_y: float,
    inv_res: float,
    map_w: int,
    map_h: int,
):
    i = wp.tid()
    x = candidates[i, 0]
    y = candidates[i, 1]
    th = candidates[i, 2]
    total = float(0.0)
    for j in range(LIDAR_POINTS_C):
        a = LIDAR_MIN_C + LIDAR_INC_C * float(j) + th
        ca = wp.cos(a)
        sa = wp.sin(a)
        r = ranges[j]
        wx = x + r * ca
        wy = y + r * sa
        gx = int((wx - origin_x) * inv_res)
        gy = int((wy - origin_y) * inv_res)
        if gx >= 0 and gx < map_w and gy >= 0 and gy < map_h:
            total += occupancy[gy, gx]
    scores[i] = total


# --- Kernel 2: precomputed cartesian + per-candidate sin/cos ---
@wp.kernel
def score_cart(
    occupancy: wp.array2d(dtype=wp.float32),
    cart: wp.array2d(dtype=wp.float32),  # (P, 2) = (r cos β, r sin β)
    valid: wp.array(dtype=wp.uint8),
    candidates: wp.array2d(dtype=wp.float32),
    scores: wp.array(dtype=wp.float32),
    n_pts: int,
    origin_x: float,
    origin_y: float,
    inv_res: float,
    map_w: int,
    map_h: int,
):
    cid = wp.tid()
    x = candidates[cid, 0]
    y = candidates[cid, 1]
    c = wp.cos(candidates[cid, 2])
    s = wp.sin(candidates[cid, 2])
    total = float(0.0)
    for i in range(n_pts):
        if valid[i] == wp.uint8(0):
            continue
        cx = cart[i, 0]
        cy = cart[i, 1]
        wx = c * cx - s * cy + x
        wy = s * cx + c * cy + y
        gx = int((wx - origin_x) * inv_res)
        gy = int((wy - origin_y) * inv_res)
        if gx >= 0 and gx < map_w and gy >= 0 and gy < map_h:
            total += occupancy[gy, gx]
    scores[cid] = total


# --- Synthetic data ---
rng = np.random.default_rng(0)
ranges_np = rng.uniform(0.1, 10.0, size=LIDAR_POINTS).astype(np.float32)
betas = (LIDAR_MIN_VAL + LIDAR_INC_VAL * np.arange(LIDAR_POINTS)).astype(np.float32)
cart_np = np.stack(
    [ranges_np * np.cos(betas), ranges_np * np.sin(betas)], axis=1
).astype(np.float32)
valid_np = np.ones(
    LIDAR_POINTS, dtype=np.uint8
)  # all valid; flip some to 0 to test masking
occ_np = rng.random((MAP_H, MAP_W)).astype(np.float32)

cands_np = np.empty((NUM_CANDIDATES, 3), dtype=np.float32)
cands_np[:, 0] = rng.uniform(-5, 5, size=NUM_CANDIDATES)
cands_np[:, 1] = rng.uniform(-5, 5, size=NUM_CANDIDATES)
cands_np[:, 2] = rng.uniform(-np.pi, np.pi, size=NUM_CANDIDATES)

device = wp.get_device()
ranges_d = wp.array(ranges_np, dtype=wp.float32, device=device)
cart_d = wp.array(cart_np, dtype=wp.float32, device=device)
valid_d = wp.array(valid_np, dtype=wp.uint8, device=device)
occ_d = wp.array(occ_np, dtype=wp.float32, device=device)
cands_d = wp.array(cands_np, dtype=wp.float32, device=device)
scores_polar_d = wp.zeros(NUM_CANDIDATES, dtype=wp.float32, device=device)
scores_cart_d = wp.zeros(NUM_CANDIDATES, dtype=wp.float32, device=device)

inv_res = 1.0 / RESOLUTION


def launch_polar():
    wp.launch(
        score_polar,
        dim=NUM_CANDIDATES,
        inputs=[
            ranges_d,
            occ_d,
            cands_d,
            scores_polar_d,
            ORIGIN_X,
            ORIGIN_Y,
            inv_res,
            MAP_W,
            MAP_H,
        ],
        device=device,
    )


def launch_cart():
    wp.launch(
        score_cart,
        dim=NUM_CANDIDATES,
        inputs=[
            occ_d,
            cart_d,
            valid_d,
            cands_d,
            scores_cart_d,
            LIDAR_POINTS,
            ORIGIN_X,
            ORIGIN_Y,
            inv_res,
            MAP_W,
            MAP_H,
        ],
        device=device,
    )


def bench(fn, label):
    for _ in range(WARMUP):
        fn()
    wp.synchronize_device()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    wp.synchronize_device()
    ms = (time.perf_counter() - t0) / ITERS * 1000
    print(f"  {label:32s} {ms:7.3f} ms/launch")
    return ms


print(
    f"Config: {NUM_CANDIDATES} candidates × {LIDAR_POINTS} beams, map {MAP_W}×{MAP_H}"
)
print(f"Iters:  {ITERS} (after {WARMUP} warmup)\n")

t1 = bench(launch_polar, "Kernel 1 (polar in-kernel)")
t2 = bench(launch_cart, "Kernel 2 (precomputed cart)")
print(f"\n  Speedup: {t1 / t2:.2f}×")

sp = scores_polar_d.numpy()
sc = scores_cart_d.numpy()
print(f"\nCorrectness (should be ~0 up to FP rounding):")
print(f"  max |Δ| = {np.max(np.abs(sp - sc)):.6e}")
print(f"  mean|Δ| = {np.mean(np.abs(sp - sc)):.6e}")
