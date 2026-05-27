import os
import sys
import time

import numpy as np
import pytest

pytest.importorskip("warp")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import slam  # noqa: E402

ENV_CELLS = 200
ENV_RES = 0.05
ENV_ORIGIN = -5.0

N_BEAMS = 1080
ANGLE_MIN = float(-np.radians(135.0))
ANGLE_INC = float(np.radians(270.0) / (N_BEAMS - 1))
SCAN_MAX_RANGE = 30.0


def make_room():
    occ = np.zeros((ENV_CELLS, ENV_CELLS), dtype=bool)
    occ[0, :] = True
    occ[-1, :] = True
    occ[:, 0] = True
    occ[:, -1] = True
    occ[80:90, 60:100] = True
    occ[130:170, 130:140] = True
    occ[50:55, 140:170] = True
    return occ


def simulate_scan(occ, pose):
    """Vectorised ray-cast. Returns per-beam ranges (0.0 = miss)."""
    angles = ANGLE_MIN + np.arange(N_BEAMS, dtype=np.float32) * ANGLE_INC + pose[2]
    cs = np.cos(angles)
    sn = np.sin(angles)
    h, w = occ.shape
    step = ENV_RES * 0.5
    n_steps = int(SCAN_MAX_RANGE / step)
    ranges = np.zeros(N_BEAMS, dtype=np.float32)
    done = np.zeros(N_BEAMS, dtype=bool)
    for k in range(1, n_steps + 1):
        if done.all():
            break
        d = k * step
        ix = ((pose[0] + d * cs - ENV_ORIGIN) / ENV_RES).astype(int)
        iy = ((pose[1] + d * sn - ENV_ORIGIN) / ENV_RES).astype(int)
        in_bounds = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        # Beams that have left the env are misses; mark them done.
        done |= ~done & ~in_bounds
        ok = ~done & in_bounds
        hits = np.zeros_like(done)
        hits[ok] = occ[iy[ok], ix[ok]]
        ranges[hits] = d
        done |= hits
    return ranges


def make_trajectory():
    n = 15
    gt = np.zeros((n, 3), dtype=np.float32)
    gt[:, 0] = np.linspace(0.0, 0.7, n)
    gt[:, 2] = np.linspace(0.0, np.radians(15.0), n)
    return gt


def wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def test_slam_end_to_end():
    occ = make_room()
    gt = make_trajectory()

    bridge = slam.Bridge()
    bridge.configure(N_BEAMS, ANGLE_MIN, ANGLE_INC)

    xy_err = []
    th_err = []
    timings = []
    prev = None

    for i in range(len(gt)):
        scan = simulate_scan(occ, gt[i])
        delta = np.zeros(3, dtype=np.float32) if prev is None else (gt[i] - prev)
        if prev is not None:
            delta[2] = wrap_pi(delta[2])

        t0 = time.perf_counter()
        est = bridge.step(scan, delta).copy()
        timings.append(time.perf_counter() - t0)

        xy_err.append(float(np.linalg.norm(est[:2] - gt[i][:2])))
        th_err.append(abs(float(wrap_pi(est[2] - gt[i][2]))))
        prev = gt[i].copy()

    metrics = {
        "steps": len(gt),
        "device": bridge.device,
        "xy_mean_cm": float(np.mean(xy_err)) * 100.0,
        "xy_max_cm": float(np.max(xy_err)) * 100.0,
        "th_mean_deg": float(np.degrees(np.mean(th_err))),
        "th_max_deg": float(np.degrees(np.max(th_err))),
        "latency_first_ms": timings[0] * 1000.0,
        "latency_median_ms": float(np.median(timings[1:])) * 1000.0,
        "latency_p95_ms": float(np.percentile(timings[1:], 95)) * 1000.0,
        "latency_warm_mean_ms": float(np.mean(timings[1:])) * 1000.0,
    }
    print("\n=== warpSLAM end-to-end ===")
    for k, v in metrics.items():
        print(f"  {k:24s} {v}")

    assert metrics["xy_mean_cm"] < 10.0, metrics
    assert metrics["xy_max_cm"] < 20.0, metrics
    assert metrics["th_mean_deg"] < 2.0, metrics
    assert metrics["th_max_deg"] < 5.0, metrics
