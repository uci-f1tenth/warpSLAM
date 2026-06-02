import time

import numpy as np
import pytest

pytest.importorskip("warp")

from conftest import (  # noqa: E402
    ANGLE_INC,
    ANGLE_MIN,
    N_BEAMS,
    SCAN_MAX_RANGE,
    simulate_scan,
    wrap,
)

import slam  # noqa: E402


def make_room():
    occ = np.zeros((200, 200), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    occ[80:90, 60:100] = True
    occ[130:170, 130:140] = True
    occ[50:55, 140:170] = True
    return occ


def make_trajectory(n=15):
    gt = np.zeros((n, 3), dtype=np.float32)
    gt[:, 0] = np.linspace(0.0, 0.7, n)
    gt[:, 2] = np.linspace(0.0, np.radians(15.0), n)
    return gt


def test_slam_end_to_end():
    occ = make_room()
    gt = make_trajectory()
    origin = (-5.0, -5.0)
    bridge = slam.Bridge()
    bridge.configure(N_BEAMS, ANGLE_MIN, ANGLE_INC)

    xy_err, th_err, timings = [], [], []
    prev = None
    for p in gt:
        scan = simulate_scan(
            occ,
            p,
            N_BEAMS,
            ANGLE_MIN,
            ANGLE_INC,
            SCAN_MAX_RANGE,
            res=0.05,
            origin=origin,
        )
        delta = np.zeros(3, dtype=np.float32) if prev is None else (p - prev)
        if prev is not None:
            delta[2] = wrap(delta[2])
        t0 = time.perf_counter()
        est = bridge.step(scan, delta).copy()
        timings.append(time.perf_counter() - t0)
        xy_err.append(float(np.linalg.norm(est[:2] - p[:2])))
        th_err.append(abs(float(wrap(est[2] - p[2]))))
        prev = p.copy()

    metrics = dict(
        steps=len(gt),
        device=bridge.device,
        xy_mean_cm=float(np.mean(xy_err)) * 100.0,
        xy_max_cm=float(np.max(xy_err)) * 100.0,
        th_mean_deg=float(np.degrees(np.mean(th_err))),
        th_max_deg=float(np.degrees(np.max(th_err))),
        latency_first_ms=timings[0] * 1000.0,
        latency_median_ms=float(np.median(timings[1:])) * 1000.0,
        latency_p95_ms=float(np.percentile(timings[1:], 95)) * 1000.0,
        latency_warm_mean_ms=float(np.mean(timings[1:])) * 1000.0,
    )
    print("\n=== warpSLAM end-to-end (synthetic room) ===")
    for k, v in metrics.items():
        print(f"  {k:24s} {v}")

    assert metrics["xy_mean_cm"] < 10.0, metrics
    assert metrics["xy_max_cm"] < 20.0, metrics
    assert metrics["th_mean_deg"] < 2.0, metrics
    assert metrics["th_max_deg"] < 5.0, metrics
