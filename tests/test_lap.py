"""Full-lap SLAM accuracy test on real F1Tenth maps.

For each available map under warporacer/maps (or $WARPSLAM_MAPS), this test:
  1. Extracts the track centerline from the map's free-space skeleton.
  2. Resamples it at a constant arc-length spacing -> ground-truth poses
     (x, y, yaw=tangent).
  3. Simulates a 270 deg / 1080 beam UST-10LX scan at each pose by ray-casting
     the static occupancy grid (with jittered odometry to model VESC noise).
  4. Feeds each scan + noisy odom delta to slam.Bridge.step.
  5. Aggregates translation/rotation drift and per-step latency.

Maps that don't fit in the SLAM's 2048-cell grid (>=~100 m extent) or whose
free-space skeleton doesn't form a clean closed loop are skipped.

Pure scan-matching SLAM with no loop closure accumulates drift roughly linearly
in lap length; tracks longer than LONG_LAP_M are marked xfail to document this
without forcing the test suite red.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("scipy")
pytest.importorskip("skimage")
pytest.importorskip("yaml")
pytest.importorskip("PIL")

import slam  # noqa: E402
from conftest import (  # noqa: E402
    ANGLE_INC, ANGLE_MIN, N_BEAMS, SCAN_MAX_RANGE,
    extract_centerline, find_centerline_start, load_map, maps_dir,
    rc_to_world, recenter_map, resample_centerline, simulate_scan, wrap,
)

SLAM_HALF_EXTENT = float(-slam.ORIGIN)  # 51.2 m
LAP_STEP_M = 0.15                       # 15 cm waypoint spacing (~ 0.5 m/s @ 3 Hz)
MAX_POSES = 800
LONG_LAP_M = 80.0                       # tracks longer than this are xfail'd


def _discover_maps():
    md = maps_dir()
    if md is None:
        return []
    return sorted(md.glob("*.yaml"))


MAPS = _discover_maps()
MAP_PARAMS = [pytest.param(y, id=y.stem) for y in MAPS]


def _build_lap(yaml_path):
    """Load + recenter + extract waypoint trajectory, or pytest.skip()."""
    m = load_map(yaml_path)
    rc = extract_centerline(m["free"])
    cl = rc_to_world(rc, m)

    arc = float(np.linalg.norm(np.diff(cl, axis=0), axis=1).sum())
    if arc < 5.0:
        pytest.skip(f"centerline too short ({arc:.2f} m); not a usable track")
    chord = float(np.linalg.norm(cl[-1] - cl[0]))
    if chord > 1.5:
        pytest.skip(f"centerline doesn't close (chord={chord:.2f} m); extractor failed")

    m, cl = recenter_map(m, cl, SLAM_HALF_EXTENT)
    if m is None:
        pytest.skip(f"track too large for {2 * SLAM_HALF_EXTENT:.0f} m SLAM grid")

    start = find_centerline_start(np.column_stack([cl, np.zeros(len(cl))]), m)
    cl = np.roll(cl, -start, axis=0)
    poses = resample_centerline(cl, step=LAP_STEP_M)[:MAX_POSES]
    return m, poses, arc


def _run_lap(m, poses, odom_noise_xy=0.005, odom_noise_th=0.002, seed=0):
    bridge = slam.Bridge()
    bridge.configure(N_BEAMS, ANGLE_MIN, ANGLE_INC)
    rng = np.random.default_rng(seed)
    xy_err, th_err, timings = [], [], []
    prev = None
    for p in poses:
        scan = simulate_scan(m["occ"], p, N_BEAMS, ANGLE_MIN, ANGLE_INC,
                             SCAN_MAX_RANGE, res=m["res"], origin=m["origin"])
        if prev is None:
            # Bridge.step treats `odom` on the first call as the absolute
            # initial pose. Seed at ground truth.
            delta = p.copy()
        else:
            d = p - prev
            d[2] = wrap(d[2])
            noise = np.array([
                rng.normal(0.0, odom_noise_xy),
                rng.normal(0.0, odom_noise_xy),
                rng.normal(0.0, odom_noise_th),
            ], dtype=np.float32)
            delta = (d + noise).astype(np.float32)
        t0 = time.perf_counter()
        est = bridge.step(scan, delta).copy()
        timings.append(time.perf_counter() - t0)
        xy_err.append(float(np.linalg.norm(est[:2] - p[:2])))
        th_err.append(abs(float(wrap(est[2] - p[2]))))
        prev = p.copy()
    return bridge, np.array(xy_err), np.array(th_err), np.array(timings)


def _summarize(label, bridge, xy_err, th_err, timings, lap_length):
    warm = timings[max(1, len(timings) // 20):]
    metrics = dict(
        map=label,
        device=bridge.device,
        poses=len(xy_err),
        lap_length_m=lap_length,
        xy_mean_cm=float(xy_err.mean()) * 100.0,
        xy_p95_cm=float(np.percentile(xy_err, 95)) * 100.0,
        xy_max_cm=float(xy_err.max()) * 100.0,
        th_mean_deg=float(np.degrees(th_err.mean())),
        th_p95_deg=float(np.degrees(np.percentile(th_err, 95))),
        th_max_deg=float(np.degrees(th_err.max())),
        latency_first_ms=float(timings[0]) * 1000.0,
        latency_median_ms=float(np.median(warm)) * 1000.0,
        latency_p95_ms=float(np.percentile(warm, 95)) * 1000.0,
    )
    print(f"\n=== warpSLAM full lap: {label} ===")
    for k, v in metrics.items():
        print(f"  {k:24s} {v}")
    return metrics


@pytest.mark.skipif(not MAPS, reason="no warporacer maps found (set WARPSLAM_MAPS=...)")
@pytest.mark.parametrize("yaml_path", MAP_PARAMS)
def test_full_lap(yaml_path, request):
    m, poses, arc = _build_lap(yaml_path)
    lap_length = float(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1).sum())

    if lap_length > LONG_LAP_M:
        request.node.add_marker(pytest.mark.xfail(
            reason=f"{lap_length:.0f} m > {LONG_LAP_M:.0f} m: pure scan-matching drifts on "
                   "long featureless tracks without loop closure",
            strict=False,
        ))

    bridge, xy, th, timings = _run_lap(m, poses)
    metrics = _summarize(yaml_path.stem, bridge, xy, th, timings, lap_length)

    # Drift bounds scale with lap length: ~1 cm/m mean drift is healthy for
    # pure scan-matching (no loop closure). Angle bounds are absolute since a
    # well-localized matcher should track yaw to within ~1 deg regardless of
    # distance traveled.
    assert metrics["xy_mean_cm"] < max(20.0, lap_length), metrics
    assert metrics["xy_max_cm"] < max(40.0, 2.0 * lap_length), metrics
    assert metrics["th_mean_deg"] < 6.0, metrics
    assert metrics["th_max_deg"] < 25.0, metrics
    assert metrics["latency_median_ms"] < 50.0, metrics


@pytest.mark.skipif(not MAPS, reason="no warporacer maps found")
def test_latency_budget():
    """Median per-step latency must fit a 40 Hz F1Tenth control loop (25 ms)."""
    for y in MAPS:
        try:
            m, poses, _ = _build_lap(y)
        except pytest.skip.Exception:
            continue
        bridge, _, _, timings = _run_lap(m, poses[:80])
        warm = timings[max(1, len(timings) // 5):]
        med = float(np.median(warm)) * 1000.0
        p95 = float(np.percentile(warm, 95)) * 1000.0
        print(f"  {y.stem:18s} device={bridge.device} median={med:.2f} ms  p95={p95:.2f} ms")
        assert med < 25.0, (y.stem, med, p95)
        return  # one map is enough — they all use the same kernels
    pytest.skip("no usable maps available")
