# warpSLAM

Lidar-only scan-matching SLAM for F1Tenth, on top of NVIDIA Warp.
Two files, three kernels: coarse grid search + Gauss-Newton refinement against
a 102 m × 102 m log-odds occupancy grid.

## Run on the car

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
ros2 launch f1tenth_stack bringup_launch.py
python3 slam_node.py
```

Set `vesc_to_odom/publish_tf: true` in `f1tenth_stack/config/vesc.yaml`.

## Run the tests

```bash
uv run pytest tests/
```

- `tests/test_e2e.py` — synthetic-room smoke test (~15 steps).
- `tests/test_lap.py` — full-lap accuracy + latency test on real F1Tenth maps
  picked up from sibling `../warporacer/maps/` (or `$WARPSLAM_MAPS`).
  Each map is auto-recentered to fit the SLAM grid; tracks that don't extract
  a clean closed centerline or that are too large are skipped, and tracks
  longer than ~80 m are marked `xfail` because pure scan-matching accumulates
  drift without loop closure.

Current baselines (CPU, M-series Mac):

| Map    | Lap   | xy mean | xy max | θ mean | latency (median) |
|--------|-------|---------|--------|--------|------------------|
| skirk  |  33 m |  8 cm   | 15 cm  | 0.2°   | 1.5 ms           |
| berlin |  69 m | 22 cm   | 60 cm  | 4.2°   | 1.6 ms           |
| levine | 120 m | 130 cm  | 215 cm | 94°    | 1.6 ms (xfail)   |
