# Simplest F1Tenth SLAM in Warp

One race. One file. No threads. No graph optimization. No loop closure.

## What's gone (and why it's OK)

| Removed | Why it doesn't matter for one race |
|---|---|
| Loop closure | Drift over 30 s of racing on a 50m track is small. You don't need to close the loop because you're not coming back to the start with the same map. |
| Pose graph + Gauss-Newton | Nothing to optimize. No edges. |
| Likelihood field | Score directly against the occupancy grid. Slightly noisier scoring; doesn't matter when you update at 40 Hz. |
| Coarse-to-fine search | Single-pass search at moderate resolution is fast enough on Orin. |
| Ray-march rasterization | Just mark the endpoint cell as occupied. You lose free-space tracking, but you only need walls for localization. |
| Scan history buffer | Never read past scans. |
| Threads, double-buffer, locking | Everything is synchronous in the ROS callback. |
| cuVSLAM | Lidar alone is plenty for a flat indoor track. |

What's left: **one grid, three kernels, one synchronous loop**.

---

## The whole algorithm

```
state: occupancy[H, W] = 0, pose = (0, 0, 0), prev_pose = (0, 0, 0)

first scan:
    rasterize_endpoints(scan, pose=(0,0,0))   # seeds the map

every subsequent scan:
    cart = polar_to_cart(scan)
    predicted = pose + (pose - prev_pose)     # constant-velocity prior
    candidates = grid of (dx, dy, dθ) around predicted
    scores = for each candidate: Σ occupancy[transform(cart, candidate)]
    new_pose = candidates[argmax(scores)]
    rasterize_endpoints(cart, new_pose)
    prev_pose, pose = pose, new_pose
```

That's the entire SLAM system.

---

## Sizing the search

Worst-case motion per scan at 10 m/s and 40 Hz: 25 cm and a few degrees.
Constant-velocity prediction removes most of that, leaving residual to search:

- **Search window**: ±10 cm in xy, ±3° in θ
- **Step**: 2 cm in xy, 0.3° in θ
- **Grid**: 11 × 11 × 21 = 2541 candidates

Larger than strictly needed, but cheap. If the car oscillates between two
candidates, halve the step.

---

## The three kernels

```python
import warp as wp

@wp.kernel
def polar_to_cart(
    ranges: wp.array(dtype=wp.float32),       # (P,)
    cart:   wp.array2d(dtype=wp.float32),     # (P, 2)
    valid:  wp.array(dtype=wp.uint8),         # (P,)
    angle_min: float, angle_inc: float,
    r_min: float, r_max: float,
):
    i = wp.tid()
    r = ranges[i]
    if r > r_min and r < r_max:
        a = angle_min + float(i) * angle_inc
        cart[i, 0] = r * wp.cos(a)
        cart[i, 1] = r * wp.sin(a)
        valid[i] = wp.uint8(1)
    else:
        valid[i] = wp.uint8(0)


@wp.kernel
def score_candidates(
    occupancy: wp.array2d(dtype=wp.float32),  # (H, W)
    cart:      wp.array2d(dtype=wp.float32),  # (P, 2)
    valid:     wp.array(dtype=wp.uint8),
    cand:      wp.array2d(dtype=wp.float32),  # (NC, 3) world-frame (x, y, θ)
    scores:    wp.array(dtype=wp.float32),    # (NC,)
    n_pts: int, ox: float, oy: float, inv_res: float, mw: int, mh: int,
):
    cid = wp.tid()
    x = cand[cid, 0]
    y = cand[cid, 1]
    c = wp.cos(cand[cid, 2])
    s = wp.sin(cand[cid, 2])
    total = float(0.0)
    for i in range(n_pts):
        if valid[i] == wp.uint8(0):
            continue
        wx = c * cart[i, 0] - s * cart[i, 1] + x
        wy = s * cart[i, 0] + c * cart[i, 1] + y
        gx = int((wx - ox) * inv_res)
        gy = int((wy - oy) * inv_res)
        if gx >= 0 and gx < mw and gy >= 0 and gy < mh:
            total += occupancy[gy, gx]
    scores[cid] = total


@wp.kernel
def rasterize_endpoints(
    occupancy: wp.array2d(dtype=wp.float32),
    cart:      wp.array2d(dtype=wp.float32),
    valid:     wp.array(dtype=wp.uint8),
    pose:      wp.array(dtype=wp.float32),    # (3,)
    n_pts: int, ox: float, oy: float, inv_res: float, mw: int, mh: int,
    increment: float, clip: float,
):
    i = wp.tid()
    if valid[i] == wp.uint8(0):
        return
    c = wp.cos(pose[2])
    s = wp.sin(pose[2])
    wx = c * cart[i, 0] - s * cart[i, 1] + pose[0]
    wy = s * cart[i, 0] + c * cart[i, 1] + pose[1]
    gx = int((wx - ox) * inv_res)
    gy = int((wy - oy) * inv_res)
    if gx >= 0 and gx < mw and gy >= 0 and gy < mh:
        # Atomic add with saturation. `clip` keeps the grid from running away.
        old = wp.atomic_add(occupancy, gy, gx, increment)
        if old + increment > clip:
            occupancy[gy, gx] = clip
```

---

## The whole loop

```python
class SimpleSLAM:
    def __init__(self):
        self.H, self.W = 1024, 1024
        self.res = 0.05                       # 5 cm/cell
        self.origin = -25.6                    # map covers [-25.6, +25.6] m
        self.P = 1080

        self.occ   = wp.zeros((self.H, self.W), dtype=wp.float32, device="cuda")
        self.cart  = wp.zeros((self.P, 2), dtype=wp.float32, device="cuda")
        self.valid = wp.zeros(self.P, dtype=wp.uint8, device="cuda")
        self.ranges_buf = wp.zeros(self.P, dtype=wp.float32, device="cuda")
        self.cand_buf   = wp.zeros((2541, 3), dtype=wp.float32, device="cuda")
        self.scores     = wp.zeros(2541, dtype=wp.float32, device="cuda")
        self.pose_buf   = wp.zeros(3, dtype=wp.float32, device="cuda")

        self.pose      = np.zeros(3, dtype=np.float32)
        self.prev_pose = np.zeros(3, dtype=np.float32)
        self.first = True

    def on_scan(self, ranges, angle_min, angle_inc):
        # 1. polar → cartesian
        wp.copy(self.ranges_buf, wp.from_numpy(ranges.astype(np.float32)))
        wp.launch(polar_to_cart, dim=self.P,
                  inputs=[self.ranges_buf, self.cart, self.valid,
                          angle_min, angle_inc, 0.1, 10.0])

        if self.first:
            self._rasterize(self.pose)
            self.first = False
            return self.pose

        # 2. build candidate grid around constant-velocity prediction
        predicted = self.pose + (self.pose - self.prev_pose)     # SE(2) hack: OK at small dt
        cand = self._make_candidates(predicted)                  # (2541, 3), CPU
        wp.copy(self.cand_buf, wp.from_numpy(cand))

        # 3. score all candidates
        wp.launch(score_candidates, dim=2541,
                  inputs=[self.occ, self.cart, self.valid, self.cand_buf,
                          self.scores, self.P,
                          self.origin, self.origin, 1.0/self.res, self.W, self.H])

        # 4. argmax on host — 10 KB copy, ~10 μs
        best = int(self.scores.numpy().argmax())
        new_pose = cand[best]

        # 5. rasterize at the winning pose
        self._rasterize(new_pose)

        self.prev_pose, self.pose = self.pose, new_pose
        return new_pose

    def _make_candidates(self, center):
        # 11 × 11 × 21 = 2541
        dx = np.linspace(-0.10, 0.10, 11)
        dy = np.linspace(-0.10, 0.10, 11)
        dt = np.linspace(-0.052, 0.052, 21)        # ±3°
        gx, gy, gt = np.meshgrid(dx, dy, dt, indexing="ij")
        out = np.stack([center[0] + gx.ravel(),
                        center[1] + gy.ravel(),
                        center[2] + gt.ravel()], axis=1)
        return out.astype(np.float32)

    def _rasterize(self, pose):
        wp.copy(self.pose_buf, wp.from_numpy(pose.astype(np.float32)))
        wp.launch(rasterize_endpoints, dim=self.P,
                  inputs=[self.occ, self.cart, self.valid, self.pose_buf,
                          self.P, self.origin, self.origin, 1.0/self.res,
                          self.W, self.H, 1.0, 100.0])
```

ROS node is a thin wrapper: subscribe to `/scan`, call `on_scan`, publish
pose on `/slam_pose`. Add the `rclpy` boilerplate and you're done.

---

## Budget on Jetson Orin

| Step | Time |
|---|---|
| polar → cart | <0.5 ms |
| score (2541 × 1080) | 3–6 ms |
| argmax via numpy | <0.1 ms |
| rasterize | 0.3 ms |
| **total** | **~5–8 ms** |

You have 25 ms at 40 Hz. There's plenty of room.

---

## What this won't handle (and what to do if it bites you)

1. **First-scan ambiguity.** If the car moves before the first scan
   arrives, your map is wrong forever. Hold the car stationary until you
   see the pose stop jittering on the published topic.

2. **Walls disappear if the score saturates.** That's what the `clip` is
   for — caps each cell. If the map looks like solid bright cells
   everywhere within range, lower `clip` or use smaller `increment`.

3. **Score plateau in featureless corridors.** Rare on a real F1Tenth
   track (cones, barriers everywhere give features). If it happens,
   detect by checking that `(best_score - median_score)` exceeds a
   threshold; if not, hold pose at the prediction without updating.

4. **Search window too small at high speed.** If the car goes faster than
   constant-velocity predicts (sharp acceleration out of a corner), the
   true pose falls outside the candidate grid and the match snaps to a
   wrong-but-locally-plausible cell. Symptom: pose jumps backward by
   a few cells. Fix: widen the search window to ±20 cm and accept the
   2× cost — still fits in budget.

5. **Map drift over 30 seconds.** Probably 10–30 cm by end of race. For
   path following on a known track, fine. If it's not fine, that's when
   you reach for loop closure and graph optimization — but not for a
   class assignment.

---

## Build order

1. Get `polar_to_cart` working in isolation. Publish the Cartesian points
   to RViz, drive the car around, watch the points sweep with the car.
2. Add `rasterize_endpoints` with the pose hard-coded at (0,0,0). Drive
   the car *very slowly* and watch the map build (it'll smear because
   pose is fixed). This proves the rasterizer.
3. Add `score_candidates` and the search. Now the map should stop
   smearing as the localization picks up the motion.
4. Tune `increment`, `clip`, and the search window on real data.

The whole thing is well under 300 lines of Python. You can write it in a
weekend and have time to tune.
