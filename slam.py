import numpy as np

import warp as wp

# Map
GRID_WIDTH = wp.constant(2048)
GRID_HEIGHT = wp.constant(2048)
RESOLUTION = wp.constant(0.05)
INV_RESOLUTION = wp.constant(1.0 / 0.05)
ORIGIN = wp.constant(wp.vec2(-51.2, -51.2))

# Log-odds
L_OCC = wp.constant(0.85)
L_FREE = wp.constant(-0.4)
L_MIN = wp.constant(-5.0)
L_MAX = wp.constant(5.0)

# Lidar
LIDAR_FOV = wp.constant(wp.radians(270.0))
LIDAR_POINTS = wp.constant(1081)
RANGE_MIN = wp.constant(0.1)
RANGE_MAX = wp.constant(10.0)
BEAM_STRIDE = wp.constant(8)
NUM_MATCH_BEAMS = wp.constant(LIDAR_POINTS // BEAM_STRIDE)
LIDAR_MIN_ANGLE = wp.constant(-LIDAR_FOV * 0.5)
LIDAR_INCREMENT = wp.constant(LIDAR_FOV / LIDAR_POINTS)
INV_9 = wp.constant(1.0 / 9.0)

# ═══════════════════════════════════════════════════════════════════
# Warp helpers
# ═══════════════════════════════════════════════════════════════════


@wp.func
def _read(f: wp.array2d[float], x: int, y: int):
    return f[wp.clamp(y, 0, GRID_HEIGHT - 1), wp.clamp(x, 0, GRID_WIDTH - 1)]


@wp.func
def _sample(f: wp.array2d[float], x: float, y: float):
    ix = int(wp.floor(x))
    iy = int(wp.floor(y))
    tx = x - float(ix)
    ty = y - float(iy)
    s0 = wp.lerp(_read(f, ix, iy), _read(f, ix + 1, iy), tx)
    s1 = wp.lerp(_read(f, ix, iy + 1), _read(f, ix + 1, iy + 1), tx)
    return wp.lerp(s0, s1, ty)


# ═══════════════════════════════════════════════════════════════════
# Warp kernels
# ═══════════════════════════════════════════════════════════════════


@wp.kernel
def _search(
    ranges: wp.array[float],
    sin_table: wp.array[float],
    cos_table: wp.array[float],
    likelihood: wp.array2d[float],
    seed_x: float,
    seed_y: float,
    seed_theta: float,
    step_xy: float,
    step_theta: float,
    n_xy: int,
    n_theta: int,
    scores: wp.array[float],
):
    """Exhaustive 3-DOF search: score every (x, y, θ) against the likelihood field."""
    i, j, k = wp.tid()
    half = float(n_xy - 1) * 0.5
    x = seed_x + (float(i) - half) * step_xy
    y = seed_y + (float(j) - half) * step_xy
    theta = seed_theta + (float(k) - float(n_theta - 1) * 0.5) * step_theta
    ctheta = wp.cos(theta)
    stheta = wp.sin(theta)

    total = float(0.0)
    for b in range(NUM_MATCH_BEAMS):
        r = ranges[b * BEAM_STRIDE]
        if not (RANGE_MIN <= r < RANGE_MAX):
            continue
        # beam direction in world frame
        ca = cos_table[b] * ctheta - sin_table[b] * stheta
        sa = sin_table[b] * ctheta + cos_table[b] * stheta
        gx = (x + r * ca - ORIGIN[0]) * INV_RESOLUTION
        gy = (y + r * sa - ORIGIN[1]) * INV_RESOLUTION
        total += _sample(likelihood, gx, gy)

    scores[i * n_xy * n_theta + j * n_theta + k] = total


@wp.kernel
def _integrate(
    ranges: wp.array[float],
    pose_x: float,
    pose_y: float,
    pose_theta: float,
    logodds: wp.array2d[float],
):
    """Ray-cast one scan into the log-odds grid (free along ray, occupied at tip)."""
    i = wp.tid()
    r = ranges[i]
    if not (RANGE_MIN <= r < RANGE_MAX):
        return

    a = LIDAR_MIN_ANGLE + LIDAR_INCREMENT * float(i) + pose_theta
    ca = wp.cos(a)
    sa = wp.sin(a)

    x0 = (pose_x - ORIGIN[0]) * INV_RESOLUTION
    y0 = (pose_y - ORIGIN[1]) * INV_RESOLUTION
    x1 = (pose_x + r * ca - ORIGIN[0]) * INV_RESOLUTION
    y1 = (pose_y + r * sa - ORIGIN[1]) * INV_RESOLUTION

    dx = x1 - x0
    dy = y1 - y0
    steps = int(wp.max(wp.abs(dx), wp.abs(dy))) + 1
    ux = dx / float(steps)
    uy = dy / float(steps)

    gx = x0
    gy = y0
    for _ in range(steps):
        ix = int(gx)
        iy = int(gy)
        if 0 <= ix < GRID_WIDTH and 0 <= iy < GRID_HEIGHT:
            wp.atomic_add(logodds, iy, ix, L_FREE)
        gx += ux
        gy += uy

    ix = int(x1)
    iy = int(y1)
    if 0 <= ix < GRID_WIDTH and 0 <= iy < GRID_HEIGHT:
        wp.atomic_add(logodds, iy, ix, L_OCC - L_FREE)


@wp.kernel
def _blur(
    logodds: wp.array2d[float],
    likelihood: wp.array2d[float],
):
    """Clamp log-odds to [L_MIN, L_MAX], then build likelihood = 3×3 mean of +ve values."""
    x, y = wp.tid()
    v = wp.clamp(logodds[y, x], L_MIN, L_MAX)
    logodds[y, x] = v

    if not (1 <= x < GRID_WIDTH - 1 and 1 <= y < GRID_HEIGHT - 1):
        return

    s = 0.0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            s += wp.max(logodds[y + dy, x + dx], 0.0)
    likelihood[y, x] = s * INV_9


# ═══════════════════════════════════════════════════════════════════
# Bridge  (the SLAM engine)
# ═══════════════════════════════════════════════════════════════════


class Bridge:
    """Real-time 2D occupancy-grid SLAM for F1Tenth track mapping.

    Multi-resolution correlative scan matching against a likelihood
    field built from the log-odds occupancy grid.

    Parameters
    ----------
    config : dict, optional
        Override defaults (stages, kf_dist, kf_angle, beam_stride).
    """

    def __init__(self, config: dict | None = None):
        wp.init()

        opts = {
            "stages": [
                (0.30, 15.0, 0.05, 1.0),
                (0.05, 2.0, 0.01, 0.2),
                (0.01, 0.5, 0.002, 0.05),
            ],
            "kf_dist": 0.20,
            "kf_angle": 10.0,
            "beam_stride": 8,
        }
        opts.update(config or {})

        self._kf_d2 = opts["kf_dist"] ** 2
        self._kf_dt = np.radians(opts["kf_angle"])

        # ── beam tables for matching (subsampled) ──
        n_pts = int(LIDAR_POINTS)
        stride = opts["beam_stride"]
        n_match = n_pts // stride
        fov_rad = float(LIDAR_FOV)
        step = fov_rad / n_pts
        a = -fov_rad * 0.5 + step * stride * np.arange(n_match, dtype=np.float32)
        self._cos_table = wp.array(np.cos(a))
        self._sin_table = wp.array(np.sin(a))

        # ── grids ──
        self.logodds = wp.zeros((int(GRID_HEIGHT), int(GRID_WIDTH)))
        self.likelihood = wp.zeros((int(GRID_HEIGHT), int(GRID_WIDTH)))

        # ── per-scan buffers ──
        self._ranges = wp.zeros(n_pts)
        self._ranges_buf = np.empty(n_pts, dtype=np.float32)

        # score buffer sized for the largest search stage
        max_candidates = 0
        for hx, ht_deg, sx, st_deg in opts["stages"]:
            ht = np.radians(ht_deg)
            st = np.radians(st_deg)
            nxy = 2 * int(hx / sx) + 1
            nth = 2 * int(ht / st) + 1
            max_candidates = max(max_candidates, nxy * nxy * nth)
        self._scores = wp.zeros(max_candidates, dtype=float)

        # ── state ──
        self.pose = np.zeros(3, dtype=np.float32)
        self._last_kf = np.zeros(3, dtype=np.float32)
        self._first = True

        # diagnostics (exposed for monitoring / visualisation)
        self.last_score = 0.0
        self._last_n_valid = 0

        self._stages = opts["stages"]

    # ────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────

    def step(self, ranges_np: np.ndarray, odom_delta: np.ndarray) -> np.ndarray:
        """Process one lidar scan + odometry delta → updated pose (x, y, θ).

        Parameters
        ----------
        ranges_np : (LIDAR_POINTS,) float32/64
            Raw range readings (m). NaN/Inf → 0 recommended.
        odom_delta : (3,) float32/64
            (dx, dy, dθ) motion predicted since the previous scan.

        Returns
        -------
        pose : (3,) float32  (x, y, θ) in the map frame.
        """
        self._ranges_buf[:] = ranges_np
        self._ranges.assign(self._ranges_buf)

        if self._first:
            return self._init(odom_delta)

        seed = self.pose + odom_delta.astype(np.float32)

        hit = (ranges_np >= float(RANGE_MIN)) & (ranges_np < float(RANGE_MAX))
        if not np.any(hit):
            self.pose = seed
            return self.pose

        self.pose = self._match(seed)

        d = self.pose - self._last_kf
        if (d[0] * d[0] + d[1] * d[1]) > self._kf_d2 or abs(d[2]) > self._kf_dt:
            self._integrate()

        return self.pose

    def as_occupancy(self) -> np.ndarray:
        """ROS OccupancyGrid.data-ready array: 0=free, 100=occupied, -1=unknown."""
        lo = self.logodds.numpy()
        out = np.full(lo.shape, -1, dtype=np.int8)
        known = np.abs(lo) > 0.1
        p = 1.0 / (1.0 + np.exp(-np.clip(lo[known], -10.0, 10.0)))
        out[known] = np.clip((p * 100).astype(np.int8), 0, 100)
        return out

    def save_map(self, path: str):
        """Persist the log-odds grid and current pose."""
        np.savez_compressed(path, logodds=self.logodds.numpy(), pose=self.pose)

    def load_map(self, path: str):
        """Restore a previously saved map."""
        d = np.load(path)
        self.logodds.assign(d["logodds"])
        self.pose = d["pose"]
        self._last_kf = self.pose.copy()
        self._first = False
        self._rebuild_likelihood()

    def reset(self):
        """Clear the map and reset all state."""
        self.logodds.zero_()
        self.likelihood.zero_()
        self.pose.fill(0)
        self._last_kf.fill(0)
        self._first = True
        self.last_score = 0.0
        self._last_n_valid = 0

    # ────────────────────────────────────────────────────────────
    # Internal
    # ────────────────────────────────────────────────────────────

    def _init(self, odom_delta: np.ndarray) -> np.ndarray:
        self.pose = odom_delta.astype(np.float32)
        self._last_kf = self.pose.copy()
        self._first = False
        self._integrate()
        return self.pose

    def _match(self, seed: np.ndarray) -> np.ndarray:
        p = seed.astype(np.float32)

        for stage_idx, (hxy, hth_deg, sxy, sth_deg) in enumerate(self._stages):
            hth = np.radians(hth_deg)
            sth = np.radians(sth_deg)

            n_xy = 2 * int(hxy / sxy) + 1
            n_theta = 2 * int(hth / sth) + 1
            count = n_xy * n_xy * n_theta

            wp.launch(
                _search,
                dim=(n_xy, n_xy, n_theta),
                inputs=[
                    self._ranges,
                    self._sin_table,
                    self._cos_table,
                    self.likelihood,
                    p[0],
                    p[1],
                    p[2],
                    sxy,
                    sth,
                    n_xy,
                    n_theta,
                    self._scores,
                ],
            )

            scores_np = self._scores.numpy()[:count]
            best = int(scores_np.argmax())

            i = best // (n_xy * n_theta)
            rem = best % (n_xy * n_theta)
            j = rem // n_theta
            k_val = rem % n_theta

            p[0] += (i - (n_xy - 1) * 0.5) * sxy
            p[1] += (j - (n_xy - 1) * 0.5) * sxy
            p[2] += (k_val - (n_theta - 1) * 0.5) * sth

            if stage_idx == len(self._stages) - 1:
                self.last_score = float(scores_np[best])
                nonzero = scores_np[scores_np > 0.0]
                self._last_n_valid = int(nonzero.size)

        return p

    def _integrate(self):
        wp.launch(
            _integrate,
            dim=int(LIDAR_POINTS),
            inputs=[
                self._ranges,
                self.pose[0],
                self.pose[1],
                self.pose[2],
                self.logodds,
            ],
        )
        self._last_kf = self.pose.copy()
        self._rebuild_likelihood()

    def _rebuild_likelihood(self):
        wp.launch(
            _blur,
            dim=(int(GRID_WIDTH), int(GRID_HEIGHT)),
            inputs=[self.logodds, self.likelihood],
        )


# ═══════════════════════════════════════════════════════════════════
# Demo (standalone, no ROS)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override default Warp device."
    )
    parser.add_argument("--frames", type=int, default=10000, help="Number of frames.")
    parser.add_argument("--headless", action="store_true", help="No GUI.")
    args = parser.parse_args()

    LIDAR_STEP = float(LIDAR_FOV) / float(LIDAR_POINTS)
    RMIN = float(RANGE_MIN)
    RMAX = float(RANGE_MAX)
    LMIN_A = float(LIDAR_MIN_ANGLE)
    NPTS = int(LIDAR_POINTS)

    ROOM_X_MIN, ROOM_X_MAX = -2.0, 20.0
    ROOM_Y_MIN, ROOM_Y_MAX = -3.0, 3.0

    def _fake_scan(pos_x: float, pos_y: float = 0.0):
        angles = LMIN_A + LIDAR_STEP * np.arange(NPTS)
        ca = np.cos(angles)
        sa = np.sin(angles)
        eps = 1e-6
        t = np.full(NPTS, np.inf, dtype=np.float32)
        m = ca > eps
        t[m] = np.minimum(t[m], (ROOM_X_MAX - pos_x) / ca[m])
        m = ca < -eps
        t[m] = np.minimum(t[m], (ROOM_X_MIN - pos_x) / ca[m])
        m = sa > eps
        t[m] = np.minimum(t[m], (ROOM_Y_MAX - pos_y) / sa[m])
        m = sa < -eps
        t[m] = np.minimum(t[m], (ROOM_Y_MIN - pos_y) / sa[m])
        bad = (t < RMIN) | (t >= RMAX) | ~np.isfinite(t)
        t[bad] = 0.0
        return t

    with wp.ScopedDevice(args.device):
        bridge = Bridge()
        fake_odom = np.array([0.05, 0.0, 0.0], dtype=np.float32)

        if args.headless:
            for frame in range(args.frames):
                bridge.step(_fake_scan((1 + frame) * 0.05), fake_odom)
        else:
            import matplotlib
            import matplotlib.animation as anim
            import matplotlib.pyplot as plt

            fig, (ax_map, ax_info) = plt.subplots(1, 2, figsize=(14, 6))

            img = ax_map.imshow(
                bridge.likelihood.numpy(),
                origin="lower",
                animated=True,
                interpolation="antialiased",
            )
            img.set_norm(matplotlib.colors.Normalize(0.0, float(L_MAX)))
            ax_map.set_title("Likelihood Map")

            ax_info.set_xlim(0, 300)
            ax_info.set_ylim(0, 675)
            ax_info.set_xlabel("Frame")
            ax_info.set_ylabel("Score")
            (line_score,) = ax_info.plot([], [], label="score")
            (line_spread,) = ax_info.plot([], [], label="spread")
            ax_info.legend()

            fb, sb, spb = [], [], []

            def step_and_render(frame_num, _img):
                bridge.step(_fake_scan((1 + frame_num) * 0.05), fake_odom)
                _img.set_array(bridge.likelihood.numpy())
                fb.append(frame_num)
                sb.append(bridge.last_score)
                spb.append(0.0)
                if len(fb) > 300:
                    fb.pop(0)
                    sb.pop(0)
                    spb.pop(0)
                line_score.set_data(fb, sb)
                line_spread.set_data(fb, spb)
                ax_info.relim()
                ax_info.autoscale_view()
                return (_img, line_score, line_spread)

            seq = anim.FuncAnimation(
                fig,
                step_and_render,
                fargs=(img,),
                frames=args.frames,
                blit=True,
                interval=8,
                repeat=False,
            )
            plt.show()
