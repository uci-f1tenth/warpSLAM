import numpy as np

import warp as wp

# map
GRID_WIDTH = wp.constant(1024)
GRID_HEIGHT = wp.constant(1024)
RESOLUTION = wp.constant(0.05)
INV_RESOLUTION = wp.constant(1.0 / 0.05)
ORIGIN = wp.constant(wp.vec2(-25.6, -25.6))

# log-odds
L_OCC = wp.constant(0.85)
L_FREE = wp.constant(-0.4)
L_MIN = wp.constant(-5.0)
L_MAX = wp.constant(5.0)

# lidar
LIDAR_FOV = wp.constant(wp.radians(270.0))
LIDAR_POINTS = wp.constant(1080)
RANGE_MIN = wp.constant(0.06)
RANGE_MAX = wp.constant(10.0)
BEAM_STRIDE = wp.constant(8)
LIDAR_MIN_ANGLE = wp.constant(-LIDAR_FOV * 0.5)
LIDAR_INCREMENT = wp.constant(LIDAR_FOV / LIDAR_POINTS)
NUM_MATCH_BEAMS = wp.constant(LIDAR_POINTS // BEAM_STRIDE)
INV_9 = wp.constant(1.0 / 9.0)


@wp.func
def lookup_float(f: wp.array2d[float], x: int, y: int):
    x = wp.clamp(x, 0, GRID_WIDTH - 1)
    y = wp.clamp(y, 0, GRID_HEIGHT - 1)
    return f[y, x]


@wp.func
def sample_float(f: wp.array2d[float], x: float, y: float):
    lx = int(wp.floor(x))
    ly = int(wp.floor(y))
    tx = x - float(lx)
    ty = y - float(ly)
    s0 = wp.lerp(lookup_float(f, lx, ly), lookup_float(f, lx + 1, ly), tx)
    s1 = wp.lerp(lookup_float(f, lx, ly + 1), lookup_float(f, lx + 1, ly + 1), tx)
    s = wp.lerp(s0, s1, ty)
    return s


@wp.kernel
def search(
    ranges: wp.array[float],
    sin_table: wp.array[float],
    cos_table: wp.array[float],
    likelihood: wp.array2d[float],
    center: wp.vec3,
    step_xy: float,
    step_theta: float,
    n_xy: int,
    n_theta: int,
    scores: wp.array[float],
):
    i, j, k = wp.tid()
    x = center[0] + (float(i) - float(n_xy - 1) * 0.5) * step_xy
    y = center[1] + (float(j) - float(n_xy - 1) * 0.5) * step_xy
    θ = center[2] + (float(k) - float(n_theta - 1) * 0.5) * step_theta
    cθ = wp.cos(θ)
    sθ = wp.sin(θ)

    total = float(0.0)
    for b in range(NUM_MATCH_BEAMS):
        r = ranges[b * BEAM_STRIDE]
        if not (RANGE_MIN <= r < RANGE_MAX):
            continue
        ct = cos_table[b]
        st = sin_table[b]
        ca = ct * cθ - st * sθ
        sa = st * cθ + ct * sθ
        gx = (x + r * ca - ORIGIN[0]) * INV_RESOLUTION
        gy = (y + r * sa - ORIGIN[1]) * INV_RESOLUTION
        total += sample_float(likelihood, gx, gy)

    scores[i * n_xy * n_theta + j * n_theta + k] = total


@wp.kernel
def integrate(ranges: wp.array[float], pose: wp.vec3, logodds: wp.array2d[float]):
    i = wp.tid()
    r = ranges[i]

    if not (RANGE_MIN <= r < RANGE_MAX):
        return

    a = LIDAR_MIN_ANGLE + LIDAR_INCREMENT * float(i) + pose[2]
    ca = wp.cos(a)
    sa = wp.sin(a)

    x0 = (pose[0] - ORIGIN[0]) * INV_RESOLUTION
    y0 = (pose[1] - ORIGIN[1]) * INV_RESOLUTION
    x1 = (pose[0] + r * ca - ORIGIN[0]) * INV_RESOLUTION
    y1 = (pose[1] + r * sa - ORIGIN[1]) * INV_RESOLUTION

    dx = x1 - x0
    dy = y1 - y0
    steps = int(wp.max(wp.abs(dx), wp.abs(dy))) + 1
    ux = dx / float(steps)
    uy = dy / float(steps)

    gx = x0
    gy = y0
    for s in range(steps):
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
def blur(logodds: wp.array2d[float], likelihood: wp.array2d[float]):
    x, y = wp.tid()

    logodds[y, x] = wp.clamp(logodds[y, x], L_MIN, L_MAX)

    if not (1 <= x < GRID_WIDTH - 1 and 1 <= y < GRID_HEIGHT - 1):
        return

    s = 0.0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            s += wp.clamp(logodds[y + dy, x + dx], 0.0, L_MAX)

    likelihood[y, x] = s * INV_9


class Bridge:
    def __init__(self):
        wp.init()
        self.stages = [
            (0.30, wp.radians(15.0), 0.05, wp.radians(1.0)),
            (0.05, wp.radians(2.0), 0.01, wp.radians(0.2)),
        ]
        self.kf_d2 = 0.2 * 0.2
        self.kf_dtheta = wp.radians(10.0)

        self.logodds = wp.zeros((GRID_HEIGHT, GRID_WIDTH))
        self.likelihood = wp.zeros((GRID_HEIGHT, GRID_WIDTH))
        self.ranges = wp.zeros(LIDAR_POINTS)
        self._ranges_buf = np.empty(int(LIDAR_POINTS), dtype=np.float32)

        step = LIDAR_FOV / LIDAR_POINTS
        a = -LIDAR_FOV * 0.5 + step * BEAM_STRIDE * np.arange(
            NUM_MATCH_BEAMS, dtype=np.float32
        )
        self.sin_table = wp.array(np.sin(a))
        self.cos_table = wp.array(np.cos(a))
        max_c = 0
        for hxy, hth, sxy, sth in self.stages:
            n_xy = int(2 * hxy / sxy) + 1
            n_theta = int(2 * hth / sth) + 1
            max_c = max(max_c, n_xy * n_xy * n_theta)
        self.scores = wp.zeros(max_c, dtype=float)

        self.pose = np.zeros(3, dtype=np.float32)
        self.last_kf = self.pose.copy()
        self.first = True
        self._last_score = 0.0
        self._last_n_valid = 0
        self._last_spread = 0.0

    def integrate_scan(self):
        wp.launch(
            integrate,
            dim=LIDAR_POINTS,
            inputs=[self.ranges, wp.vec3(*self.pose), self.logodds],
        )
        wp.launch(
            blur,
            dim=(int(GRID_WIDTH), int(GRID_HEIGHT)),
            inputs=[self.logodds, self.likelihood],
        )
        self.last_kf = self.pose.copy()

    def match(self, seed):
        p = seed.astype(np.float32)

        for stage_idx, (hxy, hth, sxy, sth) in enumerate(self.stages):
            n_xy = int(2 * hxy / sxy) + 1
            n_theta = int(2 * hth / sth) + 1
            count = n_xy * n_xy * n_theta

            wp.launch(
                search,
                dim=(n_xy, n_xy, n_theta),
                inputs=[
                    self.ranges,
                    self.sin_table,
                    self.cos_table,
                    self.likelihood,
                    wp.vec3(*p),
                    sxy,
                    sth,
                    n_xy,
                    n_theta,
                    self.scores,
                ],
            )

            scores_np = self.scores.numpy()[:count]
            best = int(scores_np.argmax())
            i = best // (n_xy * n_theta)
            j = (best // n_theta) % n_xy
            k = best % n_theta
            p = np.array(
                [
                    p[0] + (i - (n_xy - 1) * 0.5) * sxy,
                    p[1] + (j - (n_xy - 1) * 0.5) * sxy,
                    p[2] + (k - (n_theta - 1) * 0.5) * sth,
                ],
                dtype=np.float32,
            )

            if stage_idx == len(self.stages) - 1:
                self._last_score = float(scores_np[best])
                nonzero = scores_np[scores_np > 0.0]
                self._last_n_valid = int(nonzero.size)
                self._last_spread = (
                    self._last_score - float(np.median(nonzero))
                    if nonzero.size > 0
                    else 0.0
                )

        return p

    def step(self, ranges_np, odom_delta):
        with wp.ScopedTimer("step"):
            self._ranges_buf[:] = ranges_np
            self.ranges.assign(self._ranges_buf)

            if self.first:
                self.pose = odom_delta.astype(np.float32)
                self.first = False
                self.integrate_scan()
                return self.pose

            seed = self.pose + odom_delta.astype(np.float32)
            hit = (ranges_np >= float(RANGE_MIN)) & (ranges_np < float(RANGE_MAX))
            if not np.any(hit):
                self.pose = seed
                return self.pose

            self.pose = self.match(seed)

            d = self.pose - self.last_kf
            if (d[0] * d[0] + d[1] * d[1]) > self.kf_d2 or abs(d[2]) > self.kf_dtheta:
                self.integrate_scan()

            return self.pose


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override the default Warp device."
    )
    parser.add_argument(
        "--num-frames", type=int, default=100000, help="Total number of frames."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    LIDAR_POINTS_I = int(LIDAR_POINTS)
    LIDAR_STEP = float(LIDAR_FOV) / LIDAR_POINTS_I
    R_MIN = float(RANGE_MIN)
    R_MAX = float(RANGE_MAX)
    LIDAR_MIN_ANGLE_VAL = float(LIDAR_MIN_ANGLE)

    ROOM_X_MIN, ROOM_X_MAX = -2.0, 20.0
    ROOM_Y_MIN, ROOM_Y_MAX = -3.0, 3.0

    def generate_scan(pos_x, pos_y=0.0):
        angles = LIDAR_MIN_ANGLE_VAL + LIDAR_STEP * np.arange(LIDAR_POINTS_I)
        ca = np.cos(angles)
        sa = np.sin(angles)
        eps = 1e-6

        t = np.full(LIDAR_POINTS_I, np.inf, dtype=np.float32)

        m = ca > eps
        t[m] = np.minimum(t[m], (ROOM_X_MAX - pos_x) / ca[m])
        m = ca < -eps
        t[m] = np.minimum(t[m], (ROOM_X_MIN - pos_x) / ca[m])
        m = sa > eps
        t[m] = np.minimum(t[m], (ROOM_Y_MAX - pos_y) / sa[m])
        m = sa < -eps
        t[m] = np.minimum(t[m], (ROOM_Y_MIN - pos_y) / sa[m])

        bad = (t < R_MIN) | (t >= R_MAX) | ~np.isfinite(t)
        t[bad] = 0.0
        return t

    with wp.ScopedDevice(args.device):
        bridge = Bridge()

        fake_odom = np.array([0.05, 0.0, 0.0], dtype=np.float32)

        if args.headless:
            for frame in range(args.num_frames):
                fake_ranges = generate_scan((1 + frame) * 0.05)
                bridge.step(fake_ranges, fake_odom)
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
            (line_score,) = ax_info.plot([], [], label="match score")
            (line_spread,) = ax_info.plot([], [], label="score spread")
            ax_info.legend()

            frame_buf = []
            score_buf = []
            spread_buf = []

            def step_and_render(frame_num, _img):
                fake_ranges = generate_scan((1 + frame_num) * 0.05)
                bridge.step(fake_ranges, fake_odom)
                _img.set_array(bridge.likelihood.numpy())

                frame_buf.append(frame_num)
                score_buf.append(bridge._last_score)
                spread_buf.append(bridge._last_spread)
                if len(frame_buf) > 300:
                    frame_buf.pop(0)
                    score_buf.pop(0)
                    spread_buf.pop(0)
                line_score.set_data(frame_buf, score_buf)
                line_spread.set_data(frame_buf, spread_buf)
                ax_info.relim()
                ax_info.autoscale_view()

                return (_img, line_score, line_spread)

            seq = anim.FuncAnimation(
                fig,
                step_and_render,
                fargs=(img,),
                frames=args.num_frames,
                blit=True,
                interval=8,
                repeat=False,
            )

            plt.show()
