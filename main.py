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
LIDAR_POINTS = wp.constant(1081)
RANGE_MIN = wp.constant(0.1)
RANGE_MAX = wp.constant(20.0)
BEAM_STRIDE = wp.constant(8)
LIDAR_MIN_ANGLE = wp.constant(-LIDAR_FOV * 0.5)
LIDAR_INCREMENT = wp.constant(LIDAR_FOV / LIDAR_POINTS)
NUM_MATCH_BEAMS = wp.constant(LIDAR_POINTS // BEAM_STRIDE)


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
    i, j, k, b = wp.tid()

    r = ranges[b * BEAM_STRIDE]
    if not RANGE_MIN <= r <= RANGE_MAX:
        return

    x = center[0] + (float(i) - float(n_xy - 1) * 0.5) * step_xy
    y = center[1] + (float(j) - float(n_xy - 1) * 0.5) * step_xy
    θ = center[2] + (float(k) - float(n_theta - 1) * 0.5) * step_theta

    cθ = wp.cos(θ)
    sθ = wp.sin(θ)
    ct = cos_table[b]
    st = sin_table[b]
    ca = ct * cθ - st * sθ
    sa = st * cθ + ct * sθ

    gx = (x + r * ca - ORIGIN[0]) * INV_RESOLUTION
    gy = (y + r * sa - ORIGIN[1]) * INV_RESOLUTION

    c = i * n_xy * n_theta + j * n_theta + k
    wp.atomic_add(scores, c, sample_float(likelihood, gx, gy))


@wp.kernel
def integrate(ranges: wp.array[float], pose: wp.vec3, logodds: wp.array2d[float]):
    i = wp.tid()
    r = ranges[i]

    if not RANGE_MIN <= r <= RANGE_MAX:
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

    if not (1 <= x < GRID_WIDTH - 1 and 1 <= y < GRID_HEIGHT - 1):
        return

    s = 0.0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            v = wp.clamp(logodds[y + dy, x + dx], L_MIN, L_MAX)
            s += wp.max(0.0, v)

    likelihood[y, x] = s * (1.0 / 9.0)


class Bridge:
    def __init__(self):
        self.stages = [
            (0.30, wp.radians(15.0), 0.05, wp.radians(1.0)),
            (0.05, wp.radians(2.0), 0.01, wp.radians(0.2)),
        ]
        self.kf_d2 = 0.2 * 0.2
        self.kf_dtheta = wp.radians(10.0)

        self.logodds = wp.zeros((GRID_HEIGHT, GRID_WIDTH))
        self.likelihood = wp.zeros((GRID_HEIGHT, GRID_WIDTH))
        self.ranges = wp.zeros(LIDAR_POINTS)

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

        for hxy, hth, sxy, sth in self.stages:
            n_xy = int(2 * hxy / sxy) + 1
            n_theta = int(2 * hth / sth) + 1
            count = n_xy * n_xy * n_theta

            self.scores.zero_()
            wp.launch(
                search,
                dim=(n_xy, n_xy, n_theta, NUM_MATCH_BEAMS),
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

            best = int(self.scores.numpy()[:count].argmax())
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

        return p

    def step(self, ranges_np, odom_delta):
        with wp.ScopedTimer("step"):
            wp.copy(self.ranges, wp.array(ranges_np.astype(np.float32)))

            if self.first:
                self.pose = odom_delta.astype(np.float32)
                self.first = False
                self.integrate_scan()
                return self.pose

            seed = self.pose + odom_delta.astype(np.float32)
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

    with wp.ScopedDevice(args.device):
        bridge = Bridge()

        # synthetic stand-in scan + zero odom delta; replace with ROS callbacks on the car
        fake_ranges = np.full(int(LIDAR_POINTS), 5.0, dtype=np.float32)
        fake_odom = np.zeros(3, dtype=np.float32)

        if args.headless:
            for _ in range(args.num_frames):
                bridge.step(fake_ranges, fake_odom)
        else:
            import matplotlib
            import matplotlib.animation as anim
            import matplotlib.pyplot as plt

            fig = plt.figure()

            img = plt.imshow(
                bridge.likelihood.numpy(),
                origin="lower",
                animated=True,
                interpolation="antialiased",
            )
            img.set_norm(matplotlib.colors.Normalize(0.0, float(L_MAX)))

            def step_and_render(frame_num, img):
                bridge.step(fake_ranges, fake_odom)
                img.set_array(bridge.likelihood.numpy())
                return (img,)

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
