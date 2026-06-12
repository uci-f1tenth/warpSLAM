import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

GRID_SIZE = wp.constant(2048)
RESOLUTION = wp.constant(0.05)
INV_RESOLUTION = wp.constant(20.0)
MAP_ORIGIN = wp.constant(-51.2)
LOGODDS_OCCUPIED = wp.constant(0.85)
LOGODDS_FREE = wp.constant(-0.4)
LOGODDS_MIN = wp.constant(-2.5)
LOGODDS_MAX = wp.constant(2.5)
RANGE_MIN = wp.constant(0.05)
RANGE_MAX = wp.constant(30.0)
BEAM_STRIDE = wp.constant(8)
GN_ITERATIONS = wp.constant(10)
LIKELIHOOD_SIGMA_INV2 = wp.constant(0.5)
OCCUPIED_GATE = wp.constant(0.5)
FREE_GATE = wp.constant(-0.25)
BACKOFF_DIST = wp.constant(0.30)
GN_MAX_STEP = wp.constant(0.05)
GN_MAX_ROTATION = wp.constant(
    0.05235988
)  # ~3 deg, GN overshoots on fast spins if you let it take more


@wp.func
def deskewed_endpoint(
    frac: float,
    motion_x: float,
    motion_y: float,
    motion_th: float,
    end_x: float,
    end_y: float,
):
    # Small-angle rotation: beams late in the sweep get more of the motion correction.
    a = frac * motion_th
    return frac * motion_x + end_x - a * end_y, frac * motion_y + a * end_x + end_y


@wp.func
def world_to_grid(world_x: float, world_y: float):
    return (world_x - MAP_ORIGIN) * INV_RESOLUTION, (
        world_y - MAP_ORIGIN
    ) * INV_RESOLUTION


@wp.func
def nearest_cell(grid_x: float, grid_y: float):
    ix = wp.clamp(int(wp.floor(grid_x + 0.5)), 0, GRID_SIZE - 1)
    iy = wp.clamp(int(wp.floor(grid_y + 0.5)), 0, GRID_SIZE - 1)
    return ix, iy


@wp.func
def hits_front_face(
    logodds: wp.array2d(dtype=float),
    sensor_x: float,
    sensor_y: float,
    offset_x: float,
    offset_y: float,
):
    # endpoint must be occupied AND the cell just short of it free, otherwise
    # thin walls get matched from the wrong side and the pose drags through them
    gx, gy = world_to_grid(sensor_x + offset_x, sensor_y + offset_y)
    end_ix, end_iy = nearest_cell(gx, gy)
    dist = wp.sqrt(offset_x * offset_x + offset_y * offset_y)
    backoff = 1.0 - wp.min(BACKOFF_DIST, 0.5 * dist) / dist
    bx, by = world_to_grid(sensor_x + offset_x * backoff, sensor_y + offset_y * backoff)
    pre_ix, pre_iy = nearest_cell(bx, by)
    return (
        logodds[end_iy, end_ix] > OCCUPIED_GATE and logodds[pre_iy, pre_ix] < FREE_GATE
    )


@wp.func
def bilinear_sample(grid: wp.array2d(dtype=float), x: float, y: float):
    fx = wp.floor(x)
    fy = wp.floor(y)
    ix = wp.clamp(int(fx), 0, GRID_SIZE - 2)
    iy = wp.clamp(int(fy), 0, GRID_SIZE - 2)
    top = wp.lerp(grid[iy, ix], grid[iy, ix + 1], x - fx)
    bottom = wp.lerp(grid[iy + 1, ix], grid[iy + 1, ix + 1], x - fx)
    return wp.lerp(top, bottom, y - fy)


@wp.func
def bilinear_sample_grad(grid: wp.array2d(dtype=float), x: float, y: float):
    fx = wp.floor(x)
    fy = wp.floor(y)
    tx = x - fx
    ty = y - fy
    ix = wp.clamp(int(fx), 0, GRID_SIZE - 2)
    iy = wp.clamp(int(fy), 0, GRID_SIZE - 2)
    v00 = grid[iy, ix]
    v10 = grid[iy, ix + 1]
    v01 = grid[iy + 1, ix]
    v11 = grid[iy + 1, ix + 1]
    value = wp.lerp(wp.lerp(v00, v10, tx), wp.lerp(v01, v11, tx), ty)
    grad_x = wp.lerp(v10 - v00, v11 - v01, ty)
    grad_y = wp.lerp(v01 - v00, v11 - v10, tx)
    return value, grad_x, grad_y


@wp.func
def beam_endpoint_world(
    ranges: wp.array(dtype=float),
    beam: int,
    angle_min: float,
    angle_inc: float,
    inv_n: float,
    control: wp.array(dtype=float),
    cos_th: float,
    sin_th: float,
):
    dist = ranges[beam]
    angle = angle_min + angle_inc * float(beam)
    sweep_frac = float(beam) * inv_n - 0.5  # -0.5..0.5 across the scan
    lx, ly = deskewed_endpoint(
        sweep_frac,
        control[3],
        control[4],
        control[5],
        dist * wp.cos(angle),
        dist * wp.sin(angle),
    )
    return lx * cos_th - ly * sin_th, lx * sin_th + ly * cos_th


@wp.kernel
def coarse_search_kernel(
    ranges: wp.array(dtype=float),
    likelihood: wp.array2d(dtype=float),
    logodds: wp.array2d(dtype=float),
    control: wp.array(dtype=float),
    angle_min: float,
    angle_inc: float,
    inv_n: float,
    step_xy: float,
    step_th: float,
    weight_trans: float,
    weight_rot: float,
    n_xy: int,
    n_th: int,
    n_beams_used: int,
    scores: wp.array(dtype=float),
):
    i, j, k = wp.tid()
    half = float(n_xy - 1) * 0.5
    dx = (float(i) - half) * step_xy
    dy = (float(j) - half) * step_xy
    dth = (float(k) - float(n_th - 1) * 0.5) * step_th
    x = control[0] + dx
    y = control[1] + dy
    cos_th = wp.cos(control[2] + dth)
    sin_th = wp.sin(control[2] + dth)
    total = float(0.0)
    # every 8th beam is plenty here, refine uses all of them anyway
    for b in range(n_beams_used):
        if ranges[b * BEAM_STRIDE] >= RANGE_MIN and ranges[b * BEAM_STRIDE] < RANGE_MAX:
            ox, oy = beam_endpoint_world(
                ranges,
                b * BEAM_STRIDE,
                angle_min,
                angle_inc,
                inv_n,
                control,
                cos_th,
                sin_th,
            )
            if hits_front_face(logodds, x, y, ox, oy):
                gx, gy = world_to_grid(x + ox, y + oy)
                total += wp.max(bilinear_sample(likelihood, gx, gy), 0.0)
    prior = wp.exp(
        -(dx * dx + dy * dy) * weight_trans * weight_trans
        - dth * dth * weight_rot * weight_rot
    )
    scores[(i * n_xy + j) * n_th + k] = total * prior


@wp.kernel
def pick_best_kernel(
    scores: wp.array(dtype=float),
    control: wp.array(dtype=float),
    step_xy: float,
    step_th: float,
    n_xy: int,
    n_th: int,
    pose: wp.array(dtype=float),
):
    tid = wp.tid()
    best_score = float(-1.0)
    best_idx = int(0)
    i = tid
    while i < n_xy * n_xy * n_th:
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
        i += wp.block_dim()
    lane = wp.tile_argmax(wp.tile(best_score))[0]
    candidates = wp.tile(best_idx)
    winner = candidates[lane]
    if tid == 0:
        half = float(n_xy - 1) * 0.5
        pose[0] = control[0] + (float(winner // (n_th * n_xy)) - half) * step_xy
        pose[1] = control[1] + (float((winner // n_th) % n_xy) - half) * step_xy
        pose[2] = control[2] + (float(winner % n_th) - float(n_th - 1) * 0.5) * step_th


@wp.kernel
def gauss_newton_refine_kernel(
    ranges: wp.array(dtype=float),
    likelihood: wp.array2d(dtype=float),
    logodds: wp.array2d(dtype=float),
    n_beams: int,
    angle_min: float,
    angle_inc: float,
    inv_n: float,
    weight_trans: float,
    weight_rot: float,
    control: wp.array(dtype=float),
    pose: wp.array(dtype=float),
):
    tid = wp.tid()
    px = pose[0]
    py = pose[1]
    pth = pose[2]
    fit = float(0.0)
    coverage = float(0.0)
    for _ in range(GN_ITERATIONS):
        hessian = wp.mat33()
        gradient = wp.vec3()
        stats = wp.vec3()  # (matched count, likelihood sum, valid count)
        cos_th = wp.cos(pth)
        sin_th = wp.sin(pth)
        b = tid
        while b < n_beams:
            if ranges[b] >= RANGE_MIN and ranges[b] < RANGE_MAX:
                stats += wp.vec3(0.0, 0.0, 1.0)
                ox, oy = beam_endpoint_world(
                    ranges, b, angle_min, angle_inc, inv_n, control, cos_th, sin_th
                )
                if hits_front_face(logodds, px, py, ox, oy):
                    gx, gy = world_to_grid(px + ox, py + oy)
                    value, du, dv = bilinear_sample_grad(likelihood, gx, gy)
                    jac = wp.vec3(du, dv, du * (-oy) + dv * ox) * -float(INV_RESOLUTION)
                    hessian += wp.outer(jac, jac)
                    gradient += jac * (1.0 - wp.max(value, 0.0))
                    stats += wp.vec3(1.0, wp.max(value, 0.0), 0.0)
            b += wp.block_dim()
        hessian_sum = wp.tile_sum(wp.tile(hessian, preserve_type=True))[0]
        gradient_sum = wp.tile_sum(wp.tile(gradient, preserve_type=True))[0]
        totals = wp.tile_sum(wp.tile(stats, preserve_type=True))[0]
        n_matched = totals[0]
        fit = totals[1] / wp.max(totals[2], 1.0)
        coverage = n_matched / wp.max(totals[2], 1.0)
        if n_matched >= 3.0:
            dth = pth - control[2]
            dth = wp.atan2(wp.sin(dth), wp.cos(dth))
            # damping doubles as a soft pull back toward the odom seed
            A = hessian_sum / n_matched + wp.diag(
                wp.vec3(weight_trans, weight_trans, weight_rot)
            )
            rhs = gradient_sum / n_matched + wp.vec3(
                weight_trans * (px - control[0]),
                weight_trans * (py - control[1]),
                weight_rot * dth,
            )
            delta = -(wp.inverse(A) * rhs)
            # clamp the step, single bad iteration can throw the whole track off
            step_len = wp.sqrt(delta[0] * delta[0] + delta[1] * delta[1])
            if step_len > GN_MAX_STEP:
                scale = GN_MAX_STEP / step_len
                delta = wp.vec3(delta[0] * scale, delta[1] * scale, delta[2])
            if wp.abs(delta[2]) > GN_MAX_ROTATION:
                delta = wp.vec3(
                    delta[0], delta[1], delta[2] * GN_MAX_ROTATION / wp.abs(delta[2])
                )
            px += delta[0]
            py += delta[1]
            pth += delta[2]
    if tid == 0:
        pose[0] = px
        pose[1] = py
        pose[2] = pth
        pose[3] = fit
        pose[4] = coverage


@wp.kernel
def integrate_scan_kernel(
    ranges: wp.array(dtype=float),
    angle_min: float,
    angle_inc: float,
    inv_n: float,
    control: wp.array(dtype=float),
    pose: wp.array(dtype=float),
    logodds: wp.array2d(dtype=float),
    likelihood: wp.array2d(dtype=float),
):
    i = wp.tid()
    if ranges[i] < RANGE_MIN or ranges[i] >= RANGE_MAX:
        return
    cos_th = wp.cos(pose[2])
    sin_th = wp.sin(pose[2])
    sweep_frac = float(i) * inv_n - 0.5
    # Ray start is the (deskewed) sensor position when this beam fired.
    start_gx, start_gy = world_to_grid(
        pose[0]
        + (sweep_frac * control[3]) * cos_th
        - (sweep_frac * control[4]) * sin_th,
        pose[1]
        + (sweep_frac * control[3]) * sin_th
        + (sweep_frac * control[4]) * cos_th,
    )
    ox, oy = beam_endpoint_world(
        ranges, i, angle_min, angle_inc, inv_n, control, cos_th, sin_th
    )
    end_gx, end_gy = world_to_grid(pose[0] + ox, pose[1] + oy)

    n_steps = int(wp.max(wp.abs(end_gx - start_gx), wp.abs(end_gy - start_gy))) + 1
    step_x = (end_gx - start_gx) / float(n_steps)
    step_y = (end_gy - start_gy) / float(n_steps)
    gx = start_gx
    gy = start_gy
    for _ in range(n_steps + 1):
        ix = int(wp.floor(gx))
        iy = int(wp.floor(gy))
        if 0 <= ix and ix < GRID_SIZE and 0 <= iy and iy < GRID_SIZE:
            wp.atomic_add(logodds, iy, ix, LOGODDS_FREE)
            wp.atomic_max(logodds, iy, ix, LOGODDS_MIN)
        gx += step_x
        gy += step_y

    # the raymarch above already hit this cell with a free update, back it out
    hit_x = int(wp.floor(end_gx))
    hit_y = int(wp.floor(end_gy))
    if 0 <= hit_x and hit_x < GRID_SIZE and 0 <= hit_y and hit_y < GRID_SIZE:
        wp.atomic_add(logodds, hit_y, hit_x, LOGODDS_OCCUPIED - LOGODDS_FREE)
        wp.atomic_min(logodds, hit_y, hit_x, LOGODDS_MAX)

    # 5x5 gaussian splat around the hit for the likelihood field
    for dj in range(-2, 3):
        for di in range(-2, 3):
            ix = hit_x + di
            iy = hit_y + dj
            if 0 <= ix and ix < GRID_SIZE and 0 <= iy and iy < GRID_SIZE:
                du = float(ix) - end_gx
                dv = float(iy) - end_gy
                wp.atomic_max(
                    likelihood,
                    iy,
                    ix,
                    wp.exp(-(du * du + dv * dv) * LIKELIHOOD_SIGMA_INV2),
                )


@wp.kernel
def occupancy_export_kernel(
    logodds: wp.array2d(dtype=float), out: wp.array2d(dtype=wp.int8)
):
    iy, ix = wp.tid()
    value = logodds[iy, ix]
    occ = int(-1)  # unknown
    if wp.abs(value) > 0.1:
        prob = 1.0 / (1.0 + wp.exp(-wp.clamp(value, -10.0, 10.0)))
        occ = int(prob * 100.0)
    out[iy, ix] = wp.int8(occ)


def _wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class Bridge:
    WEIGHT_TRANS, WEIGHT_ROT = 0.5, 2.0
    KEYFRAME_DIST2, KEYFRAME_DTH = 0.20**2, np.radians(10.0)
    MIN_MOVE_XY, MIN_MOVE_TH = 0.02, np.radians(0.5)
    COARSE_HALF_XY, COARSE_HALF_TH = 0.30, np.radians(15.0)
    COARSE_STEP_XY, COARSE_STEP_TH = 0.05, np.radians(1.0)
    COARSE_WEIGHT_TRANS, COARSE_WEIGHT_ROT = 2.0, 2.0
    FIT_FOR_MAPPING = 0.50
    COVERAGE_NEW_AREA = 0.35
    FIT_RETRY = 0.40
    COVERAGE_MIN = 0.15
    RETRY_SCALE = 3.0
    BLOCK_DIM = 256

    def __init__(self, device=None):
        wp.init()
        try:
            has_cuda = wp.get_cuda_device_count() > 0
        except Exception:
            has_cuda = False
        self.device = device or ("cuda" if has_cuda else "cpu")
        wp.set_device(self.device)
        self._is_cuda = str(self.device).startswith("cuda")

        self.logodds = wp.zeros((int(GRID_SIZE), int(GRID_SIZE)))
        self.likelihood = wp.zeros((int(GRID_SIZE), int(GRID_SIZE)))
        self._occ_grid = wp.zeros((int(GRID_SIZE), int(GRID_SIZE)), dtype=wp.int8)
        # control = [seed_x, seed_y, seed_th, motion_x, motion_y, motion_th]
        self._control = wp.zeros(6, dtype=float)
        # pose = [x, y, th, fit, coverage]
        self._pose_device = wp.zeros(5, dtype=float)
        self._n_xy = 2 * int(self.COARSE_HALF_XY / self.COARSE_STEP_XY) + 1
        self._n_th = 2 * int(self.COARSE_HALF_TH / self.COARSE_STEP_TH) + 1
        self._scores = wp.zeros(self._n_xy * self._n_xy * self._n_th, dtype=float)

        self._n_beams = self._n_beams_coarse = 0
        self._angle_min = self._angle_inc = self._inv_n = 0.0
        self._ranges = None
        self._graph = None
        self._graph_tried = False
        self.pose = np.zeros(3, dtype=np.float32)
        self._keyframe_pose = self.pose.copy()
        self._fit = self._coverage = 1.0
        self._first_scan = True
        self._integrations = 0

    @property
    def integrations(self):
        return self._integrations

    def configure(self, n_beams, angle_min, angle_increment):
        n = int(n_beams)
        if (
            n == self._n_beams
            and abs(angle_min - self._angle_min) < 1e-9
            and abs(angle_increment - self._angle_inc) < 1e-12
        ):
            return False
        if n < BEAM_STRIDE * 4:
            raise ValueError(f"scan too small: {n} beams")
        self._n_beams = n
        self._angle_min = float(angle_min)
        self._angle_inc = float(angle_increment)
        self._inv_n = 1.0 / float(n - 1)
        self._n_beams_coarse = n // int(BEAM_STRIDE)
        self._ranges = wp.zeros(n)
        # Scan geometry changed, so any captured CUDA graph is stale.
        self._graph = None
        self._graph_tried = False
        return True

    def step(self, ranges, odom, deskew_frac=0.0):
        scan = np.ascontiguousarray(ranges, dtype=np.float32)
        odom = np.asarray(odom, dtype=np.float32)
        self._ranges.assign(scan)

        if self._first_scan:
            self._first_scan = False
            self.pose = odom.copy()
            self._keyframe_pose = self.pose.copy()
            self._control.assign(
                np.array([*self.pose, 0.0, 0.0, 0.0], dtype=np.float32)
            )
            self._pose_device.assign(np.array([*self.pose, 1.0, 1.0], dtype=np.float32))
            self._integrate()
            return self.pose

        seed = self.pose + odom
        moved = (
            np.linalg.norm(odom[:2]) > self.MIN_MOVE_XY
            or abs(odom[2]) > self.MIN_MOVE_TH
        )
        has_returns = bool(
            np.any((scan >= float(RANGE_MIN)) & (scan < float(RANGE_MAX)))
        )

        # odom delta is in the old base frame, rotate into sensor frame for deskew
        frac = float(np.clip(deskew_frac, 0.0, 1.0))
        c, s = np.cos(self.pose[2]), np.sin(self.pose[2])
        motion_x = (odom[0] * c + odom[1] * s) * frac
        motion_y = (-odom[0] * s + odom[1] * c) * frac
        self._control.assign(
            np.array([*seed, motion_x, motion_y, odom[2] * frac], dtype=np.float32)
        )

        if has_returns and moved:
            self._run()
            result = self._pose_device.numpy()
            # weak fit on terrain we've mapped before -> widen the search and retry.
            # TODO: this fires a lot near glass doors, maybe gate on coverage too
            if result[3] < self.FIT_RETRY and result[4] > self.COVERAGE_MIN:
                first_try = result.copy()
                self._coarse_search(self.RETRY_SCALE)
                wp.copy(self._control, self._pose_device, count=3)
                self._run()
                result = self._pose_device.numpy()
                if result[3] <= first_try[3]:
                    result = first_try
                    self._pose_device.assign(first_try)
            self.pose = result[:3].astype(np.float32).copy()
            self._fit, self._coverage = float(result[3]), float(result[4])
        else:
            self.pose = seed.copy()
            self._pose_device.assign(
                np.array([*self.pose, self._fit, self._coverage], dtype=np.float32)
            )

        self.pose[2] = _wrap_angle(float(self.pose[2]))
        delta = self.pose - self._keyframe_pose
        delta[2] = _wrap_angle(float(delta[2]))
        moved_enough = (
            delta[0] * delta[0] + delta[1] * delta[1] > self.KEYFRAME_DIST2
            or abs(delta[2]) > self.KEYFRAME_DTH
        )
        # map if the match is solid, OR coverage is low (new territory, have to map
        # something or we never localize there again)
        if moved_enough and (
            self._fit >= self.FIT_FOR_MAPPING or self._coverage < self.COVERAGE_NEW_AREA
        ):
            self._integrate()
        return self.pose

    def snapshot(self):
        occ = self.occupancy()
        known = occ != -1
        rows = np.flatnonzero(known.any(axis=1))
        if rows.size == 0:
            return None
        cols = np.flatnonzero(known.any(axis=0))
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        return occ[y0:y1, x0:x1], x0, y0

    def occupancy(self):
        if self._is_cuda:
            wp.launch(
                occupancy_export_kernel,
                dim=(int(GRID_SIZE), int(GRID_SIZE)),
                inputs=[self.logodds, self._occ_grid],
            )
            return self._occ_grid.numpy()
        # cpu fallback, numpy is fine at this size
        lo = self.logodds.numpy()
        out = np.full(lo.shape, -1, dtype=np.int8)
        known = np.abs(lo) > 0.1
        if known.any():
            p = 1.0 / (1.0 + np.exp(-np.clip(lo[known], -10.0, 10.0)))
            out[known] = (p * 100.0).astype(np.int8)
        return out

    def _coarse_search(self, scale):
        wp.launch(
            coarse_search_kernel,
            dim=(self._n_xy, self._n_xy, self._n_th),
            inputs=[
                self._ranges,
                self.likelihood,
                self.logodds,
                self._control,
                self._angle_min,
                self._angle_inc,
                self._inv_n,
                self.COARSE_STEP_XY * scale,
                self.COARSE_STEP_TH * scale,
                self.COARSE_WEIGHT_TRANS / scale,
                self.COARSE_WEIGHT_ROT / scale,
                self._n_xy,
                self._n_th,
                self._n_beams_coarse,
                self._scores,
            ],
        )
        wp.launch_tiled(
            pick_best_kernel,
            dim=[1],
            block_dim=self.BLOCK_DIM,
            inputs=[
                self._scores,
                self._control,
                self.COARSE_STEP_XY * scale,
                self.COARSE_STEP_TH * scale,
                self._n_xy,
                self._n_th,
                self._pose_device,
            ],
        )

    def _pipeline(self):
        self._coarse_search(1.0)
        wp.launch_tiled(
            gauss_newton_refine_kernel,
            dim=[1],
            block_dim=self.BLOCK_DIM,
            inputs=[
                self._ranges,
                self.likelihood,
                self.logodds,
                self._n_beams,
                self._angle_min,
                self._angle_inc,
                self._inv_n,
                float(self.WEIGHT_TRANS),
                float(self.WEIGHT_ROT),
                self._control,
                self._pose_device,
            ],
        )

    def _run(self):
        self._ensure_graph()
        if self._graph is not None:
            try:
                wp.capture_launch(self._graph)
                return
            except Exception as e:
                print(f"graph replay failed: {e}")
                self._graph = None
        self._pipeline()

    def _ensure_graph(self):
        if self._graph is not None or self._graph_tried or not self._is_cuda:
            return
        self._graph_tried = True
        try:
            # warm up first so module compile doesn't happen inside capture
            # (saw ~2ms/step -> ~0.6ms on the Orin Nano with graphs on)
            self._pipeline()
            wp.synchronize_device(self.device)
            with wp.ScopedCapture(device=self.device) as cap:
                self._pipeline()
            self._graph = cap.graph
        except Exception as e:
            print(f"graph capture failed: {e}")
            self._graph = None

    def _integrate(self):
        wp.launch(
            integrate_scan_kernel,
            dim=self._n_beams,
            inputs=[
                self._ranges,
                self._angle_min,
                self._angle_inc,
                self._inv_n,
                self._control,
                self._pose_device,
                self.logodds,
                self.likelihood,
            ],
        )
        self._keyframe_pose = self.pose.copy()
        self._integrations += 1
