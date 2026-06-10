import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

GRID = wp.constant(2048)
RES = wp.constant(0.05)
INV_RES = wp.constant(20.0)
ORIGIN = wp.constant(-51.2)
L_OCC = wp.constant(0.85)
L_FREE = wp.constant(-0.4)
L_MIN = wp.constant(-5.0)
L_MAX = wp.constant(5.0)
L_SPLAT = wp.constant(0.5)
RMIN = wp.constant(0.05)
RMAX = wp.constant(30.0)
STRIDE = wp.constant(8)
LIKE_SIG_INV2 = wp.constant(0.5)
OCC_GATE = wp.constant(0.5)
FREE_GATE = wp.constant(-0.25)
BACK_DIST = wp.constant(0.30)
GN_MAX_STEP = wp.constant(0.05)
GN_MAX_ROT = wp.constant(0.05235988)


@wp.func
def beam_local(f: float, mx: float, my: float, mt: float, ex: float, ey: float):
    a = f * mt
    return f * mx + ex - a * ey, f * my + a * ex + ey


@wp.func
def to_grid(wx: float, wy: float):
    return (wx - ORIGIN) * INV_RES, (wy - ORIGIN) * INV_RES


@wp.func
def nearest(gx: float, gy: float):
    ix = wp.clamp(int(wp.floor(gx + 0.5)), 0, GRID - 1)
    iy = wp.clamp(int(wp.floor(gy + 0.5)), 0, GRID - 1)
    return ix, iy


@wp.func
def front_face(occ: wp.array2d(dtype=float), x: float, y: float, ox: float, oy: float):
    gx, gy = to_grid(x + ox, y + oy)
    ix, iy = nearest(gx, gy)
    dist = wp.sqrt(ox * ox + oy * oy)
    back = 1.0 - wp.min(BACK_DIST, 0.5 * dist) / dist
    bx, by = to_grid(x + ox * back, y + oy * back)
    jx, jy = nearest(bx, by)
    return occ[iy, ix] > OCC_GATE and occ[jy, jx] < FREE_GATE


@wp.func
def sample(m: wp.array2d(dtype=float), x: float, y: float):
    fx = wp.floor(x)
    fy = wp.floor(y)
    tx = x - fx
    ty = y - fy
    ix = wp.clamp(int(fx), 0, GRID - 2)
    iy = wp.clamp(int(fy), 0, GRID - 2)
    v00 = m[iy, ix]
    v10 = m[iy, ix + 1]
    v01 = m[iy + 1, ix]
    v11 = m[iy + 1, ix + 1]
    return wp.lerp(wp.lerp(v00, v10, tx), wp.lerp(v01, v11, tx), ty)


@wp.func
def sample_grad(m: wp.array2d(dtype=float), x: float, y: float):
    fx = wp.floor(x)
    fy = wp.floor(y)
    tx = x - fx
    ty = y - fy
    ix = wp.clamp(int(fx), 0, GRID - 2)
    iy = wp.clamp(int(fy), 0, GRID - 2)
    v00 = m[iy, ix]
    v10 = m[iy, ix + 1]
    v01 = m[iy + 1, ix]
    v11 = m[iy + 1, ix + 1]
    val = wp.lerp(wp.lerp(v00, v10, tx), wp.lerp(v01, v11, tx), ty)
    du = wp.lerp(v10 - v00, v11 - v01, ty)
    dv = wp.lerp(v01 - v00, v11 - v10, tx)
    return val, du, dv


@wp.kernel
def search_k(
    r: wp.array(dtype=float),
    ct: wp.array(dtype=float),
    st: wp.array(dtype=float),
    like: wp.array2d(dtype=float),
    occ: wp.array2d(dtype=float),
    ctrl: wp.array(dtype=float),
    pen: wp.array(dtype=float),
    step_xy: float,
    step_th: float,
    nx: int,
    nt: int,
    nm: int,
    inv_n: float,
    out: wp.array(dtype=float),
):
    i, j, k = wp.tid()
    h = float(nx - 1) * 0.5
    x = ctrl[0] + (float(i) - h) * step_xy
    y = ctrl[1] + (float(j) - h) * step_xy
    t = ctrl[2] + (float(k) - float(nt - 1) * 0.5) * step_th
    mx = ctrl[3]
    my = ctrl[4]
    mt = ctrl[5]
    c = wp.cos(t)
    s = wp.sin(t)
    total = float(0.0)
    for b in range(nm):
        v = r[b * STRIDE]
        if v >= RMIN and v < RMAX:
            f = float(b * STRIDE) * inv_n - 0.5
            lx, ly = beam_local(f, mx, my, mt, v * ct[b], v * st[b])
            ox = lx * c - ly * s
            oy = lx * s + ly * c
            if front_face(occ, x, y, ox, oy):
                gx, gy = to_grid(x + ox, y + oy)
                total += wp.max(sample(like, gx, gy), 0.0)
    idx = (i * nx + j) * nt + k
    out[idx] = total * pen[idx]


@wp.kernel
def argmax_k(
    score: wp.array(dtype=float),
    off_xy: wp.array(dtype=float),
    off_th: wp.array(dtype=float),
    ctrl: wp.array(dtype=float),
    nx: int,
    nt: int,
    pose: wp.array(dtype=float),
):
    best = float(-1.0)
    bi = int(0)
    for idx in range(nx * nx * nt):
        v = score[idx]
        if v > best:
            best = v
            bi = idx
    k = bi % nt
    j = (bi // nt) % nx
    i = bi // (nt * nx)
    pose[0] = ctrl[0] + off_xy[i]
    pose[1] = ctrl[1] + off_xy[j]
    pose[2] = ctrl[2] + off_th[k]


@wp.kernel
def gn_k(
    r: wp.array(dtype=float),
    like: wp.array2d(dtype=float),
    occ: wp.array2d(dtype=float),
    a_min: float,
    a_inc: float,
    inv_n: float,
    ctrl: wp.array(dtype=float),
    pose: wp.array(dtype=float),
    acc: wp.array(dtype=float),
):
    i = wp.tid()
    v = r[i]
    if v < RMIN or v >= RMAX:
        return
    ang = a_min + a_inc * float(i)
    f = float(i) * inv_n - 0.5
    lx, ly = beam_local(f, ctrl[3], ctrl[4], ctrl[5], v * wp.cos(ang), v * wp.sin(ang))
    c = wp.cos(pose[2])
    s = wp.sin(pose[2])
    ox = lx * c - ly * s
    oy = lx * s + ly * c
    if not front_face(occ, pose[0], pose[1], ox, oy):
        return
    gx, gy = to_grid(pose[0] + ox, pose[1] + oy)
    l, du, dv = sample_grad(like, gx, gy)
    res = 1.0 - wp.max(l, 0.0)
    sc = -float(INV_RES)
    j0 = sc * du
    j1 = sc * dv
    j2 = sc * (du * (-oy) + dv * ox)
    wp.atomic_add(acc, 0, j0 * j0)
    wp.atomic_add(acc, 1, j0 * j1)
    wp.atomic_add(acc, 2, j0 * j2)
    wp.atomic_add(acc, 3, j1 * j1)
    wp.atomic_add(acc, 4, j1 * j2)
    wp.atomic_add(acc, 5, j2 * j2)
    wp.atomic_add(acc, 6, j0 * res)
    wp.atomic_add(acc, 7, j1 * res)
    wp.atomic_add(acc, 8, j2 * res)
    wp.atomic_add(acc, 9, 1.0)


@wp.kernel
def solve_k(
    acc: wp.array(dtype=float),
    ctrl: wp.array(dtype=float),
    wt: float,
    wr: float,
    pose: wp.array(dtype=float),
):
    nv = acc[9]
    if nv < 3.0:
        return
    inv = 1.0 / nv
    A = wp.mat33(
        acc[0] * inv + wt,
        acc[1] * inv,
        acc[2] * inv,
        acc[1] * inv,
        acc[3] * inv + wt,
        acc[4] * inv,
        acc[2] * inv,
        acc[4] * inv,
        acc[5] * inv + wr,
    )
    dt = pose[2] - ctrl[2]
    dt = wp.atan2(wp.sin(dt), wp.cos(dt))
    b = wp.vec3(
        acc[6] * inv + wt * (pose[0] - ctrl[0]),
        acc[7] * inv + wt * (pose[1] - ctrl[1]),
        acc[8] * inv + wr * dt,
    )
    d = -(wp.inverse(A) * b)
    dn = wp.sqrt(d[0] * d[0] + d[1] * d[1])
    if dn > GN_MAX_STEP:
        k = GN_MAX_STEP / dn
        d = wp.vec3(d[0] * k, d[1] * k, d[2])
    da = wp.abs(d[2])
    if da > GN_MAX_ROT:
        d = wp.vec3(d[0], d[1], d[2] * GN_MAX_ROT / da)
    pose[0] = pose[0] + d[0]
    pose[1] = pose[1] + d[1]
    pose[2] = pose[2] + d[2]


@wp.kernel
def integrate_k(
    r: wp.array(dtype=float),
    a_min: float,
    a_inc: float,
    inv_n: float,
    ctrl: wp.array(dtype=float),
    pose: wp.array(dtype=float),
    logodds: wp.array2d(dtype=float),
    like: wp.array2d(dtype=float),
):
    i = wp.tid()
    v = r[i]
    if v < RMIN or v >= RMAX:
        return
    ang = a_min + a_inc * float(i)
    f = float(i) * inv_n - 0.5
    sx = f * ctrl[3]
    sy = f * ctrl[4]
    lx, ly = beam_local(f, ctrl[3], ctrl[4], ctrl[5], v * wp.cos(ang), v * wp.sin(ang))
    c = wp.cos(pose[2])
    s = wp.sin(pose[2])
    x0, y0 = to_grid(pose[0] + sx * c - sy * s, pose[1] + sx * s + sy * c)
    x1, y1 = to_grid(pose[0] + lx * c - ly * s, pose[1] + lx * s + ly * c)
    dx = x1 - x0
    dy = y1 - y0
    n = int(wp.max(wp.abs(dx), wp.abs(dy))) + 1
    ux = dx / float(n)
    uy = dy / float(n)
    gx = x0
    gy = y0
    for _ in range(n + 1):
        ix = int(wp.floor(gx))
        iy = int(wp.floor(gy))
        if 0 <= ix and ix < GRID and 0 <= iy and iy < GRID:
            wp.atomic_add(logodds, iy, ix, L_FREE)
            wp.atomic_max(logodds, iy, ix, L_MIN)
        gx += ux
        gy += uy
    hx = int(wp.floor(x1))
    hy = int(wp.floor(y1))
    hit = L_OCC - L_FREE
    splat = hit * L_SPLAT
    for dj in range(-1, 2):
        for di in range(-1, 2):
            ix = hx + di
            iy = hy + dj
            if 0 <= ix and ix < GRID and 0 <= iy and iy < GRID:
                wp.atomic_add(logodds, iy, ix, hit if (di == 0 and dj == 0) else splat)
                wp.atomic_min(logodds, iy, ix, L_MAX)
    for dj in range(-2, 3):
        for di in range(-2, 3):
            ix = hx + di
            iy = hy + dj
            if 0 <= ix and ix < GRID and 0 <= iy and iy < GRID:
                du = float(ix) - x1
                dv = float(iy) - y1
                wp.atomic_max(
                    like, iy, ix, wp.exp(-(du * du + dv * dv) * LIKE_SIG_INV2)
                )


@wp.kernel
def occ_k(logodds: wp.array2d(dtype=float), out: wp.array2d(dtype=wp.int8)):
    iy, ix = wp.tid()
    l = logodds[iy, ix]
    o = int(-1)
    if wp.abs(l) > 0.1:
        p = 1.0 / (1.0 + wp.exp(-wp.clamp(l, -10.0, 10.0)))
        o = int(p * 100.0)
    out[iy, ix] = wp.int8(o)


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class Bridge:
    GN_ITERS = 10
    W_TRANS, W_ROT = 0.5, 2.0
    KF_D2, KF_DTH = 0.20**2, np.radians(10.0)
    MIN_MOVE_XY, MIN_MOVE_TH = 0.02, np.radians(0.5)
    COARSE_HX, COARSE_HTH = 0.30, np.radians(15.0)
    COARSE_SX, COARSE_STH = 0.05, np.radians(1.0)
    CS_WT, CS_WR = 2.0, 2.0

    def __init__(self, device=None):
        wp.init()
        try:
            has_cuda = wp.get_cuda_device_count() > 0
        except Exception:
            has_cuda = False
        self.device = device or ("cuda" if has_cuda else "cpu")
        wp.set_device(self.device)
        self._is_cuda = str(self.device).startswith("cuda")

        self.logodds = wp.zeros((int(GRID), int(GRID)))
        self.like = wp.zeros((int(GRID), int(GRID)))
        self._occ = wp.zeros((int(GRID), int(GRID)), dtype=wp.int8)
        self._acc = wp.zeros(10, dtype=float)
        self._ctrl = wp.zeros(6, dtype=float)
        self._pose_d = wp.zeros(3, dtype=float)

        self._nx = 2 * int(self.COARSE_HX / self.COARSE_SX) + 1
        self._nt = 2 * int(self.COARSE_HTH / self.COARSE_STH) + 1
        self._score = wp.zeros(self._nx * self._nx * self._nt, dtype=float)
        off_xy = (np.arange(self._nx) - (self._nx - 1) * 0.5) * self.COARSE_SX
        off_th = (np.arange(self._nt) - (self._nt - 1) * 0.5) * self.COARSE_STH
        gx, gy, gt = np.meshgrid(off_xy, off_xy, off_th, indexing="ij")
        pen = np.exp(-(gx**2 + gy**2) * self.CS_WT**2 - gt**2 * self.CS_WR**2)
        self._off_xy = wp.array(off_xy.astype(np.float32))
        self._off_th = wp.array(off_th.astype(np.float32))
        self._pen = wp.array(pen.ravel().astype(np.float32))

        self._n = 0
        self._nm = 0
        self._a_min = self._a_inc = self._inv_n = 0.0
        self._ct = self._st = self._r = None
        self._graph = None
        self._graph_tried = False
        self.pose = np.zeros(3, dtype=np.float32)
        self._kf = self.pose.copy()
        self._first = True
        self._integrations = 0

    @property
    def n_beams(self):
        return self._n

    @property
    def keyframe_pose(self):
        return self._kf.copy()

    @property
    def integrations(self):
        return self._integrations

    def configure(self, n_beams, angle_min, angle_increment):
        n = int(n_beams)
        if (
            n == self._n
            and abs(angle_min - self._a_min) < 1e-9
            and abs(angle_increment - self._a_inc) < 1e-12
        ):
            return False
        if n < STRIDE * 4:
            raise ValueError(f"scan too small: {n} beams")
        self._n, self._a_min, self._a_inc = n, float(angle_min), float(angle_increment)
        self._inv_n = 1.0 / float(n - 1)
        self._nm = n // int(STRIDE)
        ang = self._a_min + self._a_inc * int(STRIDE) * np.arange(
            self._nm, dtype=np.float32
        )
        self._ct = wp.array(np.cos(ang).astype(np.float32))
        self._st = wp.array(np.sin(ang).astype(np.float32))
        self._r = wp.zeros(n)
        self._graph = None
        self._graph_tried = False
        return True

    def step(self, ranges, odom, deskew_frac=0.0):
        r = np.ascontiguousarray(ranges, dtype=np.float32)
        odom = np.asarray(odom, dtype=np.float32)
        self._r.assign(r)

        if self._first:
            self._first = False
            self.pose = odom.copy()
            self._kf = self.pose.copy()
            self._ctrl.assign(np.array([*self.pose, 0.0, 0.0, 0.0], dtype=np.float32))
            self._pose_d.assign(self.pose)
            self._integrate()
            return self.pose

        seed = self.pose + odom
        moved = (
            np.linalg.norm(odom[:2]) > self.MIN_MOVE_XY
            or abs(odom[2]) > self.MIN_MOVE_TH
        )
        hit = bool(np.any((r >= float(RMIN)) & (r < float(RMAX))))

        fr = float(np.clip(deskew_frac, 0.0, 1.0))
        c, s = np.cos(self.pose[2]), np.sin(self.pose[2])
        mx = (odom[0] * c + odom[1] * s) * fr
        my = (-odom[0] * s + odom[1] * c) * fr
        self._ctrl.assign(np.array([*seed, mx, my, odom[2] * fr], dtype=np.float32))

        if hit and moved:
            self._ensure_graph()
            if self._graph is not None:
                wp.capture_launch(self._graph)
            else:
                self._pipeline()
            self.pose = self._pose_d.numpy().astype(np.float32).copy()
        else:
            self.pose = seed.copy()
            self._pose_d.assign(self.pose)

        self.pose[2] = _wrap(float(self.pose[2]))
        d = self.pose - self._kf
        d[2] = _wrap(float(d[2]))
        if d[0] * d[0] + d[1] * d[1] > self.KF_D2 or abs(d[2]) > self.KF_DTH:
            self._integrate()
        return self.pose

    def occupancy(self):
        if self._is_cuda:
            wp.launch(
                occ_k, dim=(int(GRID), int(GRID)), inputs=[self.logodds, self._occ]
            )
            return self._occ.numpy()
        lo = self.logodds.numpy()
        out = np.full(lo.shape, -1, dtype=np.int8)
        known = np.abs(lo) > 0.1
        if known.any():
            p = 1.0 / (1.0 + np.exp(-np.clip(lo[known], -10.0, 10.0)))
            out[known] = (p * 100.0).astype(np.int8)
        return out

    def _pipeline(self):
        wp.launch(
            search_k,
            dim=(self._nx, self._nx, self._nt),
            inputs=[
                self._r,
                self._ct,
                self._st,
                self.like,
                self.logodds,
                self._ctrl,
                self._pen,
                self.COARSE_SX,
                self.COARSE_STH,
                self._nx,
                self._nt,
                self._nm,
                self._inv_n,
                self._score,
            ],
        )
        wp.launch(
            argmax_k,
            dim=1,
            inputs=[
                self._score,
                self._off_xy,
                self._off_th,
                self._ctrl,
                self._nx,
                self._nt,
                self._pose_d,
            ],
        )
        for _ in range(self.GN_ITERS):
            self._acc.zero_()
            wp.launch(
                gn_k,
                dim=self._n,
                inputs=[
                    self._r,
                    self.like,
                    self.logodds,
                    self._a_min,
                    self._a_inc,
                    self._inv_n,
                    self._ctrl,
                    self._pose_d,
                    self._acc,
                ],
            )
            wp.launch(
                solve_k,
                dim=1,
                inputs=[
                    self._acc,
                    self._ctrl,
                    float(self.W_TRANS),
                    float(self.W_ROT),
                    self._pose_d,
                ],
            )

    def _ensure_graph(self):
        if self._graph is not None or self._graph_tried or not self._is_cuda:
            return
        self._graph_tried = True
        try:
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
            integrate_k,
            dim=self._n,
            inputs=[
                self._r,
                self._a_min,
                self._a_inc,
                self._inv_n,
                self._ctrl,
                self._pose_d,
                self.logodds,
                self.like,
            ],
        )
        self._kf = self.pose.copy()
        self._integrations += 1
