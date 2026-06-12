import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

GRID = wp.constant(2048)
RES = wp.constant(0.05)
INV_RES = wp.constant(20.0)
ORIGIN = wp.constant(-51.2)
L_OCC = wp.constant(0.85)
L_FREE = wp.constant(-0.4)
L_MIN = wp.constant(-2.5)
L_MAX = wp.constant(2.5)
RMIN = wp.constant(0.05)
RMAX = wp.constant(30.0)
STRIDE = wp.constant(8)
ITERS = wp.constant(10)
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
    ix = wp.clamp(int(fx), 0, GRID - 2)
    iy = wp.clamp(int(fy), 0, GRID - 2)
    v0 = wp.lerp(m[iy, ix], m[iy, ix + 1], x - fx)
    v1 = wp.lerp(m[iy + 1, ix], m[iy + 1, ix + 1], x - fx)
    return wp.lerp(v0, v1, y - fy)


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


@wp.func
def beam_world(
    r: wp.array(dtype=float),
    b: int,
    a_min: float,
    a_inc: float,
    inv_n: float,
    ctrl: wp.array(dtype=float),
    c: float,
    s: float,
):
    v = r[b]
    ang = a_min + a_inc * float(b)
    f = float(b) * inv_n - 0.5
    lx, ly = beam_local(f, ctrl[3], ctrl[4], ctrl[5], v * wp.cos(ang), v * wp.sin(ang))
    return lx * c - ly * s, lx * s + ly * c


@wp.kernel
def search_k(
    r: wp.array(dtype=float),
    like: wp.array2d(dtype=float),
    occ: wp.array2d(dtype=float),
    ctrl: wp.array(dtype=float),
    a_min: float,
    a_inc: float,
    inv_n: float,
    step_xy: float,
    step_th: float,
    wt: float,
    wr: float,
    nx: int,
    nt: int,
    nm: int,
    out: wp.array(dtype=float),
):
    i, j, k = wp.tid()
    h = float(nx - 1) * 0.5
    dx = (float(i) - h) * step_xy
    dy = (float(j) - h) * step_xy
    dt = (float(k) - float(nt - 1) * 0.5) * step_th
    x = ctrl[0] + dx
    y = ctrl[1] + dy
    c = wp.cos(ctrl[2] + dt)
    s = wp.sin(ctrl[2] + dt)
    total = float(0.0)
    for b in range(nm):
        if r[b * STRIDE] >= RMIN and r[b * STRIDE] < RMAX:
            ox, oy = beam_world(r, b * STRIDE, a_min, a_inc, inv_n, ctrl, c, s)
            if front_face(occ, x, y, ox, oy):
                gx, gy = to_grid(x + ox, y + oy)
                total += wp.max(sample(like, gx, gy), 0.0)
    pen = wp.exp(-(dx * dx + dy * dy) * wt * wt - dt * dt * wr * wr)
    out[(i * nx + j) * nt + k] = total * pen


@wp.kernel
def argmax_k(
    score: wp.array(dtype=float),
    ctrl: wp.array(dtype=float),
    step_xy: float,
    step_th: float,
    nx: int,
    nt: int,
    pose: wp.array(dtype=float),
):
    tid = wp.tid()
    best = float(-1.0)
    bi = int(0)
    i = tid
    while i < nx * nx * nt:
        if score[i] > best:
            best = score[i]
            bi = i
        i += wp.block_dim()
    lane = wp.tile_argmax(wp.tile(best))[0]
    cand = wp.tile(bi)
    win = cand[lane]
    if tid == 0:
        h = float(nx - 1) * 0.5
        pose[0] = ctrl[0] + (float(win // (nt * nx)) - h) * step_xy
        pose[1] = ctrl[1] + (float((win // nt) % nx) - h) * step_xy
        pose[2] = ctrl[2] + (float(win % nt) - float(nt - 1) * 0.5) * step_th


@wp.kernel
def refine_k(
    r: wp.array(dtype=float),
    like: wp.array2d(dtype=float),
    occ: wp.array2d(dtype=float),
    n: int,
    a_min: float,
    a_inc: float,
    inv_n: float,
    wt: float,
    wr: float,
    ctrl: wp.array(dtype=float),
    pose: wp.array(dtype=float),
):
    tid = wp.tid()
    px = pose[0]
    py = pose[1]
    pt = pose[2]
    fit = float(0.0)
    cov = float(0.0)
    for _ in range(ITERS):
        H = wp.mat33()
        g = wp.vec3()
        st = wp.vec3()
        c = wp.cos(pt)
        s = wp.sin(pt)
        b = tid
        while b < n:
            if r[b] >= RMIN and r[b] < RMAX:
                st += wp.vec3(0.0, 0.0, 1.0)
                ox, oy = beam_world(r, b, a_min, a_inc, inv_n, ctrl, c, s)
                if front_face(occ, px, py, ox, oy):
                    gx, gy = to_grid(px + ox, py + oy)
                    l, du, dv = sample_grad(like, gx, gy)
                    j = wp.vec3(du, dv, du * (-oy) + dv * ox) * -float(INV_RES)
                    H += wp.outer(j, j)
                    g += j * (1.0 - wp.max(l, 0.0))
                    st += wp.vec3(1.0, wp.max(l, 0.0), 0.0)
            b += wp.block_dim()
        Ht = wp.tile_sum(wp.tile(H, preserve_type=True))[0]
        gt = wp.tile_sum(wp.tile(g, preserve_type=True))[0]
        cnt = wp.tile_sum(wp.tile(st, preserve_type=True))[0]
        nv = cnt[0]
        fit = cnt[1] / wp.max(cnt[2], 1.0)
        cov = nv / wp.max(cnt[2], 1.0)
        if nv >= 3.0:
            dth = pt - ctrl[2]
            dth = wp.atan2(wp.sin(dth), wp.cos(dth))
            A = Ht / nv + wp.diag(wp.vec3(wt, wt, wr))
            rhs = gt / nv + wp.vec3(wt * (px - ctrl[0]), wt * (py - ctrl[1]), wr * dth)
            d = -(wp.inverse(A) * rhs)
            dn = wp.sqrt(d[0] * d[0] + d[1] * d[1])
            if dn > GN_MAX_STEP:
                k = GN_MAX_STEP / dn
                d = wp.vec3(d[0] * k, d[1] * k, d[2])
            if wp.abs(d[2]) > GN_MAX_ROT:
                d = wp.vec3(d[0], d[1], d[2] * GN_MAX_ROT / wp.abs(d[2]))
            px += d[0]
            py += d[1]
            pt += d[2]
    if tid == 0:
        pose[0] = px
        pose[1] = py
        pose[2] = pt
        pose[3] = fit
        pose[4] = cov


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
    if r[i] < RMIN or r[i] >= RMAX:
        return
    c = wp.cos(pose[2])
    s = wp.sin(pose[2])
    f = float(i) * inv_n - 0.5
    x0, y0 = to_grid(
        pose[0] + (f * ctrl[3]) * c - (f * ctrl[4]) * s,
        pose[1] + (f * ctrl[3]) * s + (f * ctrl[4]) * c,
    )
    ox, oy = beam_world(r, i, a_min, a_inc, inv_n, ctrl, c, s)
    x1, y1 = to_grid(pose[0] + ox, pose[1] + oy)
    n = int(wp.max(wp.abs(x1 - x0), wp.abs(y1 - y0))) + 1
    ux = (x1 - x0) / float(n)
    uy = (y1 - y0) / float(n)
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
    if 0 <= hx and hx < GRID and 0 <= hy and hy < GRID:
        wp.atomic_add(logodds, hy, hx, L_OCC - L_FREE)
        wp.atomic_min(logodds, hy, hx, L_MAX)
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
    W_TRANS, W_ROT = 0.5, 2.0
    KF_D2, KF_DTH = 0.20**2, np.radians(10.0)
    MIN_MOVE_XY, MIN_MOVE_TH = 0.02, np.radians(0.5)
    COARSE_HX, COARSE_HTH = 0.30, np.radians(15.0)
    COARSE_SX, COARSE_STH = 0.05, np.radians(1.0)
    CS_WT, CS_WR = 2.0, 2.0
    FIT_MAP = 0.50
    COV_NEW = 0.35
    FIT_RETRY = 0.40
    COV_MIN = 0.15
    RETRY_SCALE = 3.0
    BLOCK = 256

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
        self._ctrl = wp.zeros(6, dtype=float)
        self._pose_d = wp.zeros(5, dtype=float)
        self._nx = 2 * int(self.COARSE_HX / self.COARSE_SX) + 1
        self._nt = 2 * int(self.COARSE_HTH / self.COARSE_STH) + 1
        self._score = wp.zeros(self._nx * self._nx * self._nt, dtype=float)

        self._n = self._nm = 0
        self._a_min = self._a_inc = self._inv_n = 0.0
        self._r = None
        self._graph = None
        self._graph_tried = False
        self.pose = np.zeros(3, dtype=np.float32)
        self._kf = self.pose.copy()
        self._fit = self._cov = 1.0
        self._first = True
        self._integrations = 0

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
            self._pose_d.assign(np.array([*self.pose, 1.0, 1.0], dtype=np.float32))
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
            self._run()
            p = self._pose_d.numpy()
            if p[3] < self.FIT_RETRY and p[4] > self.COV_MIN:
                p0 = p.copy()
                self._search(self.RETRY_SCALE)
                wp.copy(self._ctrl, self._pose_d, count=3)
                self._run()
                p = self._pose_d.numpy()
                if p[3] <= p0[3]:
                    p = p0
                    self._pose_d.assign(p0)
            self.pose = p[:3].astype(np.float32).copy()
            self._fit, self._cov = float(p[3]), float(p[4])
        else:
            self.pose = seed.copy()
            self._pose_d.assign(
                np.array([*self.pose, self._fit, self._cov], dtype=np.float32)
            )

        self.pose[2] = _wrap(float(self.pose[2]))
        d = self.pose - self._kf
        d[2] = _wrap(float(d[2]))
        if (d[0] * d[0] + d[1] * d[1] > self.KF_D2 or abs(d[2]) > self.KF_DTH) and (
            self._fit >= self.FIT_MAP or self._cov < self.COV_NEW
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

    def _search(self, scale):
        wp.launch(
            search_k,
            dim=(self._nx, self._nx, self._nt),
            inputs=[
                self._r,
                self.like,
                self.logodds,
                self._ctrl,
                self._a_min,
                self._a_inc,
                self._inv_n,
                self.COARSE_SX * scale,
                self.COARSE_STH * scale,
                self.CS_WT / scale,
                self.CS_WR / scale,
                self._nx,
                self._nt,
                self._nm,
                self._score,
            ],
        )
        wp.launch_tiled(
            argmax_k,
            dim=[1],
            block_dim=self.BLOCK,
            inputs=[
                self._score,
                self._ctrl,
                self.COARSE_SX * scale,
                self.COARSE_STH * scale,
                self._nx,
                self._nt,
                self._pose_d,
            ],
        )

    def _pipeline(self):
        self._search(1.0)
        wp.launch_tiled(
            refine_k,
            dim=[1],
            block_dim=self.BLOCK,
            inputs=[
                self._r,
                self.like,
                self.logodds,
                self._n,
                self._a_min,
                self._a_inc,
                self._inv_n,
                float(self.W_TRANS),
                float(self.W_ROT),
                self._ctrl,
                self._pose_d,
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
