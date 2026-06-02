import numpy as np
import warp as wp

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


@wp.func
def bilin(m: wp.array2d[float], x: float, y: float):
    ix = int(wp.floor(x))
    iy = int(wp.floor(y))
    tx = x - float(ix)
    ty = y - float(iy)
    ix = wp.clamp(ix, 0, GRID - 2)
    iy = wp.clamp(iy, 0, GRID - 2)
    v00 = m[iy, ix]
    v10 = m[iy, ix + 1]
    v01 = m[iy + 1, ix]
    v11 = m[iy + 1, ix + 1]
    return wp.lerp(wp.lerp(v00, v10, tx), wp.lerp(v01, v11, tx), ty), v00, v10, v01, v11


@wp.kernel
def search_k(
    r: wp.array[float],
    ct: wp.array[float],
    st: wp.array[float],
    like: wp.array2d[float],
    sx: float,
    sy: float,
    sth: float,
    ss: float,
    ts: float,
    nx: int,
    nt: int,
    nm: int,
    out: wp.array[float],
):
    i, j, k = wp.tid()
    h = float(nx - 1) * 0.5
    x = sx + (float(i) - h) * ss
    y = sy + (float(j) - h) * ss
    t = sth + (float(k) - float(nt - 1) * 0.5) * ts
    c = wp.cos(t)
    s = wp.sin(t)
    total = float(0.0)
    for b in range(nm):
        v = r[b * STRIDE]
        if RMIN <= v < RMAX:
            ca = ct[b] * c - st[b] * s
            sa = st[b] * c + ct[b] * s
            l, _, _, _, _ = bilin(
                like, (x + v * ca - ORIGIN) * INV_RES, (y + v * sa - ORIGIN) * INV_RES
            )
            total += wp.max(l, 0.0)
    out[(i * nx + j) * nt + k] = total


@wp.kernel
def gn_k(
    r: wp.array[float],
    like: wp.array2d[float],
    a_min: float,
    a_inc: float,
    x: float,
    y: float,
    theta: float,
    acc: wp.array[float],
):
    i = wp.tid()
    v = r[i]
    if not (RMIN <= v < RMAX):
        return
    a = a_min + a_inc * float(i) + theta
    ca = wp.cos(a)
    sa = wp.sin(a)
    gx = (x + v * ca - ORIGIN) * INV_RES
    gy = (y + v * sa - ORIGIN) * INV_RES
    l, v00, v10, v01, v11 = bilin(like, gx, gy)
    tx = gx - wp.floor(gx)
    ty = gy - wp.floor(gy)
    ddu = wp.lerp(v10 - v00, v11 - v01, ty)
    ddv = wp.lerp(v01 - v00, v11 - v10, tx)
    res = 1.0 - wp.max(l, 0.0) / float(L_MAX)
    s = -float(INV_RES) / float(L_MAX)
    j0 = s * ddu
    j1 = s * ddv
    j2 = s * (ddu * (-v * sa) + ddv * (v * ca))
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
def integrate_k(
    r: wp.array[float],
    a_min: float,
    a_inc: float,
    px: float,
    py: float,
    pt: float,
    g: wp.array2d[float],
):
    i = wp.tid()
    v = r[i]
    if not (RMIN <= v < RMAX):
        return
    a = a_min + a_inc * float(i) + pt
    ca = wp.cos(a)
    sa = wp.sin(a)
    x0 = (px - ORIGIN) * INV_RES
    y0 = (py - ORIGIN) * INV_RES
    x1 = (px + v * ca - ORIGIN) * INV_RES
    y1 = (py + v * sa - ORIGIN) * INV_RES
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
        if 0 <= ix < GRID and 0 <= iy < GRID:
            wp.atomic_add(g, iy, ix, L_FREE)
            wp.atomic_max(g, iy, ix, L_MIN)
        gx += ux
        gy += uy
    hx = int(wp.floor(x1))
    hy = int(wp.floor(y1))
    d = L_OCC - L_FREE
    sp = d * L_SPLAT
    for dyk in range(-1, 2):
        for dxk in range(-1, 2):
            nxk = hx + dxk
            nyk = hy + dyk
            if 0 <= nxk < GRID and 0 <= nyk < GRID:
                wp.atomic_add(g, nyk, nxk, d if (dxk == 0 and dyk == 0) else sp)
                wp.atomic_min(g, nyk, nxk, L_MAX)


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class Bridge:
    GN_ITERS = 10
    W_TRANS, W_ROT = 500.0, 100.0
    GN_MAX_STEP, GN_MAX_ROT = 0.05, np.radians(3.0)
    KF_D2, KF_DTH = 0.20**2, np.radians(10.0)
    MIN_MOVE_XY, MIN_MOVE_TH = 0.02, np.radians(0.5)
    COARSE_HX, COARSE_HTH, COARSE_SX, COARSE_STH = (
        0.30,
        np.radians(15.0),
        0.05,
        np.radians(1.0),
    )
    CS_WT, CS_WR = 2.0, 2.0

    def __init__(self, device=None):
        wp.init()
        try:
            has_cuda = wp.get_cuda_device_count() > 0
        except Exception:
            has_cuda = False
        self.device = device or ("cuda" if has_cuda else "cpu")
        wp.set_device(self.device)

        self.logodds = wp.zeros((int(GRID), int(GRID)))
        self._nx = 2 * int(self.COARSE_HX / self.COARSE_SX) + 1
        self._nt = 2 * int(self.COARSE_HTH / self.COARSE_STH) + 1
        self._sc = wp.zeros(self._nx * self._nx * self._nt, dtype=float)
        ox = (np.arange(self._nx) - (self._nx - 1) * 0.5) * self.COARSE_SX
        ot = (np.arange(self._nt) - (self._nt - 1) * 0.5) * self.COARSE_STH
        gx, gy, gt = np.meshgrid(ox, ox, ot, indexing="ij")
        self._coarse_ox = ox.astype(np.float32)
        self._coarse_ot = ot.astype(np.float32)
        self._coarse_pen = np.exp(
            -(gx**2 + gy**2) * self.CS_WT**2 - gt**2 * self.CS_WR**2
        ).astype(np.float32)

        self._n = 0
        self._a_min = self._a_inc = 0.0
        self._nm = 0
        self._ct = self._st = self._r = None
        self._acc = wp.zeros(10, dtype=float)
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
        self._nm = n // int(STRIDE)
        ang = self._a_min + self._a_inc * int(STRIDE) * np.arange(
            self._nm, dtype=np.float32
        )
        self._ct = wp.array(np.cos(ang).astype(np.float32))
        self._st = wp.array(np.sin(ang).astype(np.float32))
        self._r = wp.zeros(n)
        return True

    def step(self, ranges, odom):
        r = np.ascontiguousarray(ranges, dtype=np.float32)
        self._r.assign(r)
        if self._first:
            self.pose = odom.astype(np.float32)
            self._kf = self.pose.copy()
            self._first = False
            self._integrate()
            return self.pose
        seed = self.pose + odom.astype(np.float32)
        moved = (
            np.linalg.norm(odom[:2]) > self.MIN_MOVE_XY
            or abs(odom[2]) > self.MIN_MOVE_TH
        )
        hit = np.any((r >= float(RMIN)) & (r < float(RMAX)))
        self.pose = seed if (not hit or not moved) else self._match(seed)
        self.pose[2] = _wrap(self.pose[2])
        d = self.pose - self._kf
        d[2] = _wrap(d[2])
        if d[0] * d[0] + d[1] * d[1] > self.KF_D2 or abs(d[2]) > self.KF_DTH:
            self._integrate()
        return self.pose

    def _match(self, seed):
        p = seed.astype(np.float32).copy()
        nx, nt = self._nx, self._nt
        wp.launch(
            search_k,
            dim=(nx, nx, nt),
            inputs=[
                self._r,
                self._ct,
                self._st,
                self.logodds,
                p[0],
                p[1],
                p[2],
                self.COARSE_SX,
                self.COARSE_STH,
                nx,
                nt,
                self._nm,
                self._sc,
            ],
        )
        sc = self._sc.numpy().reshape(nx, nx, nt)
        best = np.unravel_index(int(np.argmax(sc * self._coarse_pen)), sc.shape)
        p[0] = seed[0] + float(self._coarse_ox[best[0]])
        p[1] = seed[1] + float(self._coarse_ox[best[1]])
        p[2] = seed[2] + float(self._coarse_ot[best[2]])

        for _ in range(self.GN_ITERS):
            self._acc.zero_()
            wp.launch(
                gn_k,
                dim=self._n,
                inputs=[
                    self._r,
                    self.logodds,
                    self._a_min,
                    self._a_inc,
                    p[0],
                    p[1],
                    p[2],
                    self._acc,
                ],
            )
            a = self._acc.numpy()
            nv = int(a[9])
            if nv < 3:
                break
            inv = 1.0 / nv
            JtJ = np.array(
                [[a[0], a[1], a[2]], [a[1], a[3], a[4]], [a[2], a[4], a[5]]]
            ) * inv + np.diag([self.W_TRANS, self.W_TRANS, self.W_ROT])
            dt = _wrap(p[2] - seed[2])
            Jtr = np.array([a[6], a[7], a[8]]) * inv + np.array(
                [
                    self.W_TRANS * (p[0] - seed[0]),
                    self.W_TRANS * (p[1] - seed[1]),
                    self.W_ROT * dt,
                ]
            )
            try:
                delta = np.linalg.solve(JtJ, -Jtr)
            except np.linalg.LinAlgError:
                break
            dn = np.linalg.norm(delta[:2])
            da = abs(delta[2])
            if dn > self.GN_MAX_STEP:
                delta[:2] *= self.GN_MAX_STEP / dn
            if da > self.GN_MAX_ROT:
                delta[2] *= self.GN_MAX_ROT / da
            p += delta.astype(np.float32)
            if dn < 1e-4 and da < 1e-4:
                break
        return p

    def _integrate(self):
        wp.launch(
            integrate_k,
            dim=self._n,
            inputs=[
                self._r,
                self._a_min,
                self._a_inc,
                self.pose[0],
                self.pose[1],
                self.pose[2],
                self.logodds,
            ],
        )
        self._kf = self.pose.copy()
        self._integrations += 1
