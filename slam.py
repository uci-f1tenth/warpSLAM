import numpy as np
import warp as wp

GRID_W = wp.constant(2048)
GRID_H = wp.constant(2048)
RES = wp.constant(0.05)
INV_RES = wp.constant(20.0)
OX = wp.constant(-51.2)
OY = wp.constant(-51.2)
L_OCC = wp.constant(0.85)
L_FREE = wp.constant(-0.4)
L_MIN = wp.constant(-5.0)
L_MAX = wp.constant(5.0)
L_SPLAT = wp.constant(0.5)
RMIN = wp.constant(0.05)
RMAX = wp.constant(30.0)
STRIDE = wp.constant(8)
EPS = wp.constant(1e-8)


@wp.func
def _get(m: wp.array2d[float], x: int, y: int):
    return m[wp.clamp(y, 0, GRID_H - 1), wp.clamp(x, 0, GRID_W - 1)]


@wp.func
def _bilin(m: wp.array2d[float], x: float, y: float):
    ix = int(wp.floor(x))
    iy = int(wp.floor(y))
    tx = x - float(ix)
    ty = y - float(iy)
    a = wp.lerp(_get(m, ix, iy), _get(m, ix + 1, iy), tx)
    b = wp.lerp(_get(m, ix, iy + 1), _get(m, ix + 1, iy + 1), tx)
    return wp.lerp(a, b, ty)


@wp.kernel
def search_k(
    r: wp.array[float],
    ct: wp.array[float],
    st: wp.array[float],
    like: wp.array2d[float],
    sx: float,
    sy: float,
    stheta: float,
    ss: float,
    ts: float,
    nx: int,
    nt: int,
    nm: int,
    sc: wp.array[float],
):
    i, j, k = wp.tid()
    h = float(nx - 1) * 0.5
    x = sx + (float(i) - h) * ss
    y = sy + (float(j) - h) * ss
    t = stheta + (float(k) - float(nt - 1) * 0.5) * ts
    c = wp.cos(t)
    s = wp.sin(t)
    total = float(0.0)
    for b in range(nm):
        v = r[b * STRIDE]
        if not (RMIN <= v < RMAX):
            continue
        ca = ct[b] * c - st[b] * s
        sa = st[b] * c + ct[b] * s
        gx = (x + v * ca - OX) * INV_RES
        gy = (y + v * sa - OY) * INV_RES
        total += wp.max(_bilin(like, gx, gy), 0.0)
    sc[(i * nx + j) * nt + k] = total


@wp.kernel
def gn_reduce_k(
    r: wp.array[float],
    like: wp.array2d[float],
    a_min: float,
    a_inc: float,
    x: float,
    y: float,
    theta: float,
    w: float,
    acc: wp.array[float],
):
    i = wp.tid()
    v = r[i]
    if not (RMIN <= v < RMAX):
        return
    a = a_min + a_inc * float(i) + theta
    ca = wp.cos(a)
    sa = wp.sin(a)
    gx = (x + v * ca - OX) * INV_RES
    gy = (y + v * sa - OY) * INV_RES
    ix = int(wp.floor(gx))
    iy = int(wp.floor(gy))
    tx = gx - float(ix)
    ty = gy - float(iy)
    v00 = _get(like, ix, iy)
    v10 = _get(like, ix + 1, iy)
    v01 = _get(like, ix, iy + 1)
    v11 = _get(like, ix + 1, iy + 1)
    l = wp.lerp(wp.lerp(v00, v10, tx), wp.lerp(v01, v11, tx), ty)
    l0 = wp.max(l, 0.0)
    ddu = wp.lerp(v10 - v00, v11 - v01, ty)
    ddv = wp.lerp(v01 - v00, v11 - v10, tx)
    cost = 1.0 - l0 / float(L_MAX)
    rw = wp.sqrt(wp.max(w, EPS))
    res = rw * cost
    s = -rw / float(L_MAX) * float(INV_RES)
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
def add_k(
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
    x0 = (px - OX) * INV_RES
    y0 = (py - OY) * INV_RES
    x1 = (px + v * ca - OX) * INV_RES
    y1 = (py + v * sa - OY) * INV_RES
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
        if 0 <= ix < GRID_W and 0 <= iy < GRID_H:
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
            if 0 <= nxk < GRID_W and 0 <= nyk < GRID_H:
                wp.atomic_add(g, nyk, nxk, d if (dxk == 0 and dyk == 0) else sp)
                wp.atomic_min(g, nyk, nxk, L_MAX)


class Bridge:
    def __init__(self, config=None):
        wp.init()
        try:
            has_cuda = wp.get_cuda_device_count() > 0
        except Exception:
            has_cuda = False
        opts = dict(config or {})
        self.device = opts.get("device") or ("cuda" if has_cuda else "cpu")
        wp.set_device(self.device)

        self._coarse = opts.get("coarse", (0.30, 15.0, 0.05, 1.0))
        self._cs_wt = opts.get("cs_wt", 2.0)
        self._cs_wr = opts.get("cs_wr", 2.0)
        self._gn_iters = opts.get("gn_iters", 10)
        self._w_occ = opts.get("w_occ", 1.0)
        self._w_trans = opts.get("w_trans", 500.0)
        self._w_rot = opts.get("w_rot", 100.0)
        self._gn_max_step = opts.get("gn_max_step", 0.05)
        self._gn_max_rot = opts.get("gn_max_rot", np.radians(3.0))
        self._kf_d2 = opts.get("kf_dist", 0.20) ** 2
        self._kf_dt = np.radians(opts.get("kf_angle", 10.0))
        self._min_move = opts.get("min_move", 0.02)

        self.logodds = wp.zeros((int(GRID_H), int(GRID_W)))
        hx, htd, sx, std = self._coarse
        self._nx = 2 * int(hx / sx) + 1
        self._nt = 2 * int(np.radians(htd) / np.radians(std)) + 1
        self._sx = sx
        self._st_step = float(np.radians(std))
        self._sc = wp.zeros(self._nx * self._nx * self._nt, dtype=float)
        ox = (np.arange(self._nx) - (self._nx - 1) * 0.5) * self._sx
        ot = (np.arange(self._nt) - (self._nt - 1) * 0.5) * self._st_step
        gx, gy, gt = np.meshgrid(ox, ox, ot, indexing="ij")
        self._coarse_ox = ox.astype(np.float32)
        self._coarse_ot = ot.astype(np.float32)
        self._coarse_pen = np.exp(
            -(gx**2 + gy**2) * self._cs_wt**2 - gt**2 * self._cs_wr**2
        ).astype(np.float32)

        self._n = 0
        self._a_min = 0.0
        self._a_inc = 0.0
        self._nm = 0
        self._ct = self._st = self._r = None
        self._gn_acc = wp.zeros(10, dtype=float)
        self.pose = np.zeros(3, dtype=np.float32)
        self._kf = np.zeros(3, dtype=np.float32)
        self._first = True

    @property
    def n_beams(self):
        return self._n

    @property
    def keyframe_pose(self):
        return self._kf.copy()

    def configure(self, n_beams, angle_min, angle_increment):
        n_beams = int(n_beams)
        if (
            n_beams == self._n
            and abs(angle_min - self._a_min) < 1e-9
            and abs(angle_increment - self._a_inc) < 1e-12
        ):
            return False
        if n_beams < STRIDE * 4:
            raise ValueError(f"scan too small: {n_beams} beams")
        self._n = n_beams
        self._a_min = float(angle_min)
        self._a_inc = float(angle_increment)
        self._nm = n_beams // int(STRIDE)
        ang = self._a_min + self._a_inc * int(STRIDE) * np.arange(
            self._nm, dtype=np.float32
        )
        self._ct = wp.array(np.cos(ang).astype(np.float32))
        self._st = wp.array(np.sin(ang).astype(np.float32))
        self._r = wp.zeros(self._n)
        return True

    def step(self, rn, odom):
        rn = np.ascontiguousarray(rn, dtype=np.float32)
        self._r.assign(rn)
        if self._first:
            self.pose = odom.astype(np.float32)
            self._kf = self.pose.copy()
            self._first = False
            self._integrate()
            return self.pose
        seed = self.pose + odom.astype(np.float32)
        moved = np.linalg.norm(odom[:2]) > self._min_move or abs(odom[2]) > np.radians(
            0.5
        )
        hit = np.any((rn >= float(RMIN)) & (rn < float(RMAX)))
        self.pose = seed if (not hit or not moved) else self._match(seed)
        self.pose[2] = (self.pose[2] + np.pi) % (2.0 * np.pi) - np.pi
        d = self.pose - self._kf
        d[2] = (d[2] + np.pi) % (2.0 * np.pi) - np.pi
        if d[0] * d[0] + d[1] * d[1] > self._kf_d2 or abs(d[2]) > self._kf_dt:
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
                self._sx,
                self._st_step,
                nx,
                nt,
                self._nm,
                self._sc,
            ],
        )
        sc = self._sc.numpy()[: nx * nx * nt].reshape(nx, nx, nt)
        best = np.unravel_index(int(np.argmax(sc * self._coarse_pen)), sc.shape)
        p[0] = seed[0] + float(self._coarse_ox[best[0]])
        p[1] = seed[1] + float(self._coarse_ox[best[1]])
        p[2] = seed[2] + float(self._coarse_ot[best[2]])
        n = self._n
        for _ in range(self._gn_iters):
            self._gn_acc.zero_()
            wp.launch(
                gn_reduce_k,
                dim=n,
                inputs=[
                    self._r,
                    self.logodds,
                    self._a_min,
                    self._a_inc,
                    p[0],
                    p[1],
                    p[2],
                    self._w_occ,
                    self._gn_acc,
                ],
            )
            acc = self._gn_acc.numpy()
            nv = int(acc[9])
            if nv < 3:
                break
            inv_nv = 1.0 / float(nv)
            JtJ = np.array(
                [
                    [acc[0], acc[1], acc[2]],
                    [acc[1], acc[3], acc[4]],
                    [acc[2], acc[4], acc[5]],
                ]
            ) * inv_nv + np.diag([self._w_trans, self._w_trans, self._w_rot])
            dt = (p[2] - seed[2] + np.pi) % (2.0 * np.pi) - np.pi
            Jtr = np.array([acc[6], acc[7], acc[8]]) * inv_nv + np.array(
                [
                    self._w_trans * (p[0] - seed[0]),
                    self._w_trans * (p[1] - seed[1]),
                    self._w_rot * dt,
                ]
            )
            try:
                delta = np.linalg.solve(JtJ, -Jtr)
            except np.linalg.LinAlgError:
                break
            dn = np.linalg.norm(delta[:2])
            da = abs(delta[2])
            if dn > self._gn_max_step:
                delta[:2] *= self._gn_max_step / dn
            if da > self._gn_max_rot:
                delta[2] *= self._gn_max_rot / da
            p[:2] += delta[:2]
            p[2] += delta[2]
            if dn < 1e-4 and da < 1e-4:
                break
        return p

    def _integrate(self):
        wp.launch(
            add_k,
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
