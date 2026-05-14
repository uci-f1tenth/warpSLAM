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

LIDAR_FOV = wp.constant(wp.radians(270.0))
LIDAR_N = wp.constant(1081)
RMIN = wp.constant(0.1)
RMAX = wp.constant(40.0)
STRIDE = wp.constant(8)
N_MATCH = wp.constant(LIDAR_N // STRIDE)
A_MIN = wp.constant(-LIDAR_FOV * 0.5)
A_INC = wp.constant(LIDAR_FOV / LIDAR_N)
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
    for b in range(N_MATCH):
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
def gn_k(
    r: wp.array[float],
    like: wp.array2d[float],
    x: float,
    y: float,
    theta: float,
    w: float,
    res: wp.array[float],
    jac: wp.array[float],
    vld: wp.array[int],
):
    i = wp.tid()
    res[i] = 0.0
    jac[i * 3 + 0] = 0.0
    jac[i * 3 + 1] = 0.0
    jac[i * 3 + 2] = 0.0
    v = r[i]
    if not (RMIN <= v < RMAX):
        vld[i] = 0
        return
    vld[i] = 1

    a = A_MIN + A_INC * float(i) + theta
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
    res[i] = rw * cost

    s = -rw / float(L_MAX) * float(INV_RES)
    jac[i * 3] = s * ddu
    jac[i * 3 + 1] = s * ddv
    jac[i * 3 + 2] = s * (ddu * (-v * sa) + ddv * (v * ca))


@wp.kernel
def add_k(
    r: wp.array[float],
    px: float,
    py: float,
    pt: float,
    g: wp.array2d[float],
):
    i = wp.tid()
    v = r[i]
    if not (RMIN <= v < RMAX):
        return
    a = A_MIN + A_INC * float(i) + pt
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
    for _ in range(n):
        ix = int(gx)
        iy = int(gy)
        if 0 <= ix < GRID_W and 0 <= iy < GRID_H:
            wp.atomic_add(g, iy, ix, L_FREE)
        gx += ux
        gy += uy
    hx = int(x1)
    hy = int(y1)
    d = L_OCC - L_FREE
    sp = d * L_SPLAT
    for dyk in range(-1, 2):
        for dxk in range(-1, 2):
            nxk = hx + dxk
            nyk = hy + dyk
            if 0 <= nxk < GRID_W and 0 <= nyk < GRID_H:
                wp.atomic_add(g, nyk, nxk, d if (dxk == 0 and dyk == 0) else sp)


@wp.kernel
def clamp_k(g: wp.array2d[float]):
    i, j = wp.tid()
    g[i, j] = wp.clamp(g[i, j], L_MIN, L_MAX)


class Bridge:
    def __init__(self, config=None):
        wp.init()
        opts = dict(config or {})
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

        n = int(LIDAR_N)
        stride = opts.get("beam_stride", 8)
        nm = n // stride
        fov = float(LIDAR_FOV)
        step = fov / n
        a = -fov * 0.5 + step * stride * np.arange(nm, dtype=np.float32)
        self._ct = wp.array(np.cos(a))
        self._st = wp.array(np.sin(a))
        self.logodds = wp.zeros((int(GRID_H), int(GRID_W)))
        self._r = wp.zeros(n)

        hx, htd, sx, std = self._coarse
        self._nx = 2 * int(hx / sx) + 1
        self._nt = 2 * int(np.radians(htd) / np.radians(std)) + 1
        self._sx = sx
        self._st_step = float(np.radians(std))
        self._sc = wp.zeros(self._nx * self._nx * self._nt, dtype=float)

        ox = (np.arange(self._nx) - (self._nx - 1) * 0.5) * self._sx
        ot = (np.arange(self._nt) - (self._nt - 1) * 0.5) * self._st_step
        gx, gy, gt = np.meshgrid(ox, ox, ot, indexing="ij")
        self._coarse_ot = ot.astype(np.float32)
        self._coarse_pen = np.exp(
            -(gx**2 + gy**2) * self._cs_wt**2 - gt**2 * self._cs_wr**2
        ).astype(np.float32)

        self._gn_res = wp.zeros(n)
        self._gn_jac = wp.zeros(n * 3)
        self._gn_vld = wp.zeros(n, dtype=int)

        self.pose = np.zeros(3, dtype=np.float32)
        self._kf = np.zeros(3, dtype=np.float32)
        self._first = True
        self.last_score = 0.0

    def step(self, rn, odom):
        rn = np.ascontiguousarray(rn, dtype=np.float32)
        self._r.assign(rn)
        if self._first:
            return self._init(odom)
        seed = self.pose + odom.astype(np.float32)
        moved = np.linalg.norm(odom[:2]) > self._min_move or abs(odom[2]) > np.radians(
            0.5
        )
        hit = np.any((rn >= float(RMIN)) & (rn < float(RMAX)))
        if not hit:
            self.pose = seed
        elif not moved:
            self.pose = seed
        else:
            self.pose = self._match(seed)
        d = self.pose - self._kf
        if d[0] * d[0] + d[1] * d[1] > self._kf_d2 or abs(d[2]) > self._kf_dt:
            self._integrate()
        return self.pose

    def as_occupancy(self):
        lo = self.logodds.numpy()
        out = np.full(lo.shape, -1, dtype=np.int8)
        known = np.abs(lo) > 0.1
        p = 1.0 / (1.0 + np.exp(-np.clip(lo[known], -10.0, 10.0)))
        out[known] = np.clip((p * 100).astype(np.int8), 0, 100)
        return out

    def save_map(self, path):
        np.savez_compressed(path, logodds=self.logodds.numpy(), pose=self.pose)

    def load_map(self, path):
        d = np.load(path)
        self.logodds.assign(d["logodds"])
        self.pose = d["pose"]
        self._kf = self.pose.copy()
        self._first = False

    def reset(self):
        self.logodds.zero_()
        self.pose.fill(0)
        self._kf.fill(0)
        self._first = True
        self.last_score = 0.0

    def _init(self, odom):
        self.pose = odom.astype(np.float32)
        self._kf = self.pose.copy()
        self._first = False
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
                self._sc,
            ],
        )
        sc = self._sc.numpy()[: nx * nx * nt].reshape(nx, nx, nt)
        weighted = sc * self._coarse_pen
        best = np.unravel_index(int(np.argmax(weighted)), sc.shape)
        p[2] = seed[2] + float(self._coarse_ot[best[2]])
        self.last_score = float(weighted[best])

        n_lidar = int(LIDAR_N)
        for _ in range(self._gn_iters):
            wp.launch(
                gn_k,
                dim=n_lidar,
                inputs=[
                    self._r,
                    self.logodds,
                    p[0],
                    p[1],
                    p[2],
                    self._w_occ,
                    self._gn_res,
                    self._gn_jac,
                    self._gn_vld,
                ],
            )
            v = self._gn_vld.numpy().astype(bool)
            nv = v.sum()
            if nv < 3:
                break
            r = self._gn_res.numpy()[v]
            J = self._gn_jac.numpy().reshape(n_lidar, 3)[v]

            sn = np.sqrt(nv)
            r /= sn
            J /= sn

            dt = (p[2] - seed[2] + np.pi) % (2.0 * np.pi) - np.pi
            JtJ = J.T @ J + np.diag([self._w_trans, self._w_trans, self._w_rot])
            Jtr = J.T @ r + np.array(
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
            dim=int(LIDAR_N),
            inputs=[
                self._r,
                self.pose[0],
                self.pose[1],
                self.pose[2],
                self.logodds,
            ],
        )
        wp.launch(clamp_k, dim=(int(GRID_H), int(GRID_W)), inputs=[self.logodds])
        self._kf = self.pose.copy()
