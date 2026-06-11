import math
import time

import numpy as np
import warp

import slam

FAILURES = []
CHECKS = 0


def check(name, cond, detail=""):
    global CHECKS
    CHECKS += 1
    mark = "pass" if cond else "FAIL"
    print(f"{mark}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILURES.append(name)


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def compose(p, d):
    c, s = np.cos(p[2]), np.sin(p[2])
    return np.array(
        [p[0] + c * d[0] - s * d[1], p[1] + s * d[0] + c * d[1], p[2] + d[2]]
    )


def body_delta(p0, p1):
    c, s = np.cos(p0[2]), np.sin(p0[2])
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(p1[2] - p0[2])])


def pose_err(p, true):
    e = p - true
    return float(np.hypot(e[0], e[1])), abs(float(wrap(e[2])))


def segments(points, closed=True):
    pts = np.asarray(points, dtype=np.float64)
    cur = pts if closed else pts[:-1]
    nxt = np.roll(pts, -1, axis=0) if closed else pts[1:]
    return np.concatenate([cur, nxt], axis=1)


def box(cx, cy, w, h):
    return segments(
        [(cx - w, cy - h), (cx + w, cy - h), (cx + w, cy + h), (cx - w, cy + h)]
    )


def square_room(half=4.0, cx=0.0, cy=0.0):
    return box(cx, cy, half, half)


def corridor(half=1.0, length=200.0):
    return np.array(
        [[-length, -half, length, -half], [-length, half, length, half]],
        dtype=np.float64,
    )


def raycast(segs, origins, headings, angles, noise=0.0, rng=None):
    origins = np.broadcast_to(np.atleast_2d(origins), (len(angles), 2))
    ang = np.asarray(headings) + angles
    d = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    p1 = segs[None, :, 0:2]
    e = (segs[:, 2:4] - segs[:, 0:2])[None, :, :]
    ao = p1 - origins[:, None, :]
    denom = d[:, None, 0] * e[..., 1] - d[:, None, 1] * e[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ao[..., 0] * e[..., 1] - ao[..., 1] * e[..., 0]) / denom
        u = (ao[..., 0] * d[:, None, 1] - ao[..., 1] * d[:, None, 0]) / denom
    ok = (np.abs(denom) > 1e-12) & (t > 1e-6) & (u >= 0.0) & (u <= 1.0)
    r = np.where(ok, t, 1e9).min(axis=1)
    if noise and rng is not None:
        r = r + rng.normal(0.0, noise, r.shape)
    return r.astype(np.float32)


class Lidar:
    def __init__(self, n=1081, fov=270.0, period=0.025, acq_frac=0.75):
        self.n = n
        self.period = period
        self.acq_frac = acq_frac
        self.a_min = -np.radians(fov) / 2.0
        self.a_inc = np.radians(fov) / (n - 1)
        self.angles = self.a_min + self.a_inc * np.arange(n)
        self.beam_dt = (np.arange(n) / (n - 1) - 0.5) * acq_frac * period

    def scan(self, segs, pose_fn, t_mid, noise, rng, shutter=True):
        if shutter:
            poses = pose_fn(t_mid + self.beam_dt)
            return raycast(segs, poses[:, :2], poses[:, 2], self.angles, noise, rng)
        p = pose_fn(np.array([t_mid]))[0]
        return raycast(segs, p[:2], p[2], self.angles, noise, rng)


class Track:
    def __init__(self, sx=8.0, sy=3.0, rc=1.6, half_width=1.2):
        self.rc, self.w = rc, half_width
        self.sx, self.sy = sx, sy
        q = np.pi * rc / 2.0
        self.lens = np.array([sx, q, sy, q, sx, q, sy, q])
        self.bounds = np.concatenate([[0.0], np.cumsum(self.lens)])
        self.length = self.bounds[-1]
        self.segs = np.concatenate(
            [self._wall(rc - half_width), self._wall(rc + half_width)]
        )

    def centerline(self, s):
        s = np.asarray(s, dtype=np.float64) % self.length
        sx, sy, rc, b = self.sx, self.sy, self.rc, self.bounds
        x, y, h = np.empty_like(s), np.empty_like(s), np.empty_like(s)
        kap = np.zeros_like(s)
        cx = np.array([sx / 2, sx / 2, -sx / 2, -sx / 2])
        cy = np.array([-sy / 2, sy / 2, sy / 2, -sy / 2])
        a0 = np.array([-np.pi / 2, 0.0, np.pi / 2, np.pi])
        for p in range(8):
            m = (s >= b[p]) & (s < b[p + 1])
            if not m.any():
                continue
            u = s[m] - b[p]
            if p % 2 == 0:
                side = p // 2
                if side == 0:
                    x[m], y[m], h[m] = -sx / 2 + u, -sy / 2 - rc, 0.0
                elif side == 1:
                    x[m], y[m], h[m] = sx / 2 + rc, -sy / 2 + u, np.pi / 2
                elif side == 2:
                    x[m], y[m], h[m] = sx / 2 - u, sy / 2 + rc, np.pi
                else:
                    x[m], y[m], h[m] = -sx / 2 - rc, sy / 2 - u, -np.pi / 2
            else:
                ci = p // 2
                a = a0[ci] + u / rc
                x[m] = cx[ci] + rc * np.cos(a)
                y[m] = cy[ci] + rc * np.sin(a)
                h[m] = a + np.pi / 2
                kap[m] = 1.0 / rc
        return np.stack([x, y, h], axis=1), kap

    def _wall(self, radius):
        pts = []
        for p in range(8):
            if p % 2 == 0:
                s = np.array([self.bounds[p]])
            else:
                s = self.bounds[p] + np.linspace(0.0, self.lens[p], 17)
            c, _ = self.centerline(s)
            n = np.stack([-np.sin(c[:, 2]), np.cos(c[:, 2])], axis=1)
            pts.append(c[:, :2] + (radius - self.rc) * n)
        return segments(np.concatenate(pts))

    def trajectory(self, v_straight=7.0, v_corner=4.0, accel=5.0, ds=0.02):
        s = np.arange(0.0, 3 * self.length, ds)
        _, kap = self.centerline(s)
        v = np.where(kap > 0, v_corner, v_straight)
        v[0] = 0.0
        for i in range(1, len(v)):
            v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * accel * ds))
        for i in range(len(v) - 2, -1, -1):
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * accel * ds))
        t_of_s = np.concatenate([[0.0], np.cumsum(ds / np.maximum(v[1:], 1e-3))])
        return lambda t: self.centerline(np.interp(t, t_of_s, s))[0]


def arc_pose_fn(start, v, w):
    start = np.asarray(start, dtype=np.float64)

    def fn(t):
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        th = start[2] + w * t
        if abs(w) > 1e-9:
            x = start[0] + v / w * (np.sin(th) - np.sin(start[2]))
            y = start[1] - v / w * (np.cos(th) - np.cos(start[2]))
        else:
            x = start[0] + v * t * np.cos(start[2])
            y = start[1] + v * t * np.sin(start[2])
        return np.stack([x, y, th], axis=1)

    return fn


def drive(
    bridge,
    lidar,
    segs_fn,
    pose_fn,
    t_grid,
    rng,
    noise=0.005,
    odom_hook=None,
    scan_hook=None,
    shutter=True,
    deskew=None,
):
    if deskew is None:
        deskew = lidar.acq_frac if shutter else 0.0
    mids = pose_fn(t_grid)
    errs = []
    step_s = 0.0
    last = None
    for k, t in enumerate(t_grid):
        r = lidar.scan(segs_fn(t), pose_fn, t, noise, rng, shutter)
        if scan_hook is not None:
            r = scan_hook(k, r)
        if last is None:
            bridge.configure(lidar.n, lidar.a_min, lidar.a_inc)
            bridge.step(r, mids[k].astype(np.float32))
        else:
            d = body_delta(last, mids[k])
            d[0] *= 0.97
            d += rng.normal(0.0, [0.002, 0.001, np.radians(0.05)])
            if odom_hook is not None:
                d = odom_hook(k, d)
            cy, sy = np.cos(bridge.pose[2]), np.sin(bridge.pose[2])
            dk = deskew(k) if callable(deskew) else deskew
            t0 = time.perf_counter()
            bridge.step(
                r,
                np.array(
                    [cy * d[0] - sy * d[1], sy * d[0] + cy * d[1], d[2]],
                    dtype=np.float32,
                ),
                deskew_frac=dk,
            )
            step_s += time.perf_counter() - t0
        last = mids[k]
        errs.append(pose_err(bridge.pose, mids[k]))
    return np.array(errs), step_s / max(len(t_grid) - 1, 1) * 1e3


def dead_reckoning(pose_fn, t_grid, rng, odom_hook=None):
    mids = pose_fn(t_grid)
    p = mids[0].copy()
    for k in range(1, len(t_grid)):
        d = body_delta(mids[k - 1], mids[k])
        d[0] *= 0.97
        d += rng.normal(0.0, [0.002, 0.001, np.radians(0.05)])
        if odom_hook is not None:
            d = odom_hook(k, d)
        p = compose(p, d)
    return pose_err(p, mids[-1])


def test_room_tracking():
    rng = np.random.default_rng(0)
    lidar = Lidar()
    room = square_room()
    pose_fn = arc_pose_fn([-1.5, -1.0, 0.3], v=2.0, w=2.0 / 1.6)
    t = np.arange(140) * lidar.period
    b = slam.Bridge(device="cpu")
    errs, ms = drive(b, lidar, lambda _: room, pose_fn, t, rng)
    oxy, _ = dead_reckoning(pose_fn, t, np.random.default_rng(0))
    exy, eth = errs[-1]
    check(
        "room_tracking",
        exy < 0.03
        and np.degrees(eth) < 0.5
        and errs[:, 0].max() < 0.05
        and exy < oxy / 5.0,
        f"final {exy * 100:.2f} cm / {np.degrees(eth):.3f} deg, "
        f"max {errs[:, 0].max() * 100:.2f} cm, odom {oxy * 100:.1f} cm, "
        f"{ms:.1f} ms/scan cpu",
    )
    return b


def test_occupancy(bridge):
    occ = bridge.occupancy()
    grid, res, org = int(slam.GRID), float(slam.RES), float(slam.ORIGIN)

    def at(wx, wy):
        return occ[int(round((wy - org) / res)), int(round((wx - org) / res))]

    wall = max(at(4.0, 0.5), at(4.0, 0.0), at(0.5, -4.0))
    free = at(-1.0, -1.0)
    unknown = at(30.0, 30.0)
    check(
        "occupancy_semantics",
        occ.shape == (grid, grid)
        and occ.dtype == np.int8
        and wall > 60
        and 0 <= free < 20
        and unknown == -1,
        f"wall {wall}, free {free}, unknown {unknown}",
    )


def test_full_speed_track():
    track = Track()
    lidar = Lidar()
    pose_fn = track.trajectory()
    t = np.arange(0.0, 2.6 * track.length / 5.0, lidar.period)
    mids = pose_fn(t)
    speeds = np.linalg.norm(np.diff(mids[:, :2], axis=0), axis=1) / lidar.period
    yaw = np.abs(wrap(np.diff(mids[:, 2]))) / lidar.period

    b = slam.Bridge(device="cpu")
    errs, ms = drive(
        b, lidar, lambda _: track.segs, pose_fn, t, np.random.default_rng(1)
    )
    half = len(t) // 2
    check(
        "full_speed_track",
        errs[:, 0].max() < 0.10
        and np.degrees(errs[:, 1].max()) < 2.0
        and errs[half:, 0].max() < 0.10,
        f"{len(t)} scans, peak {speeds.max():.1f} m/s / {yaw.max():.1f} rad/s, "
        f"max {errs[:, 0].max() * 100:.1f} cm / "
        f"{np.degrees(errs[:, 1].max()):.2f} deg, "
        f"lap2 max {errs[half:, 0].max() * 100:.1f} cm, {ms:.1f} ms/scan cpu",
    )

    b2 = slam.Bridge(device="cpu")
    errs_off, _ = drive(
        b2,
        lidar,
        lambda _: track.segs,
        pose_fn,
        t,
        np.random.default_rng(1),
        deskew=0.0,
    )
    check(
        "deskew_helps_at_speed",
        errs[:, 0].max() < errs_off[:, 0].max()
        and errs[:, 1].max() < errs_off[:, 1].max(),
        f"max err {errs[:, 0].max() * 100:.1f} cm with deskew vs "
        f"{errs_off[:, 0].max() * 100:.1f} cm without",
    )


def test_dropped_scans_at_speed():
    track = Track()
    lidar = Lidar()
    pose_fn = track.trajectory()
    t_all = np.arange(0.0, 1.6 * track.length / 5.0, lidar.period)
    t = t_all[np.arange(len(t_all)) % 7 != 3]
    gaps = np.diff(t, prepend=t[0] - lidar.period)
    fracs = np.clip(lidar.acq_frac * lidar.period / gaps, 0.0, 1.0)
    b = slam.Bridge(device="cpu")
    errs, _ = drive(
        b,
        lidar,
        lambda _: track.segs,
        pose_fn,
        t,
        np.random.default_rng(8),
        deskew=lambda k: fracs[k],
    )
    check(
        "dropped_scans_at_speed",
        errs[:, 0].max() < 0.12 and np.degrees(errs[:, 1].max()) < 2.0,
        f"1 in 7 frames lost, {len(t)} scans, "
        f"max {errs[:, 0].max() * 100:.1f} cm / "
        f"{np.degrees(errs[:, 1].max()):.2f} deg",
    )


def test_corridor_degenerate():
    rng = np.random.default_rng(0)
    lidar = Lidar()
    segs = corridor()
    pose_fn = arc_pose_fn([0.0, 0.2, 0.0], v=2.0, w=0.0)
    t = np.arange(100) * lidar.period
    mids = pose_fn(t)
    b = slam.Bridge(device="cpu")
    lat = head = 0.0
    last = None
    for k, tk in enumerate(t):
        r = lidar.scan(segs, pose_fn, tk, 0.005, rng)
        if last is None:
            b.configure(lidar.n, lidar.a_min, lidar.a_inc)
            b.step(r, mids[k].astype(np.float32))
        else:
            d = body_delta(last, mids[k])
            d[0] *= 0.95
            cy, sy = np.cos(b.pose[2]), np.sin(b.pose[2])
            b.step(
                r,
                np.array(
                    [cy * d[0] - sy * d[1], sy * d[0] + cy * d[1], d[2]],
                    dtype=np.float32,
                ),
                deskew_frac=0.75,
            )
        last = mids[k]
        lat = max(lat, abs(b.pose[1] - mids[k][1]))
        head = max(head, abs(wrap(b.pose[2] - mids[k][2])))
    ex = abs(b.pose[0] - mids[-1][0])
    odom_ex = abs(0.05 * (mids[-1][0] - mids[0][0]))
    check(
        "corridor_degenerate",
        lat < 0.01 and np.degrees(head) < 0.3 and ex < odom_ex + 0.10,
        f"lateral {lat * 100:.2f} cm, heading {np.degrees(head):.3f} deg, "
        f"along {ex * 100:.1f} cm (odom {odom_ex * 100:.1f} cm)",
    )


def test_slip_and_glitch():
    rng = np.random.default_rng(2)
    lidar = Lidar()
    room = square_room()
    pose_fn = arc_pose_fn([-1.5, -1.0, 0.3], v=3.0, w=3.0 / 1.6)
    t = np.arange(120) * lidar.period

    def faults(k, d):
        if 40 <= k < 46:
            d[0] *= 0.55
        if k == 80:
            d[1] += 0.22
        return d

    b = slam.Bridge(device="cpu")
    errs, _ = drive(b, lidar, lambda _: room, pose_fn, t, rng, odom_hook=faults)
    check(
        "slip_and_seed_glitch",
        errs[:, 0].max() < 0.25
        and errs[50:60, 0].max() < 0.05
        and errs[84:, 0].max() < 0.05
        and errs[-1, 0] < 0.03,
        f"worst during faults {errs[:, 0].max() * 100:.1f} cm, "
        f"after slip {errs[50:60, 0].max() * 100:.1f} cm, "
        f"after glitch {errs[84:, 0].max() * 100:.1f} cm",
    )


def test_dropout():
    rng = np.random.default_rng(3)
    lidar = Lidar()
    room = square_room()
    pose_fn = arc_pose_fn([-1.5, -1.0, 0.3], v=2.5, w=2.5 / 1.6)
    t = np.arange(120) * lidar.period

    def holes(k, r):
        r = r.copy()
        r[rng.random(len(r)) < 0.35] = 0.0
        r[200:440] = np.float32(np.inf)
        return r

    b = slam.Bridge(device="cpu")
    errs, _ = drive(b, lidar, lambda _: room, pose_fn, t, rng, scan_hook=holes)
    check(
        "dropout_35pct_plus_dead_sector",
        errs[:, 0].max() < 0.06 and errs[-1, 0] < 0.04,
        f"max {errs[:, 0].max() * 100:.1f} cm, final {errs[-1, 0] * 100:.1f} cm",
    )


def test_dynamic_obstacle():
    rng = np.random.default_rng(4)
    lidar = Lidar()
    room = square_room()
    pose_fn = arc_pose_fn([-1.5, -1.0, 0.3], v=2.0, w=2.0 / 1.6)
    t = np.arange(120) * lidar.period

    def world(tk):
        k = tk / lidar.period
        if 30 <= k <= 80:
            return np.concatenate([room, box(0.8, -2.5 + (k - 30) * 0.08, 0.25, 0.18)])
        return room

    b = slam.Bridge(device="cpu")
    errs, _ = drive(b, lidar, world, pose_fn, t, rng)
    check(
        "dynamic_obstacle",
        errs[:, 0].max() < 0.06 and errs[-1, 0] < 0.04,
        f"max {errs[:, 0].max() * 100:.1f} cm with crossing obstacle",
    )


def test_spin_in_place():
    rng = np.random.default_rng(5)
    lidar = Lidar()
    room = square_room()
    pose_fn = arc_pose_fn([0.5, -0.7, 0.0], v=1e-4, w=5.0)
    t = np.arange(90) * lidar.period
    b = slam.Bridge(device="cpu")
    errs, _ = drive(b, lidar, lambda _: room, pose_fn, t, rng)
    check(
        "spin_5_rad_s",
        errs[:, 0].max() < 0.06 and np.degrees(errs[:, 1].max()) < 1.0,
        f"xy max {errs[:, 0].max() * 100:.1f} cm, heading max "
        f"{np.degrees(errs[:, 1].max()):.2f} deg over "
        f"{np.degrees(5.0 * t[-1]):.0f} deg of spin",
    )


def test_stationary_and_garbage():
    rng = np.random.default_rng(0)
    lidar = Lidar()
    room = square_room()
    pose = np.array([-1.5, -1.0, 0.3])
    b = slam.Bridge(device="cpu")
    b.configure(lidar.n, lidar.a_min, lidar.a_inc)
    b.step(
        raycast(room, pose[:2], pose[2], lidar.angles, 0.005, rng),
        pose.astype(np.float32),
    )
    p0 = b.pose.copy()
    for _ in range(15):
        b.step(
            raycast(room, pose[:2], pose[2], lidar.angles, 0.005, rng),
            np.zeros(3, dtype=np.float32),
        )
    drift = float(np.hypot(*(b.pose - p0)[:2]))

    finite = True
    for g in (
        np.full(lidar.n, np.nan, np.float32),
        np.full(lidar.n, np.inf, np.float32),
        np.zeros(lidar.n, np.float32),
    ):
        p = b.step(
            np.nan_to_num(g, nan=0.0, posinf=0.0),
            np.array([0.05, 0.0, 0.0], np.float32),
        )
        finite &= bool(np.all(np.isfinite(p)))
    for _ in range(8):
        pose = compose(pose, (0.05, 0.0, 0.0))
        b.step(
            raycast(room, pose[:2], pose[2], lidar.angles, 0.005, rng),
            np.array([0.05, 0.0, 0.0], np.float32),
        )
    exy, _ = pose_err(b.pose, pose)
    check(
        "stationary_and_garbage_scans",
        drift == 0.0 and finite and exy < 0.05,
        f"stationary drift {drift * 1000:.2f} mm, "
        f"recovery after garbage {exy * 100:.1f} cm",
    )


def test_reconfigure_and_map_edge():
    rng = np.random.default_rng(6)
    room = square_room(half=3.0, cx=-46.0, cy=-46.0)
    pose_fn = arc_pose_fn([-46.5, -46.5, 0.2], v=1.5, w=np.radians(50.0))
    a, b2 = Lidar(), Lidar(n=897, fov=240.0)
    t = np.arange(40) * a.period
    br = slam.Bridge(device="cpu")
    e1, _ = drive(br, a, lambda _: room, pose_fn, t, rng)
    reconfigured = br.configure(b2.n, b2.a_min, b2.a_inc)
    t2 = t[-1] + a.period + np.arange(40) * b2.period
    mids = pose_fn(t2)
    last = pose_fn(np.array([t[-1]]))[0]
    worst = 0.0
    for k, tk in enumerate(t2):
        r = b2.scan(room, pose_fn, tk, 0.005, rng)
        d = body_delta(last, mids[k])
        cy, sy = np.cos(br.pose[2]), np.sin(br.pose[2])
        br.step(
            r,
            np.array([cy * d[0] - sy * d[1], sy * d[0] + cy * d[1], d[2]], np.float32),
            deskew_frac=0.75,
        )
        worst = max(worst, pose_err(br.pose, mids[k])[0])
        last = mids[k]
    check(
        "reconfigure_and_map_edge",
        reconfigured and e1[:, 0].max() < 0.05 and worst < 0.05,
        f"near grid corner, 1081 -> 897 beams, "
        f"max {max(e1[:, 0].max(), worst) * 100:.1f} cm",
    )


def test_runtime_contracts():
    counts = {"launch": 0, "up": 0, "down": 0, "alloc": 0}
    arr_cls = type(warp.zeros(1, device="cpu"))
    o_launch, o_zeros, o_array = warp.launch, warp.zeros, warp.array
    o_numpy, o_assign = arr_cls.numpy, arr_cls.assign

    def launch(*a, **k):
        counts["launch"] += 1
        return o_launch(*a, **k)

    def zeros(*a, **k):
        counts["alloc"] += 1
        return o_zeros(*a, **k)

    def arr(*a, **k):
        counts["alloc"] += 1
        return o_array(*a, **k)

    def npy(self_):
        counts["down"] += 1
        return o_numpy(self_)

    def assign(self_, v):
        counts["up"] += 1
        return o_assign(self_, v)

    rng = np.random.default_rng(7)
    lidar = Lidar()
    room = square_room()
    pose = np.array([-1.5, -1.0, 0.3])
    b = slam.Bridge(device="cpu")
    b.configure(lidar.n, lidar.a_min, lidar.a_inc)
    b.step(
        raycast(room, pose[:2], pose[2], lidar.angles, 0.005, rng),
        pose.astype(np.float32),
    )
    for _ in range(5):
        pose = compose(pose, (0.05, 0.0, 0.02))
        b.step(
            raycast(room, pose[:2], pose[2], lidar.angles, 0.005, rng),
            np.array([0.05, 0.0, 0.02], np.float32),
        )

    warp.launch, warp.zeros, warp.array = launch, zeros, arr
    arr_cls.numpy, arr_cls.assign = npy, assign
    try:
        per_step = []
        for _ in range(30):
            pose = compose(pose, (0.05, 0.0, 0.02))
            before = dict(counts)
            b.step(
                raycast(room, pose[:2], pose[2], lidar.angles, 0.005, rng),
                np.array([0.05, 0.0, 0.02], np.float32),
            )
            per_step.append({k: counts[k] - before[k] for k in counts})
    finally:
        warp.launch, warp.zeros, warp.array = o_launch, o_zeros, o_array
        arr_cls.numpy, arr_cls.assign = o_numpy, o_assign

    launches = {p["launch"] for p in per_step}
    downs = {p["down"] for p in per_step}
    ups = max(p["up"] for p in per_step)
    allocs = sum(p["alloc"] for p in per_step)
    check(
        "runtime_contracts",
        launches <= {22, 23} and downs == {1} and ups <= 2 and allocs == 0,
        f"launches/step {sorted(launches)}, downloads/step {sorted(downs)}, "
        f"uploads/step <= {ups}, device allocations {allocs}",
    )


def main():
    t0 = time.perf_counter()
    b = test_room_tracking()
    test_occupancy(b)
    test_full_speed_track()
    test_dropped_scans_at_speed()
    test_corridor_degenerate()
    test_slip_and_glitch()
    test_dropout()
    test_dynamic_obstacle()
    test_spin_in_place()
    test_stationary_and_garbage()
    test_reconfigure_and_map_edge()
    test_runtime_contracts()
    dt = time.perf_counter() - t0
    print(
        f"\n{CHECKS - len(FAILURES)}/{CHECKS} passed in {dt:.0f} s"
        + (f", failures: {FAILURES}" if FAILURES else "")
    )
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
