import array
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster

import slam


def yaw_q(z, w):
    return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def compose(a, b):
    ca, sa = math.cos(a[2]), math.sin(a[2])
    return (a[0] + ca * b[0] - sa * b[1], a[1] + sa * b[0] + ca * b[1], wrap(a[2] + b[2]))


def invert(p):
    c, s = math.cos(p[2]), math.sin(p[2])
    return (-(c * p[0] + s * p[1]), s * p[0] - c * p[1], -p[2])


PARAMS = dict(
    scan_topic="/scan", odom_topic="/odom",
    map_frame="map", odom_frame="odom",
    invert_scan=False, max_usable_range=10.0,
    laser_offset_x=0.0, map_publish_period_s=2.0,
)


class SlamNode(Node):
    def __init__(self):
        super().__init__("slam_node")
        for k, v in PARAMS.items():
            self.declare_parameter(k, v)
        g = lambda k: self.get_parameter(k).value
        self.map_frame, self.odom_frame = g("map_frame"), g("odom_frame")
        self._invert = bool(g("invert_scan"))
        self._max_range = float(g("max_usable_range"))
        self._loff = float(g("laser_offset_x"))

        self.bridge = slam.Bridge()
        self._cfg = False
        self.pose = np.zeros(3, dtype=np.float32)
        self.odom = None
        self.prev_odom = None
        self._last_integ = -1

        self.create_subscription(LaserScan, g("scan_topic"), self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, g("odom_topic"), self.on_odom, qos_profile_sensor_data)
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", map_qos)
        self.pose_pub = self.create_publisher(PoseStamped, "/slam_pose", 10)
        self.tf = TransformBroadcaster(self)
        self.create_timer(float(g("map_publish_period_s")), self.publish_map)

        gw = int(slam.GRID)
        self._md = np.full(gw * gw, -1, dtype=np.int8)
        self._mm = OccupancyGrid()
        self._mm.info.resolution = float(slam.RES)
        self._mm.info.width = gw
        self._mm.info.height = gw
        self._mm.info.origin.position.x = float(slam.ORIGIN)
        self._mm.info.origin.position.y = float(slam.ORIGIN)
        self._mm.info.origin.orientation.w = 1.0
        self._ps = PoseStamped()
        self._ps.header.frame_id = self.map_frame
        self._tfm = TransformStamped()
        self._tfm.header.frame_id = self.map_frame
        self._tfm.child_frame_id = self.odom_frame

        self.get_logger().info(
            f"slam_node up | device={self.bridge.device} | "
            f"scan={g('scan_topic')} odom={g('odom_topic')} | "
            f"invert_scan={self._invert} | publishes {self.map_frame}->{self.odom_frame} "
            "(needs vesc_to_odom/publish_tf:=true)"
        )

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose
        self.odom = np.array(
            [p.position.x, p.position.y, yaw_q(p.orientation.z, p.orientation.w)],
            dtype=np.float32,
        )

    def on_scan(self, msg: LaserScan):
        n = len(msg.ranges)
        if n == 0:
            return
        r = np.asarray(msg.ranges, dtype=np.float32).copy()
        np.nan_to_num(r, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        hi = min(float(msg.range_max), self._max_range)
        r[~np.isfinite(r) | (r < float(msg.range_min)) | (r > hi)] = 0.0

        sgn = -1.0 if self._invert else 1.0
        a_min, a_inc = sgn * float(msg.angle_min), sgn * float(msg.angle_increment)
        if not self._cfg or self.bridge.n_beams != n:
            if self.bridge.configure(n, a_min, a_inc):
                fov = math.degrees(abs(a_inc) * (n - 1))
                self.get_logger().info(f"scan: {n} beams, FOV={fov:.1f} deg")
                if not (1000 <= n <= 1200) or abs(fov - 270.0) > 5.0:
                    self.get_logger().warn("scan unlike UST-10LX; check driver / invert_scan")
            self._cfg = True

        delta = np.zeros(3, dtype=np.float32)
        if self.odom is not None and self.prev_odom is not None:
            dx, dy = float(self.odom[0] - self.prev_odom[0]), float(self.odom[1] - self.prev_odom[1])
            py = float(self.prev_odom[2])
            cp, sp = math.cos(py), math.sin(py)
            bx, by = dx * cp + dy * sp, -dx * sp + dy * cp
            dyaw = wrap(float(self.odom[2]) - py)
            sy = float(self.pose[2])
            cs, ss = math.cos(sy), math.sin(sy)
            delta = np.array([bx * cs - by * ss, bx * ss + by * cs, dyaw], dtype=np.float32)
        self.prev_odom = None if self.odom is None else self.odom.copy()

        self.pose = self.bridge.step(r, delta)
        self.publish_pose_tf(msg.header.stamp)

    def publish_pose_tf(self, stamp):
        laser = (float(self.pose[0]), float(self.pose[1]), float(self.pose[2]))
        mb = compose(laser, (-self._loff, 0.0, 0.0))
        ps = self._ps
        ps.header.stamp = stamp
        ps.pose.position.x, ps.pose.position.y = mb[0], mb[1]
        ps.pose.orientation.z = math.sin(mb[2] * 0.5)
        ps.pose.orientation.w = math.cos(mb[2] * 0.5)
        self.pose_pub.publish(ps)

        if self.odom is None:
            return
        mo = compose(mb, invert((float(self.odom[0]), float(self.odom[1]), float(self.odom[2]))))
        t = self._tfm
        t.header.stamp = stamp
        t.transform.translation.x, t.transform.translation.y = mo[0], mo[1]
        t.transform.rotation.z = math.sin(mo[2] * 0.5)
        t.transform.rotation.w = math.cos(mo[2] * 0.5)
        self.tf.sendTransform(t)

    def publish_map(self):
        count = self.bridge.integrations
        if count == self._last_integ:
            return
        self._last_integ = count
        now = self.get_clock().now().to_msg()
        m = self._mm
        m.header.stamp = m.info.map_load_time = now
        m.header.frame_id = self.map_frame
        lo = self.bridge.logodds.numpy().ravel()
        self._md.fill(-1)
        self._md[lo > 0.5] = 100
        self._md[lo < -0.5] = 0
        m.data = array.array("b", self._md.tobytes())
        self.map_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = SlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
