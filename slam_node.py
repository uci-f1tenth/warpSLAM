import math
import time
from array import array

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster

import slam


def _yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class SlamNode(Node):
    def __init__(self):
        super().__init__("warp_slam")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("max_usable_range", 10.0)
        self.declare_parameter("deskew", True)
        self.declare_parameter("map_publish_period", 0.2)
        self.declare_parameter("path_min_step", 0.10)
        self.declare_parameter("device", "")

        self.map_frame = self.get_parameter("map_frame").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.max_range = float(self.get_parameter("max_usable_range").value)
        self.deskew = bool(self.get_parameter("deskew").value)
        self.path_step = float(self.get_parameter("path_min_step").value)
        dev = self.get_parameter("device").value or None

        self.bridge = slam.Bridge(device=dev)
        self.get_logger().info(f"slam device: {self.bridge.device}")
        if not str(self.bridge.device).startswith("cuda"):
            self.get_logger().warning("no CUDA device, running on CPU")

        self._odom = None
        self._last_odom = None
        self._last_stamp = None
        self._last_body = (0.0, 0.0, 0.0)
        self._scan_period = 0.025
        self._published_integrations = -1
        self._path = []
        self._path_published = 0
        self._t_acc = 0.0
        self._t_max = 0.0
        self._t_n = 0

        sensor_qos = QoSProfile(depth=1)
        sensor_qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        self.create_subscription(
            LaserScan, self.get_parameter("scan_topic").value,
            self.on_scan, sensor_qos)
        self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value,
            self.on_odom, 20)

        latched = QoSProfile(depth=1)
        latched.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.pose_pub = self.create_publisher(PoseStamped, "/slam_pose", 10)
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", latched)
        self.path_pub = self.create_publisher(Path, "/slam_path", latched)
        self.tf = TransformBroadcaster(self)
        self.create_timer(
            float(self.get_parameter("map_publish_period").value),
            self.publish_map)

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose
        self._odom = (p.position.x, p.position.y, _yaw_from_quat(p.orientation))

    def on_scan(self, msg: LaserScan):
        try:
            self._step(msg)
        except Exception as e:
            self.get_logger().error(f"step failed: {e}",
                                    throttle_duration_sec=2.0)

    def _step(self, msg: LaserScan):
        if self._odom is None:
            self.get_logger().warning("no odom yet", throttle_duration_sec=2.0)
            return

        n = len(msg.ranges)
        if self.bridge.configure(n, msg.angle_min, msg.angle_increment):
            self.get_logger().info(f"configured {n} beams")

        r = np.nan_to_num(
            np.asarray(msg.ranges, dtype=np.float32),
            nan=0.0, posinf=0.0, neginf=0.0)
        if self.max_range > 0.0:
            r[r > self.max_range] = 0.0

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._last_stamp is not None:
            dt = stamp - self._last_stamp
            if 1e-4 < dt < 0.5:
                self._scan_period = dt
        self._last_stamp = stamp

        odom = self._odom
        if self._last_odom is None:
            self._last_odom = odom
            self.bridge.step(r, np.array(odom, dtype=np.float32))
            self._publish_pose(msg)
            return

        lx, ly, lth = self._last_odom
        dxw, dyw = odom[0] - lx, odom[1] - ly
        c, s = math.cos(lth), math.sin(lth)
        bx, by = c * dxw + s * dyw, -s * dxw + c * dyw
        dth = _wrap(odom[2] - lth)
        self._last_odom = odom
        if math.hypot(bx, by) > 1.0 or abs(dth) > 1.5:
            self.get_logger().warning(
                f"odom jump ({bx:.2f}, {by:.2f}, {dth:.2f}), holding velocity",
                throttle_duration_sec=1.0)
            bx, by, dth = self._last_body
        else:
            self._last_body = (bx, by, dth)

        yaw = float(self.bridge.pose[2])
        cy, sy = math.cos(yaw), math.sin(yaw)
        delta = np.array([cy * bx - sy * by, sy * bx + cy * by, dth],
                         dtype=np.float32)

        frac = 0.0
        if self.deskew:
            sweep = abs(msg.time_increment) * (n - 1)
            if sweep == 0.0:
                sweep = abs(msg.angle_increment) * (n - 1) / (2.0 * math.pi) \
                    * self._scan_period
            frac = min(max(sweep / self._scan_period, 0.0), 1.0)

        t0 = time.perf_counter()
        self.bridge.step(r, delta, deskew_frac=frac)
        dt = time.perf_counter() - t0
        self._t_acc += dt
        self._t_max = max(self._t_max, dt)
        self._t_n += 1
        if self._t_n == 400:
            self.get_logger().info(
                f"step avg {self._t_acc / self._t_n * 1e3:.2f} ms "
                f"max {self._t_max * 1e3:.2f} ms")
            self._t_acc = self._t_max = 0.0
            self._t_n = 0

        self._publish_pose(msg)

    def _publish_pose(self, scan_msg):
        x, y, th = (float(v) for v in self.bridge.pose)
        stamp = scan_msg.header.stamp

        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.map_frame
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.orientation.z = math.sin(th * 0.5)
        ps.pose.orientation.w = math.cos(th * 0.5)
        self.pose_pub.publish(ps)

        if (not self._path
                or math.hypot(x - self._path[-1].pose.position.x,
                              y - self._path[-1].pose.position.y)
                >= self.path_step):
            self._path.append(ps)
            if len(self._path) > 2000:
                del self._path[:200]
                self._path_published = 0

        ox, oy, oth = self._last_odom if self._last_odom else (0.0, 0.0, 0.0)
        dth = _wrap(th - oth)
        c, s = math.cos(dth), math.sin(dth)
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame
        t.transform.translation.x = x - (c * ox - s * oy)
        t.transform.translation.y = y - (s * ox + c * oy)
        t.transform.rotation.z = math.sin(dth * 0.5)
        t.transform.rotation.w = math.cos(dth * 0.5)
        self.tf.sendTransform(t)

    def publish_map(self):
        now = self.get_clock().now().to_msg()
        if len(self._path) != self._path_published:
            self._path_published = len(self._path)
            path = Path()
            path.header.stamp = now
            path.header.frame_id = self.map_frame
            path.poses = self._path
            self.path_pub.publish(path)

        if self.bridge.integrations == self._published_integrations:
            return
        snap = self.bridge.snapshot()
        if snap is None:
            return
        self._published_integrations = self.bridge.integrations
        crop, x0, y0 = snap
        res = float(slam.RES)
        origin = float(slam.ORIGIN) - 0.5 * res

        m = OccupancyGrid()
        m.header.stamp = now
        m.header.frame_id = self.map_frame
        m.info.map_load_time = now
        m.info.resolution = res
        m.info.width = crop.shape[1]
        m.info.height = crop.shape[0]
        m.info.origin.position.x = origin + x0 * res
        m.info.origin.position.y = origin + y0 * res
        m.info.origin.orientation.w = 1.0
        m.data = array("b", np.ascontiguousarray(crop).tobytes())
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
