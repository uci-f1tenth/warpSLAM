import array
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster

import slam


def euler_from_quat(x, y, z, w):
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def compose(a, b):
    ca, sa = math.cos(a[2]), math.sin(a[2])
    return (
        a[0] + ca * b[0] - sa * b[1],
        a[1] + sa * b[0] + ca * b[1],
        wrap_to_pi(a[2] + b[2]),
    )


def invert(p):
    c, s = math.cos(p[2]), math.sin(p[2])
    return (-(c * p[0] + s * p[1]), -(-s * p[0] + c * p[1]), -p[2])


class SlamNode(Node):
    def __init__(self):
        super().__init__("slam_node")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tf_mode", "map_base")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("map_publish_period_s", 1.0)
        self.declare_parameter("invert_scan", False)
        self.declare_parameter("max_usable_range", 10.0)
        self.declare_parameter("min_usable_range", 0.0)
        self.declare_parameter("laser_offset_x", 0.0)
        self.declare_parameter("laser_offset_y", 0.0)
        self.declare_parameter("laser_offset_yaw", 0.0)
        self.declare_parameter("warp_device", "")
        self.declare_parameter("filter_ground", False)
        self.declare_parameter("lidar_height", 0.3)
        self.declare_parameter("ground_filter_margin", 0.15)

        gp = self.get_parameter
        scan_topic = gp("scan_topic").value
        odom_topic = gp("odom_topic").value
        self.map_frame = gp("map_frame").value
        self.odom_frame = gp("odom_frame").value
        self.base_frame = gp("base_frame").value
        self.tf_mode = str(gp("tf_mode").value).lower()
        self.publish_tf_flag = bool(gp("publish_tf").value)
        map_period = float(gp("map_publish_period_s").value)
        self._invert_scan = bool(gp("invert_scan").value)
        self._max_range = float(gp("max_usable_range").value)
        self._min_range = float(gp("min_usable_range").value)
        self._laser_off = (
            float(gp("laser_offset_x").value),
            float(gp("laser_offset_y").value),
            float(gp("laser_offset_yaw").value),
        )
        warp_device = str(gp("warp_device").value).strip()
        self._filter_ground = bool(gp("filter_ground").value)
        self._lidar_height = float(gp("lidar_height").value)
        self._ground_margin = float(gp("ground_filter_margin").value)

        if self.tf_mode not in ("map_base", "map_odom"):
            self.tf_mode = "map_base"

        cfg = {"device": warp_device} if warp_device else {}
        self.bridge = slam.Bridge(cfg)
        self._scan_configured = False

        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, qos_profile_sensor_data
        )
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, qos_profile_sensor_data
        )
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", map_qos)
        self.pose_pub = self.create_publisher(PoseStamped, "/slam_pose", 10)
        self.tf_bcast = TransformBroadcaster(self) if self.publish_tf_flag else None
        self.map_timer = self.create_timer(map_period, self.publish_map)

        self.pose = np.zeros(3, dtype=np.float32)
        self.latest_odom = None
        self.prev_odom_at_scan = None

        gw, gh = int(slam.GRID_W), int(slam.GRID_H)
        self._map_data = np.full(gw * gh, -1, dtype=np.int8)
        self._map_msg = OccupancyGrid()
        self._map_msg.info.resolution = float(slam.RES)
        self._map_msg.info.width = gw
        self._map_msg.info.height = gh
        self._map_msg.info.origin.position.x = float(slam.OX)
        self._map_msg.info.origin.position.y = float(slam.OY)
        self._map_msg.info.origin.orientation.w = 1.0

        self._ps_msg = PoseStamped()
        self._ps_msg.header.frame_id = self.map_frame
        self._tf_msg = TransformStamped()
        self._gf_cache = None
        self._last_kf_pose = None

        self.get_logger().info(
            f"slam_node up | device={self.bridge.device} tf_mode={self.tf_mode}"
            f" | scan={scan_topic} odom={odom_topic}"
            f" | invert_scan={self._invert_scan}"
        )

    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose
        q = p.orientation
        roll, pitch, yaw = euler_from_quat(q.x, q.y, q.z, q.w)
        self.latest_odom = np.array(
            [p.position.x, p.position.y, yaw, roll, pitch], dtype=np.float32
        )

    def scan_callback(self, msg: LaserScan):
        n = len(msg.ranges)
        if n == 0:
            return
        ranges = np.asarray(msg.ranges, dtype=np.float32).copy()
        np.nan_to_num(ranges, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        lo = max(float(msg.range_min), float(self._min_range))
        hi = min(float(msg.range_max), float(self._max_range))
        ranges[~np.isfinite(ranges) | (ranges < lo) | (ranges > hi)] = 0.0

        if self._invert_scan:
            a_min, a_inc = -float(msg.angle_min), -float(msg.angle_increment)
        else:
            a_min, a_inc = float(msg.angle_min), float(msg.angle_increment)

        if not self._scan_configured or self.bridge._n != n:
            if self.bridge.configure(n, a_min, a_inc):
                fov = math.degrees(abs(a_inc) * (n - 1))
                self.get_logger().info(f"scan configured: {n} beams, FOV={fov:.1f} deg")
                if not (1000 <= n <= 1200) or abs(fov - 270.0) > 5.0:
                    self.get_logger().warn(
                        "scan unlike standard UST-10LX; check driver/invert_scan"
                    )
            self._scan_configured = True

        if self.latest_odom is not None and self.prev_odom_at_scan is not None:
            dx = float(self.latest_odom[0] - self.prev_odom_at_scan[0])
            dy = float(self.latest_odom[1] - self.prev_odom_at_scan[1])
            prev_yaw = float(self.prev_odom_at_scan[2])
            cp, sp = math.cos(prev_yaw), math.sin(prev_yaw)
            bx = dx * cp + dy * sp
            by = -dx * sp + dy * cp
            dyaw = wrap_to_pi(float(self.latest_odom[2]) - prev_yaw)
            cs, ss = math.cos(float(self.pose[2])), math.sin(float(self.pose[2]))
            odom_delta = np.array(
                [bx * cs - by * ss, bx * ss + by * cs, dyaw], dtype=np.float32
            )
        else:
            odom_delta = np.zeros(3, dtype=np.float32)
        self.prev_odom_at_scan = (
            self.latest_odom.copy() if self.latest_odom is not None else None
        )

        if (
            self._filter_ground
            and self.latest_odom is not None
            and (
                abs(float(self.latest_odom[3])) > 1e-3
                or abs(float(self.latest_odom[4])) > 1e-3
            )
        ):
            roll = float(self.latest_odom[3])
            pitch = float(self.latest_odom[4])
            c = self._gf_cache
            if c is None or c[0] != n or c[1] != a_min or c[2] != a_inc:
                angs = a_min + a_inc * np.arange(n, dtype=np.float32)
                self._gf_cache = (n, a_min, a_inc, np.cos(angs), np.sin(angs))
            _, _, _, cos_a, sin_a = self._gf_cache
            z_dir = -cos_a * pitch + sin_a * roll
            down = z_dir < -1e-6
            if down.any():
                d_ground = self._lidar_height / (-z_dir[down])
                rd = ranges[down]
                rd[rd >= (1.0 - self._ground_margin) * d_ground] = 0.0
                ranges[down] = rd

        self.pose = self.bridge.step(ranges, odom_delta)
        self._publish_pose_and_tf(msg.header.stamp)

    def _publish_pose_and_tf(self, stamp):
        map_base = compose(
            (float(self.pose[0]), float(self.pose[1]), float(self.pose[2])),
            invert(self._laser_off),
        )
        x, y, yaw = map_base
        qz, qw = math.sin(yaw * 0.5), math.cos(yaw * 0.5)

        ps = self._ps_msg
        ps.header.stamp = stamp
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        self.pose_pub.publish(ps)

        if self.tf_bcast is None:
            return
        tf = self._tf_msg
        tf.header.stamp = stamp
        if self.tf_mode == "map_odom" and self.latest_odom is not None:
            mo = compose(
                map_base,
                invert(
                    (
                        float(self.latest_odom[0]),
                        float(self.latest_odom[1]),
                        float(self.latest_odom[2]),
                    )
                ),
            )
            tf.header.frame_id = self.map_frame
            tf.child_frame_id = self.odom_frame
            tf.transform.translation.x = mo[0]
            tf.transform.translation.y = mo[1]
            tf.transform.rotation.z = math.sin(mo[2] * 0.5)
            tf.transform.rotation.w = math.cos(mo[2] * 0.5)
        else:
            tf.header.frame_id = self.map_frame
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = x
            tf.transform.translation.y = y
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
        self.tf_bcast.sendTransform(tf)

    def publish_map(self):
        kf = tuple(self.bridge._kf.tolist())
        if self._last_kf_pose == kf:
            return
        self._last_kf_pose = kf
        now = self.get_clock().now().to_msg()
        msg = self._map_msg
        msg.header.stamp = now
        msg.header.frame_id = self.map_frame
        msg.info.map_load_time = now
        lo = self.bridge.logodds.numpy().ravel()
        data = self._map_data
        data.fill(-1)
        known = np.abs(lo) > 0.1
        if known.any():
            p = 1.0 / (1.0 + np.exp(-np.clip(lo[known], -10.0, 10.0)))
            data[known] = (p * 100.0).astype(np.int8)
        msg.data = array.array("b", data.tobytes())
        self.map_pub.publish(msg)


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
