import math
import time
from array import array

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster

import slam


def _yaw_from_quat(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


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
        self.declare_parameter("map_publish_period", 1.0)
        self.declare_parameter("device", "")

        self.map_frame = self.get_parameter("map_frame").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.max_range = float(self.get_parameter("max_usable_range").value)
        self.deskew = bool(self.get_parameter("deskew").value)
        dev = self.get_parameter("device").value or None

        self.bridge = slam.Bridge(device=dev)
        self.get_logger().info(f"slam device: {self.bridge.device}")

        self._odom = None
        self._last_odom = None
        self._last_stamp = None
        self._scan_period = 0.025
        self._published_integrations = -1
        self._t_acc = 0.0
        self._t_max = 0.0
        self._t_n = 0

        sensor_qos = QoSProfile(depth=1)
        sensor_qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        self.create_subscription(
            LaserScan, self.get_parameter("scan_topic").value, self.on_scan, sensor_qos
        )
        self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value, self.on_odom, 20
        )

        self.pose_pub = self.create_publisher(PoseStamped, "/slam_pose", 10)
        map_qos = QoSProfile(depth=1)
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", map_qos)
        self.tf = TransformBroadcaster(self)
        self.create_timer(
            float(self.get_parameter("map_publish_period").value), self.publish_map
        )

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose
        self._odom = (p.position.x, p.position.y, _yaw_from_quat(p.orientation))

    def on_scan(self, msg: LaserScan):
        if self._odom is None:
            self.get_logger().warning("no odom yet", throttle_duration_sec=2.0)
            return

        n = len(msg.ranges)
        if self.bridge.configure(n, msg.angle_min, msg.angle_increment):
            self.get_logger().info(f"configured {n} beams")

        r = np.nan_to_num(
            np.asarray(msg.ranges, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
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
        yaw = float(self.bridge.pose[2])
        cy, sy = math.cos(yaw), math.sin(yaw)
        delta = np.array([cy * bx - sy * by, sy * bx + cy * by, dth], dtype=np.float32)
        self._last_odom = odom

        frac = 0.0
        if self.deskew and msg.time_increment > 0.0:
            frac = msg.time_increment * (n - 1) / self._scan_period
            frac = min(max(frac, 0.0), 1.0)

        t0 = time.perf_counter()
        self.bridge.step(r, delta, deskew_frac=frac)
        dt = time.perf_counter() - t0
        self._t_acc += dt
        self._t_max = max(self._t_max, dt)
        self._t_n += 1
        if self._t_n == 400:
            self.get_logger().info(
                f"step avg {self._t_acc / self._t_n * 1e3:.2f} ms "
                f"max {self._t_max * 1e3:.2f} ms"
            )
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

        ox, oy, oth = self._last_odom if self._last_odom else (0.0, 0.0, 0.0)
        dth = _wrap(th - oth)
        c, s = math.cos(dth), math.sin(dth)
        tx = x - (c * ox - s * oy)
        ty = y - (s * ox + c * oy)
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame
        t.transform.translation.x = tx
        t.transform.translation.y = ty
        t.transform.rotation.z = math.sin(dth * 0.5)
        t.transform.rotation.w = math.cos(dth * 0.5)
        self.tf.sendTransform(t)

    def publish_map(self):
        if self.bridge.integrations == self._published_integrations:
            return
        self._published_integrations = self.bridge.integrations

        occ = self.bridge.occupancy()
        grid = int(slam.GRID)
        res = float(slam.RES)
        origin = float(slam.ORIGIN)

        m = OccupancyGrid()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.map_frame
        m.info.map_load_time = m.header.stamp
        m.info.resolution = res
        m.info.width = grid
        m.info.height = grid
        m.info.origin.position.x = origin - 0.5 * res
        m.info.origin.position.y = origin - 0.5 * res
        m.info.origin.orientation.w = 1.0
        m.data = array("b", occ.tobytes())
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
