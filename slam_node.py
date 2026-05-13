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
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

import slam


def yaw_from_quat(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class SlamNode(Node):
    def __init__(self):
        super().__init__("slam_node")
        self.bridge = slam.Bridge()  # Bridge.__init__ already calls wp.init()

        # --- Parameters -----------------------------------------------------
        self.declare_parameter("scan_topic", "/autodrive/roboracer_1/lidar")
        self.declare_parameter("odom_topic", "/autodrive/roboracer_1/odom")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("map_publish_period_s", 1.0)
        self.declare_parameter("publish_tf", True)

        scan_topic = self.get_parameter("scan_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        map_period = float(self.get_parameter("map_publish_period_s").value)
        self.publish_tf_flag = bool(self.get_parameter("publish_tf").value)

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

        self.pose = np.zeros(3, dtype=np.float32)  # SLAM pose in map frame
        self.latest_odom = None  # latest odom reading
        self.prev_odom_at_scan = None  # odom at the previous scan

        self._map_data = np.full(
            int(slam.GRID_WIDTH) * int(slam.GRID_HEIGHT), -1, dtype=np.int8
        )

        self.get_logger().info(f"slam_node ready  scan={scan_topic}  odom={odom_topic}")

    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose
        self.latest_odom = np.array(
            [p.position.x, p.position.y, yaw_from_quat(p.orientation)],
            dtype=np.float32,
        )

    # --------------------------------------------------------------- scan ---
    def scan_callback(self, msg: LaserScan):
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        np.nan_to_num(ranges, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if self.latest_odom is not None and self.prev_odom_at_scan is not None:
            dx = float(self.latest_odom[0] - self.prev_odom_at_scan[0])
            dy = float(self.latest_odom[1] - self.prev_odom_at_scan[1])
            prev_yaw = float(self.prev_odom_at_scan[2])
            cp, sp = math.cos(prev_yaw), math.sin(prev_yaw)
            bx = dx * cp + dy * sp
            by = -dx * sp + dy * cp
            dyaw = wrap_to_pi(float(self.latest_odom[2]) - prev_yaw)
            slam_yaw = float(self.pose[2])
            cs, ss = math.cos(slam_yaw), math.sin(slam_yaw)
            odom_delta = np.array(
                [bx * cs - by * ss, bx * ss + by * cs, dyaw],
                dtype=np.float32,
            )
        else:
            odom_delta = np.zeros(3, dtype=np.float32)

        self.prev_odom_at_scan = (
            self.latest_odom.copy() if self.latest_odom is not None else None
        )

        self.pose = self.bridge.step(ranges, odom_delta)

        self._publish_pose_and_tf(msg.header.stamp)

    def _publish_pose_and_tf(self, stamp):
        x = float(self.pose[0])
        y = float(self.pose[1])
        yaw = float(self.pose[2])
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)

        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.map_frame
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        self.pose_pub.publish(ps)

        if self.tf_bcast is not None:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.map_frame
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = x
            tf.transform.translation.y = y
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self.tf_bcast.sendTransform(tf)

    # ---------------------------------------------------------- map out ---
    def publish_map(self):
        msg = OccupancyGrid()
        now = self.get_clock().now().to_msg()
        msg.header = Header(stamp=now, frame_id=self.map_frame)
        msg.info.map_load_time = now
        msg.info.resolution = float(slam.RESOLUTION)
        msg.info.width = int(slam.GRID_WIDTH)
        msg.info.height = int(slam.GRID_HEIGHT)
        msg.info.origin.position.x = float(slam.ORIGIN[0])
        msg.info.origin.position.y = float(slam.ORIGIN[1])
        msg.info.origin.orientation.w = 1.0

        lo = self.bridge.logodds.numpy().ravel()
        data = self._map_data
        data.fill(-1)  # default: unknown
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
