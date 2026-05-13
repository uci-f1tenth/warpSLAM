import rclpy
from rclpy.node import Node
import numpy as np
import warp as wp

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

import main as slam


class SlamNode(Node):
    def __init__(self):
        super().__init__("slam_node")

        wp.init()
        self.bridge = slam.Bridge()

        self.scan_sub = self.create_subscription(
            LaserScan, "/autodrive/roboracer_1/lidar", self.scan_callback, 1
        )

        self.map_pub = self.create_publisher(OccupancyGrid, "/map", 1)
        self.map_timer = self.create_timer(1.0, self.publish_map)

        self.pose = np.zeros(3, dtype=np.float32)
        self.prev_pose = np.zeros(3, dtype=np.float32)
        self.first_scan = True

        self.get_logger().info("slam_node ready")

    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        np.nan_to_num(ranges, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if self.first_scan:
            odom_delta = np.zeros(3, dtype=np.float32)
            self.first_scan = False
        else:
            odom_delta = self.pose - self.prev_pose

        self.prev_pose = self.pose.copy()
        self.pose = self.bridge.step(ranges, odom_delta)

    def publish_map(self):
        msg = OccupancyGrid()
        msg.header = Header(
            stamp=self.get_clock().now().to_msg(), frame_id="map"
        )
        msg.info.map_load_time = self.get_clock().now().to_msg()
        msg.info.resolution = float(slam.RESOLUTION)
        msg.info.width = int(slam.GRID_WIDTH)
        msg.info.height = int(slam.GRID_HEIGHT)
        msg.info.origin.position.x = float(slam.ORIGIN[0])
        msg.info.origin.position.y = float(slam.ORIGIN[1])

        lo = self.bridge.logodds.numpy()
        data = np.full(msg.info.width * msg.info.height, -1, dtype=np.int8)

        # logistic: p = 1 / (1 + exp(-lo))
        known = np.abs(lo) > 0.1
        lo_known = lo[known]
        p = 1.0 / (1.0 + np.exp(-np.clip(lo_known, -10.0, 10.0)))
        data[known] = np.clip((p * 100).astype(np.int8), 0, 100)

        msg.data = data.tolist()
        self.map_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SlamNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
