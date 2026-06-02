# warpSLAM

Lidar-only scan-matching SLAM for F1Tenth, built on NVIDIA Warp.

## Run on the car

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
ros2 launch f1tenth_stack bringup_launch.py
python3 slam_node.py
```

Set `vesc_to_odom/publish_tf: true` in `f1tenth_stack/config/vesc.yaml`.
