```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
ros2 launch disparity_extender disparity_extender.launch.py
ros2 launch f1tenth_stack bringup_launch.py
python3 slam_node.py
```
configure `vesc_to_odom/publish_tf: true` in `f1tenth_stack/config/vesc.yaml`
