import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node

# ── Config ────────────────────────────────────────────────────────────────────
PX4_AUTOPILOT_DIR = os.path.expanduser('~/PX4-Autopilot')
WORLD             = 'mars'
MODEL             = 'gz_x500_depth_down'
SPAWN_POSE        = '-20,-50,5,0,0,0'
DDS_PORT          = '8888'
QGC_PATH          = os.path.expanduser('~/Downloads/QGroundControl-x86_64.AppImage')
# ─────────────────────────────────────────────────────────────────────────────

def generate_launch_description():

    # 1. PX4 SITL + Gazebo
    px4 = TimerAction(
        period=10.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'bash', '-c',
                    f'cd {PX4_AUTOPILOT_DIR} && '
                    f'PX4_GZ_WORLD={WORLD} '
                    f'PX4_GZ_MODEL_POSE="{SPAWN_POSE}" '
                    f'make px4_sitl {MODEL}'
                ],
                output='screen',
                name='px4_sitl'
            )
        ]
    )

    # 2. micro-ROS agent — wait for Gazebo to finish loading
    dds_agent = TimerAction(
        period=30.0,
        actions=[
            ExecuteProcess(
                cmd=['/snap/bin/micro-ros-agent', 'udp4', '--port', DDS_PORT],
                output='screen',
                name='dds_agent'
            )
        ]
    )

    # 3. ROS-GZ bridge — depth camera only (IMX214 RGB removed)
    bridge = TimerAction(
        period=50.0,
        actions=[
            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                name='gz_bridge',
                output='screen',
                arguments=[
                    '/depth_camera@sensor_msgs/msg/Image[gz.msgs.Image',
                    '/depth_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
                ],
            )
        ]
    )

    # 4. QGC
    qgc = ExecuteProcess(
        cmd=[QGC_PATH],
        output='screen',
        name='qgc'
    )

    return LaunchDescription([
        qgc,
        px4,
        dds_agent,
        bridge,
    ])