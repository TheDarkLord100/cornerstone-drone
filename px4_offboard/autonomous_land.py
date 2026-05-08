import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
    VehicleLandDetected,
)
from enum import Enum, auto
import math

class State(Enum):
    IDLE = auto()
    TAKEOFF = auto()
    FLY_TO_COORDINATE = auto()
    DESCEND = auto()
    LANDED = auto()

class AutonomousLand(Node):
    def __init__(self):
        super().__init__('autonomous_land')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos)

        # Subscribers
        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1',
            self.local_pos_callback, qos)
        self.status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v4',
            self.status_callback, qos)
        self.land_detected_sub = self.create_subscription(
            VehicleLandDetected, '/fmu/out/vehicle_land_detected',
            self.land_detected_callback, qos)

        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz

        # State
        self.state = State.IDLE
        self.local_pos = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.land_detected = False
        self.offboard_setpoint_counter = 0

        # Mission parameters — edit these
        self.takeoff_altitude = -5.0      # NED: negative = up, 5m
        self.target_x = 10.0             # metres in local NED frame
        self.target_y = 5.0
        self.target_yaw = 0.0            # radians, 0 = north
        self.descent_speed = 0.5         # m/s downward (positive in NED = down)
        self.xy_acceptance_radius = 0.3  # metres — how close counts as "arrived"
        self.alt_acceptance = 0.2        # metres — how close to target alt

    def local_pos_callback(self, msg):
        self.local_pos = msg

    def status_callback(self, msg):
        self.vehicle_status = msg

    def land_detected_callback(self, msg):
        self.land_detected = msg.landed

    # ── Helpers ────────────────────────────────────────────────────────────────

    def publish_offboard_control_mode(self, position=False, velocity=False):
        msg = OffboardControlMode()
        msg.position = position
        msg.velocity = velocity
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self.ts()
        self.offboard_pub.publish(msg)

    def publish_position_setpoint(self, x, y, z, yaw):
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = yaw
        msg.timestamp = self.ts()
        self.setpoint_pub.publish(msg)

    def publish_velocity_setpoint(self, vx, vy, vz, yaw):
        msg = TrajectorySetpoint()
        # NaN position tells PX4 to ignore position and use velocity only
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.velocity = [vx, vy, vz]
        msg.yaw = yaw
        msg.timestamp = self.ts()
        self.setpoint_pub.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.ts()
        self.command_pub.publish(msg)

    def arm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info('Offboard mode engaged')

    def ts(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def xy_distance(self):
        dx = self.local_pos.x - self.target_x
        dy = self.local_pos.y - self.target_y
        return math.sqrt(dx*dx + dy*dy)

    def at_altitude(self, target_z):
        return abs(self.local_pos.z - target_z) < self.alt_acceptance

    # ── State machine ──────────────────────────────────────────────────────────

    def timer_callback(self):
        # Always publish heartbeat first
        if self.state in (State.IDLE, State.TAKEOFF, State.FLY_TO_COORDINATE):
            self.publish_offboard_control_mode(position=True)
        elif self.state == State.DESCEND:
            self.publish_offboard_control_mode(velocity=True)

        if self.state == State.IDLE:
            # Send setpoints before engaging offboard — PX4 requires >10 cycles
            self.publish_position_setpoint(
                self.local_pos.x, self.local_pos.y,
                self.takeoff_altitude, self.target_yaw)
            self.offboard_setpoint_counter += 1

            if self.offboard_setpoint_counter == 10:
                self.engage_offboard_mode()
                self.arm()
                self.state = State.TAKEOFF
                self.get_logger().info('→ TAKEOFF')

        elif self.state == State.TAKEOFF:
            self.publish_position_setpoint(
                self.local_pos.x, self.local_pos.y,
                self.takeoff_altitude, self.target_yaw)

            if self.at_altitude(self.takeoff_altitude):
                self.state = State.FLY_TO_COORDINATE
                self.get_logger().info(
                    f'→ FLY_TO_COORDINATE ({self.target_x}, {self.target_y})')

        elif self.state == State.FLY_TO_COORDINATE:
            self.publish_position_setpoint(
                self.target_x, self.target_y,
                self.takeoff_altitude, self.target_yaw)

            dist = self.xy_distance()
            self.get_logger().info(f'Distance to target: {dist:.2f}m', throttle_duration_sec=1.0)

            if dist < self.xy_acceptance_radius:
                self.state = State.DESCEND
                self.get_logger().info('→ DESCEND')

        elif self.state == State.DESCEND:
            # Hold x,y with zero lateral velocity, descend at constant rate
            self.publish_velocity_setpoint(
                0.0, 0.0, self.descent_speed, self.target_yaw)

            self.get_logger().info(
                f'Altitude: {-self.local_pos.z:.2f}m', throttle_duration_sec=1.0)

            if self.land_detected:
                self.state = State.LANDED
                self.get_logger().info('→ LANDED')
                self.disarm()

        elif self.state == State.LANDED:
            # Nothing to do — mission complete
            pass


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousLand()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()