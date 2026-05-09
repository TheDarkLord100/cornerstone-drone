import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import random
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    OffboardControlMode, TrajectorySetpoint, VehicleCommand,
    VehicleLocalPosition, VehicleStatus, VehicleLandDetected, VehicleAttitude,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from enum import Enum, auto
import math

from vision.patch_scorer import PatchScorer
from vision.obstacle_detection import ObstacleDetection
from vision.visualizer import Visualizer
from vision.terrain_map import TerrainMap


class State(Enum):
    IDLE         = auto()
    TAKEOFF      = auto()
    FLY_TO_COORD = auto()
    SWEEP        = auto()   # 3×3 lawnmower sweep over landing zone
    FLY_TO_SAFE  = auto()
    DESCEND      = auto()
    SPIRAL       = auto()   # expanding spiral search when no safe zone found
    RTL          = auto()
    LANDED       = auto()


def _spiral_offsets(step: float):
    """
    Yields (dx, dy) offsets from a centre in an outward square spiral.
    Pattern grows one step per two legs: right, up, left×2, down×2, right×3 ...
    """
    x, y = 0.0, 0.0
    dx, dy = step, 0.0
    segment_len = 1
    while True:
        for _ in range(2):
            for _ in range(segment_len):
                x += dx
                y += dy
                yield x, y
            dx, dy = -dy, dx        # rotate direction 90°
        segment_len += 1


def _sweep_waypoints(cx: float, cy: float, spacing: float = 4.0):
    """
    Generate 9 waypoints in a 3×3 grid centred on (cx, cy).
    Row order: top-left → top-right, middle-left → middle-right,
               bottom-left → bottom-right  (lawnmower, alternating direction).

    spacing : metres between adjacent waypoints
    """
    offsets = [-spacing, 0.0, spacing]
    waypoints = []
    for i, dy in enumerate(offsets):
        row = [spacing, 0.0, -spacing] if i % 2 == 0 else [-spacing, 0.0, spacing]
        for dx in row:
            waypoints.append((cx + dx, cy + dy))
    return waypoints


class SafeLand(Node):
    def __init__(self):
        super().__init__('safe_land')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos)

        self.debug_pub = self.create_publisher(
            Image, '/safe_land/debug_image', 10)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1', self.pos_cb, qos)
        self.create_subscription(VehicleStatus,
            '/fmu/out/vehicle_status_v4', self.status_cb, qos)
        self.create_subscription(VehicleLandDetected,
            '/fmu/out/vehicle_land_detected', self.land_cb, qos)
        self.create_subscription(VehicleAttitude,
            '/fmu/out/vehicle_attitude', self.attitude_cb, qos)
        # Depth camera only — feeds both terrain mapping and visual panel
        self.create_subscription(Image,
            '/depth_camera', self.depth_image_cb, img_qos)

        self.timer = self.create_timer(0.1, self.timer_callback)

        # ── ROS state ─────────────────────────────────────────────────────────
        self.state          = State.IDLE
        self.local_pos      = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.attitude       = VehicleAttitude()
        self.land_detected  = False
        self.latest_image   = None
        self.latest_depth   = None
        self.offboard_counter = 0

        # ── Camera intrinsics (from camera_info) ──────────────────────────────
        # Camera intrinsics — derived from depth measurement (HFOV ≈ 44.7°, 640×480)
        self.fx = 777.59
        self.fy = 777.59
        self.cx = 320.0
        self.cy = 240.0

        # ── Mission parameters ────────────────────────────────────────────────
        self.takeoff_alt   = -5.0
        self.scan_alt      = -3.0   # within depth sensor range (~3m max)
        self.target_x      = random.uniform(-100.0, 100.0)
        self.target_y      = random.uniform(-100.0, 100.0)
        self.target_yaw    = 0.0
        self.descent_speed = 0.5
        self.xy_radius     = 0.3
        self.alt_tol       = 0.3

        # ── Scan accumulation ─────────────────────────────────────────────────
        self.frames_per_waypoint = 10   # depth + score frames to collect per waypoint
        self.scan_counter  = 0
        self.scan_results  = []
        self.score_accum   = None
        self.safe_x        = None
        self.safe_y        = None

        # ── Sweep (3×3 lawnmower) ─────────────────────────────────────────────
        self.sweep_spacing    = 4.0    # metres between waypoints
        self.sweep_waypoints  = []     # populated when sweep begins
        self.sweep_idx        = 0      # current waypoint index
        self.sweep_arrived    = False  # True once drone is within xy_radius of wp
        self.sweep_wp_counter = 0      # frames collected at current waypoint

        # ── Spiral search ─────────────────────────────────────────────────────
        self.spiral_step         = 3.0
        self.max_spiral_attempts = 5
        self.spiral_attempt      = 0
        self._spiral_gen         = _spiral_offsets(self.spiral_step)
        self.spiral_wx           = self.target_x
        self.spiral_wy           = self.target_y
        self.spiral_arrived      = False

        # ── Vision pipeline ───────────────────────────────────────────────────
        self.bridge       = CvBridge()
        self.patch_scorer = PatchScorer()
        self.obstacle_det = ObstacleDetection()

        # TerrainMap — instantiated in _reset_sweep once waypoints are known
        self._terrain_map = None
        self._map_timer   = self.create_timer(0.5, self._publish_map)  # 2 Hz

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def pos_cb(self, msg):      self.local_pos      = msg
    def status_cb(self, msg):   self.vehicle_status = msg
    def attitude_cb(self, msg): self.attitude       = msg
    def land_cb(self, msg):     self.land_detected  = msg.landed

    def depth_image_cb(self, msg):
        """Single callback for /depth_camera."""
        self.latest_depth = msg
        self.latest_image = msg

    # ── PX4 helpers ────────────────────────────────────────────────────────────

    def ts(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def pub_offboard(self, position=False, velocity=False):
        msg = OffboardControlMode()
        msg.position     = position
        msg.velocity     = velocity
        msg.acceleration = False
        msg.attitude     = False
        msg.body_rate    = False
        msg.timestamp    = self.ts()
        self.offboard_pub.publish(msg)

    def pub_position_sp(self, x, y, z, yaw):
        msg = TrajectorySetpoint()
        msg.position  = [x, y, z]
        msg.yaw       = yaw
        msg.timestamp = self.ts()
        self.setpoint_pub.publish(msg)

    def pub_velocity_sp(self, vx, vy, vz, yaw):
        msg = TrajectorySetpoint()
        msg.position  = [float('nan')] * 3
        msg.velocity  = [vx, vy, vz]
        msg.yaw       = yaw
        msg.timestamp = self.ts()
        self.setpoint_pub.publish(msg)

    def pub_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command          = command
        msg.param1           = param1
        msg.param2           = param2
        msg.target_system    = 1
        msg.target_component = 1
        msg.source_system    = 1
        msg.source_component = 1
        msg.from_external    = True
        msg.timestamp        = self.ts()
        self.command_pub.publish(msg)

    def arm(self):
        self.pub_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Armed')

    def disarm(self):
        self.pub_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarmed')

    def engage_offboard(self):
        self.pub_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info('Offboard mode engaged')

    def rtl(self):
        self.pub_command(VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)
        self.get_logger().warn('RTL commanded — all spiral attempts exhausted')

    # ── Geometry ───────────────────────────────────────────────────────────────

    def xy_dist(self, tx, ty):
        return math.sqrt((self.local_pos.x - tx)**2 + (self.local_pos.y - ty)**2)

    def at_alt(self, target_z):
        return abs(self.local_pos.z - target_z) < self.alt_tol

    def get_yaw(self):
        q = self.attitude.q
        w, x, y, z = q[0], q[1], q[2], q[3]
        return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    def pixel_to_world(self, u, v):
        """Project pixel (u=col, v=row) to NED world frame using altitude + yaw."""
        h   = -self.local_pos.z
        x_c = (v - self.cy) * h / self.fy   # row  → camera-forward
        y_c = (u - self.cx) * h / self.fx   # col  → camera-right
        yaw = self.get_yaw()
        wx  = self.local_pos.x + x_c * math.cos(yaw) - y_c * math.sin(yaw)
        wy  = self.local_pos.y + x_c * math.sin(yaw) + y_c * math.cos(yaw)
        return wx, wy

    # ── Vision ─────────────────────────────────────────────────────────────────

    def analyze_frame(self, cv_image):
        """
        Score patches using depth flatness as primary signal.
        Accepts a ROS sensor_msgs/Image (32FC1 depth).
        Returns (best_u, best_v, best_score) or None if degenerate.

        Debug image is side-by-side:
          LEFT  — depth image false-coloured (near=red, far=blue) + best patch marker
          RIGHT — top-down TerrainMap occupancy render
        """
        # ── Decode depth ROS msg → float32 numpy array ────────────────────────
        depth_m = np.frombuffer(bytes(cv_image.data), dtype=np.float32).reshape(
            cv_image.height, cv_image.width).copy()
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        img_h, img_w = depth_m.shape

        # ── False-colour depth for LEFT display panel ──────────────────────────
        valid_mask = (depth_m > 0.1) & (depth_m < 10.0)
        depth_norm = np.zeros_like(depth_m)
        if valid_mask.any():
            d_min = depth_m[valid_mask].min()
            d_max = depth_m[valid_mask].max()
            if d_max > d_min:
                depth_norm[valid_mask] = (
                    (depth_m[valid_mask] - d_min) / (d_max - d_min))
        depth_u8 = (depth_norm * 255).astype(np.uint8)
        left_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        left_bgr[~valid_mask] = (20, 20, 20)  # dark = no return

        # ── Scoring: per-patch depth variance (low std dev = flat = safe) ──────
        patch_size = self.patch_scorer.patch_size
        ph = img_h // patch_size
        pw = img_w // patch_size
        scores = np.zeros((ph, pw), dtype=np.float32)
        for pr in range(ph):
            for pc in range(pw):
                r0, r1 = pr * patch_size, (pr + 1) * patch_size
                c0, c1 = pc * patch_size, (pc + 1) * patch_size
                patch = depth_m[r0:r1, c0:c1]
                v = patch[patch > 0.1]
                if len(v) > 4:
                    std = float(np.std(v))
                    scores[pr, pc] = max(0.0, 1.0 - std / 0.035)
                else:
                    scores[pr, pc] = 0.0   # no data = unknown, don't land here

        ph, pw = scores.shape

        if float(np.max(scores)) < 0.05:
            self.get_logger().warn('analyze_frame: all patches near-zero, skipping')
            return None

        # Accumulate score grid
        if self.score_accum is None:
            self.score_accum = np.zeros((ph, pw), dtype=np.float32)
        self.score_accum += scores.astype(np.float32)

        flat_idx         = np.argmax(self.score_accum)
        best_pr, best_pc = np.unravel_index(flat_idx, self.score_accum.shape)
        best_score       = float(scores[best_pr, best_pc])

        patch_h = img_h // ph
        patch_w = img_w // pw
        best_u  = int(best_pc * patch_w + patch_w // 2)
        best_v  = int(best_pr * patch_h + patch_h // 2)

        # ── LEFT panel: false-colour depth + YOLO boxes + best patch marker ──
        left = left_bgr.copy()

        if hasattr(self.obstacle_det, 'obj') and self.obstacle_det.obj:
            for cls_name, info in self.obstacle_det.obj.items():
                x1, y1, x2, y2 = info['cords']
                cv2.rectangle(left, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(left, f'{cls_name} {info["conf"]:.2f}',
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        cv2.circle(left, (best_u, best_v), 14, (255, 255, 0), 3)
        cv2.putText(left, f'BEST {best_score:.2f}',
                    (best_u + 16, best_v), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(left, 'DEPTH', (10, img_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # ── RIGHT panel: top-down TerrainMap render ────────────────────────────
        right = self._render_topdown(img_h, img_w)

        # ── Combine + HUD ──────────────────────────────────────────────────────
        debug_img = np.concatenate([left, right], axis=1)   # side by side

        wp_total = len(self.sweep_waypoints) if self.sweep_waypoints else 1
        cov = self._terrain_map.coverage_pct() if self._terrain_map else 0.0
        hud_lines = [
            f'SWEEP  wp {self.sweep_idx + 1}/{wp_total}  '
            f'frame {self.sweep_wp_counter + 1}/{self.frames_per_waypoint}',
            f'Altitude: {-self.local_pos.z:.1f} m    Coverage: {cov:.0f}%',
        ]
        for i, line in enumerate(hud_lines):
            cv2.putText(debug_img, line, (10, 22 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        frame_num = self.sweep_idx * self.frames_per_waypoint + self.sweep_wp_counter
        cv2.imwrite(f'/tmp/safe_land_frame_{frame_num:03d}.png', debug_img)

        ros_img = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
        ros_img.header.stamp    = self.get_clock().now().to_msg()
        ros_img.header.frame_id = 'camera_link'
        self.debug_pub.publish(ros_img)

        self.get_logger().info(
            f'wp{self.sweep_idx + 1} f{self.sweep_wp_counter + 1}: '
            f'pixel=({best_u},{best_v}) score={best_score:.3f}  cov={cov:.0f}%')

        return best_u, best_v, best_score

    def _render_topdown(self, img_h: int, img_w: int) -> np.ndarray:
        """
        Render the TerrainMap as a top-down BGR image of size (img_h, img_w).

        Colour scheme:
          grey        — unknown / no depth data yet
          green       — flat / safe  (flatness > 0.6)
          yellow      — marginal     (flatness 0.3–0.6)
          red         — rough/unsafe (flatness < 0.3)

        Overlays:
          white cross  — drone current position
          cyan circle  — committed safe zone (if set)
          white grid   — sweep waypoints
        """
        canvas = np.full((img_h, img_w, 3), 40, dtype=np.uint8)

        if self._terrain_map is None or not self.sweep_waypoints:
            cv2.putText(canvas, 'Awaiting sweep...', (20, img_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            return canvas

        # Compute display bounds from waypoints
        tm = self._terrain_map
        x_min, x_max = tm.map_x_min, tm.map_x_max
        y_min, y_max = tm.map_y_min, tm.map_y_max

        def world_to_px(wx, wy):
            col = int((wx - x_min) / (x_max - x_min) * img_w)
            row = int((wy - y_min) / (y_max - y_min) * img_h)
            return col, row

        # Draw captured snapshot footprints as coloured rectangles
        half_w_px = int(tm.half_w_m / (x_max - x_min) * img_w)
        half_h_px = int(tm.half_h_m / (y_max - y_min) * img_h)

        for i, (wpx, wpy) in enumerate(self.sweep_waypoints):
            col, row = world_to_px(wpx, wpy)
            if i in tm.snapshots:
                # Show footprint as filled semi-transparent rect
                overlay = canvas.copy()
                cv2.rectangle(overlay,
                              (col - half_w_px, row - half_h_px),
                              (col + half_w_px, row + half_h_px),
                              (0, 120, 0), -1)
                cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
                cv2.rectangle(canvas,
                              (col - half_w_px, row - half_h_px),
                              (col + half_w_px, row + half_h_px),
                              (0, 200, 0), 1)
                cv2.putText(canvas, str(i+1), (col - 6, row + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            else:
                cv2.circle(canvas, (col, row), 8, (100, 100, 100), 1)
                cv2.putText(canvas, str(i+1), (col - 6, row + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        # Drone position
        dpx = world_to_px(self.local_pos.x, self.local_pos.y)
        if 0 <= dpx[0] < img_w and 0 <= dpx[1] < img_h:
            cv2.drawMarker(canvas, dpx, (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

        # Safe zone
        if self.safe_x is not None:
            spx = world_to_px(self.safe_x, self.safe_y)
            if 0 <= spx[0] < img_w and 0 <= spx[1] < img_h:
                cv2.circle(canvas, spx, 10, (255, 255, 0), -1)

        snaps = len(tm.snapshots)
        total = len(self.sweep_waypoints)
        cv2.putText(canvas, f'SWEEP MAP  {snaps}/{total} captured',
                    (10, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1)

        return canvas

    # ── Sweep helpers ──────────────────────────────────────────────────────────

    def _reset_sweep(self):
        """Reset all accumulated scan state and build the 3×3 waypoint list."""
        self.scan_counter     = 0
        self.scan_results     = []
        self.score_accum      = None
        self.sweep_waypoints  = _sweep_waypoints(
            self.target_x, self.target_y, self.sweep_spacing)
        self.sweep_idx        = 0
        self.sweep_arrived    = False
        self.sweep_wp_counter = 0
        # Instantiate TerrainMap now that waypoints are known
        self._terrain_map = TerrainMap(self, self.sweep_waypoints, self.scan_alt)
        self.get_logger().info(
            f'Sweep: {len(self.sweep_waypoints)} waypoints, '
            f'{self.frames_per_waypoint} frames each')

    def _reset_scan(self):
        """Lightweight reset used by spiral search (score accum only)."""
        self.scan_counter     = 0
        self.scan_results     = []
        self.score_accum      = None
        self.sweep_wp_counter = 0

    def _run_sweep_step(self):
        """
        Drive the sweep state machine one timer tick.

        Returns:
          'flying'   — still flying to current waypoint
          'scanning' — at waypoint, collecting frames
          'done'     — all waypoints scanned, results ready
        """
        if self.sweep_idx >= len(self.sweep_waypoints):
            return 'done'

        wx, wy = self.sweep_waypoints[self.sweep_idx]
        dist   = self.xy_dist(wx, wy)

        # Always publish position setpoint to current waypoint
        self.pub_position_sp(wx, wy, self.scan_alt, self.target_yaw)

        if dist > self.xy_radius or not self.at_alt(self.scan_alt):
            if self.sweep_arrived:
                self.sweep_arrived    = False
                self.sweep_wp_counter = 0
            return 'flying'

        # Arrived at waypoint
        if not self.sweep_arrived:
            self.sweep_arrived = True
            self.sweep_wp_counter = 0
            self.get_logger().info(
                f'Sweep wp {self.sweep_idx + 1}/{len(self.sweep_waypoints)} '
                f'reached ({wx:.1f}, {wy:.1f})')
            # Capture terrain snapshot immediately on arrival
            if self.latest_image is not None and self._terrain_map is not None:
                self._terrain_map.capture_snapshot(
                    self.latest_image,
                    self.local_pos.x, self.local_pos.y,
                    self.local_pos.z, self.get_yaw(),
                    self.sweep_idx)
                self.latest_image = None

        # Collect scoring frames
        if self.latest_image is not None:
            depth_msg = self.latest_image
            self.latest_image = None

            result = self.analyze_frame(depth_msg)
            if result is not None:
                u, v, score = result
                world_x, world_y = self.pixel_to_world(u, v)
                self.scan_results.append((world_x, world_y, score))
                self.scan_counter += 1
                self.sweep_wp_counter += 1
                self.get_logger().info(
                    f'  wp{self.sweep_idx + 1} frame {self.sweep_wp_counter}/'
                    f'{self.frames_per_waypoint}: '
                    f'pixel=({u},{v}) → world=({world_x:.2f},{world_y:.2f}) '
                    f'score={score:.3f}')

        # Advance to next waypoint after collecting enough frames
        if self.sweep_wp_counter >= self.frames_per_waypoint:
            self.sweep_idx     += 1
            self.sweep_arrived  = False
            self.sweep_wp_counter = 0
            if self.sweep_idx >= len(self.sweep_waypoints):
                return 'done'

        return 'scanning'

    def _commit_safe_zone(self):
        """Score-weighted average of all accumulated world coordinates."""
        if not self.scan_results:
            return False
        total       = sum(r[2] for r in self.scan_results)
        self.safe_x = sum(r[0] * r[2] for r in self.scan_results) / total
        self.safe_y = sum(r[1] * r[2] for r in self.scan_results) / total
        self.get_logger().info(
            f'Safe zone committed: ({self.safe_x:.2f}, {self.safe_y:.2f})')
        return True

    def _next_spiral_waypoint(self):
        """Advance spiral generator and set new waypoint target."""
        dx, dy           = next(self._spiral_gen)
        self.spiral_wx   = self.target_x + dx
        self.spiral_wy   = self.target_y + dy
        self.spiral_arrived = False
        self.get_logger().info(
            f'Spiral attempt {self.spiral_attempt + 1}/{self.max_spiral_attempts} '
            f'→ ({self.spiral_wx:.1f}, {self.spiral_wy:.1f})')

    def _publish_map(self):
        if self._terrain_map is not None:
            cov = self._terrain_map.coverage_pct()
            self.get_logger().info(
                f'Snapshots: {len(self._terrain_map.snapshots)}/'
                f'{len(self.sweep_waypoints) if self.sweep_waypoints else 9}',
                throttle_duration_sec=5.0)

    # ── State machine ──────────────────────────────────────────────────────────

    def timer_callback(self):
        # Heartbeat — must be published every cycle or PX4 exits offboard mode
        if self.state in (State.IDLE, State.TAKEOFF, State.FLY_TO_COORD,
                          State.SWEEP, State.FLY_TO_SAFE, State.SPIRAL):
            self.pub_offboard(position=True)
        elif self.state == State.DESCEND:
            self.pub_offboard(velocity=True)

        # ── IDLE ──────────────────────────────────────────────────────────────
        if self.state == State.IDLE:
            self.pub_position_sp(
                self.local_pos.x, self.local_pos.y,
                self.takeoff_alt, self.target_yaw)
            self.offboard_counter += 1
            if self.offboard_counter == 10:
                self.engage_offboard()
                self.arm()
                self.state = State.TAKEOFF
                self.get_logger().info('→ TAKEOFF')

        # ── TAKEOFF ───────────────────────────────────────────────────────────
        elif self.state == State.TAKEOFF:
            self.pub_position_sp(
                self.local_pos.x, self.local_pos.y,
                self.takeoff_alt, self.target_yaw)
            if self.at_alt(self.takeoff_alt):
                self.state = State.FLY_TO_COORD
                self.get_logger().info(
                    f'→ FLY_TO_COORD ({self.target_x:.1f}, {self.target_y:.1f})')

        # ── FLY_TO_COORD ──────────────────────────────────────────────────────
        elif self.state == State.FLY_TO_COORD:
            self.pub_position_sp(
                self.target_x, self.target_y,
                self.scan_alt, self.target_yaw)
            dist = self.xy_dist(self.target_x, self.target_y)
            self.get_logger().info(
                f'Dist to target: {dist:.2f} m', throttle_duration_sec=1.0)
            if dist < self.xy_radius and self.at_alt(self.scan_alt):
                self._reset_sweep()
                self.state = State.SWEEP
                self.get_logger().info(
                    f'→ SWEEP ({len(self.sweep_waypoints)} waypoints, '
                    f'{self.frames_per_waypoint} frames each)')

        # ── SWEEP ─────────────────────────────────────────────────────────────
        elif self.state == State.SWEEP:
            status = self._run_sweep_step()
            if status == 'done':
                # Stitch and save terrain map
                if self._terrain_map is not None:
                    canvas = self._terrain_map.stitch()
                    stamp  = self.get_clock().now().to_msg()
                    self._terrain_map.publish_map_image(self, canvas, stamp)
                if self._commit_safe_zone():
                    self.state = State.FLY_TO_SAFE
                else:
                    self.get_logger().warn('SWEEP: no valid frames — spiral search')
                    self._next_spiral_waypoint()
                    self.state = State.SPIRAL

        # ── FLY_TO_SAFE ───────────────────────────────────────────────────────
        elif self.state == State.FLY_TO_SAFE:
            self.pub_position_sp(
                self.safe_x, self.safe_y,
                self.scan_alt, self.target_yaw)
            dist = self.xy_dist(self.safe_x, self.safe_y)
            self.get_logger().info(
                f'Dist to safe spot: {dist:.2f} m', throttle_duration_sec=1.0)
            if dist < self.xy_radius:
                self.state = State.DESCEND
                self.get_logger().info('→ DESCEND')

        # ── DESCEND ───────────────────────────────────────────────────────────
        elif self.state == State.DESCEND:
            self.pub_velocity_sp(0.0, 0.0, self.descent_speed, self.target_yaw)
            self.get_logger().info(
                f'Altitude: {-self.local_pos.z:.2f} m', throttle_duration_sec=1.0)
            if self.land_detected:
                self.disarm()
                self.state = State.LANDED
                self.get_logger().info('→ LANDED')

        # ── SPIRAL ────────────────────────────────────────────────────────────
        elif self.state == State.SPIRAL:
            self.pub_position_sp(
                self.spiral_wx, self.spiral_wy,
                self.scan_alt, self.target_yaw)
            dist = self.xy_dist(self.spiral_wx, self.spiral_wy)
            self.get_logger().info(
                f'Spiral [{self.spiral_attempt + 1}/{self.max_spiral_attempts}] '
                f'dist: {dist:.2f} m', throttle_duration_sec=1.0)

            if dist < self.xy_radius and self.at_alt(self.scan_alt):
                if not self.spiral_arrived:
                    self.spiral_arrived   = True
                    self.sweep_wp_counter = 0
                    self._reset_scan()
                    self.get_logger().info('Spiral waypoint reached — scanning...')

                # Collect one frame
                if self.latest_image is not None:
                    depth_msg = self.latest_image
                    self.latest_image = None
                    result = self.analyze_frame(depth_msg)
                    if result is not None:
                        u, v, score = result
                        wx, wy = self.pixel_to_world(u, v)
                        self.scan_results.append((wx, wy, score))
                        self.scan_counter     += 1
                        self.sweep_wp_counter += 1

                if self.sweep_wp_counter >= self.frames_per_waypoint:
                    if self._commit_safe_zone():
                        self.state = State.FLY_TO_SAFE
                    else:
                        self.spiral_attempt += 1
                        if self.spiral_attempt >= self.max_spiral_attempts:
                            self.get_logger().error(
                                'Max spiral attempts reached — RTL')
                            self.rtl()
                            self.state = State.RTL
                        else:
                            self._next_spiral_waypoint()

        # ── RTL ───────────────────────────────────────────────────────────────
        elif self.state == State.RTL:
            pass   # PX4 handles RTL autonomously

        # ── LANDED ────────────────────────────────────────────────────────────
        elif self.state == State.LANDED:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = SafeLand()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()