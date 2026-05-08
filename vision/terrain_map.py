"""
terrain_map.py  —  Simple waypoint-snapshot terrain mapper
===========================================================
At each sweep waypoint the node calls capture_snapshot().
After all waypoints are visited, stitch() produces a single
top-down BGR image showing the terrain across the full sweep area.

No accumulation, no variance, no grid cells.
Each snapshot is one depth frame projected to a top-down patch
and placed at the correct canvas position using known drone pose.
"""

import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid
import struct


class TerrainMap:
    """
    Captures one depth snapshot per sweep waypoint and stitches
    them into a top-down map.

    Usage:
        tm = TerrainMap(node, waypoints, scan_alt)
        ...
        tm.capture_snapshot(depth_msg, drone_x, drone_y, drone_z, drone_yaw, wp_idx)
        ...
        map_img = tm.stitch()   # call after all waypoints done
        tm.publish_occupancy(node, stamp, map_img)
    """

    # Depth camera intrinsics (derived from measurement at 8m altitude)
    FX = 777.59
    FY = 777.59
    CX = 320.0
    CY = 240.0

    # Map render resolution: metres per pixel in the stitched output
    MAP_RESOLUTION = 0.05   # 5cm/px — fine enough to see terrain features

    def __init__(self, node: Node, waypoints: list, scan_alt: float):
        """
        waypoints : list of (wx, wy) NED world coords — the 9 sweep stops
        scan_alt  : scan altitude in NED (negative, e.g. -3.0)
        """
        self._log       = node.get_logger()
        self.waypoints  = waypoints
        self.altitude   = -scan_alt          # positive metres above ground
        self.snapshots  = {}                 # wp_idx → (depth_m, drone_x, drone_y, drone_yaw)

        # Compute FOV footprint at scan altitude
        self.half_w_m = self.altitude * self.CX / self.FX   # half-width in metres
        self.half_h_m = self.altitude * self.CY / self.FY   # half-height in metres

        # Compute canvas bounds from waypoints + footprint margin
        xs = [wp[0] for wp in waypoints]
        ys = [wp[1] for wp in waypoints]
        margin = max(self.half_w_m, self.half_h_m) * 1.2
        self.map_x_min = min(xs) - margin
        self.map_x_max = max(xs) + margin
        self.map_y_min = min(ys) - margin
        self.map_y_max = max(ys) + margin

        self.map_w_px = int((self.map_x_max - self.map_x_min) / self.MAP_RESOLUTION)
        self.map_h_px = int((self.map_y_max - self.map_y_min) / self.MAP_RESOLUTION)

        # ROS publisher for occupancy grid (optional, for RViz)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self._grid_pub = node.create_publisher(
            OccupancyGrid, '/safe_land/occupancy_map', qos)
        self._cloud_pub = node.create_publisher(
            PointCloud2, '/safe_land/map_cloud', 10)

        self._log.info(
            f'TerrainMap: {len(waypoints)} waypoints, '
            f'FOV footprint {self.half_w_m*2:.1f}×{self.half_h_m*2:.1f}m, '
            f'canvas {self.map_w_px}×{self.map_h_px}px '
            f'({(self.map_x_max-self.map_x_min):.1f}×'
            f'{(self.map_y_max-self.map_y_min):.1f}m)')

    # ── Coordinate helpers ─────────────────────────────────────────────────────

    def _world_to_canvas(self, wx, wy):
        """NED world (wx, wy) → canvas pixel (col, row)."""
        col = int((wx - self.map_x_min) / self.MAP_RESOLUTION)
        row = int((wy - self.map_y_min) / self.MAP_RESOLUTION)
        return col, row

    # ── Snapshot capture ───────────────────────────────────────────────────────

    def capture_snapshot(self, depth_msg: Image,
                         drone_x: float, drone_y: float,
                         drone_z: float, drone_yaw: float,
                         wp_idx: int):
        """
        Store one depth frame for waypoint wp_idx.
        Call once when the drone has settled at the waypoint.
        """
        depth_m = np.frombuffer(bytes(depth_msg.data), dtype=np.float32).reshape(
            depth_msg.height, depth_msg.width).copy()
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)

        self.snapshots[wp_idx] = (depth_m, drone_x, drone_y, drone_yaw)
        self._log.info(
            f'TerrainMap: snapshot {wp_idx+1}/{len(self.waypoints)} '
            f'captured at ({drone_x:.1f},{drone_y:.1f})')

    def coverage_fraction(self):
        return len(self.snapshots) / max(len(self.waypoints), 1)

    # ── Stitching ──────────────────────────────────────────────────────────────

    def stitch(self) -> np.ndarray:
        """
        Project all captured snapshots into a single top-down BGR canvas.

        Depth values are normalised to a TURBO colourmap:
          warm (red/yellow) = close terrain (rocks/bumps)
          cool (blue/purple) = far terrain (flat/lower)

        Returns a BGR image of size (map_h_px, map_w_px).
        """
        # Canvas: start grey (unknown)
        canvas     = np.full((self.map_h_px, self.map_w_px, 3), 50, dtype=np.uint8)
        depth_acc  = np.zeros((self.map_h_px, self.map_w_px), dtype=np.float32)
        count_acc  = np.zeros((self.map_h_px, self.map_w_px), dtype=np.int32)

        for wp_idx, (depth_m, drone_x, drone_y, drone_yaw) in self.snapshots.items():
            h, w = depth_m.shape
            altitude = -(-self.altitude)   # positive

            # Build pixel grids
            u_idx = np.arange(w, dtype=np.float32)
            v_idx = np.arange(h, dtype=np.float32)
            uu, vv = np.meshgrid(u_idx, v_idx)

            valid = (depth_m > 0.1) & (depth_m < altitude * 3.0)

            z  = depth_m[valid]
            uf = uu[valid]
            vf = vv[valid]

            # Normalised ray directions
            xn = (uf - self.CX) / self.FX
            yn = (vf - self.CY) / self.FY
            ray_len = np.sqrt(1.0 + xn**2 + yn**2)

            # Vertical depth (geometry-corrected)
            z_vert = z / ray_len

            # World XY via yaw rotation
            x_cam = xn * z
            y_cam = yn * z
            cos_y = math.cos(drone_yaw)
            sin_y = math.sin(drone_yaw)
            wx = drone_x + y_cam * cos_y - x_cam * sin_y
            wy = drone_y + y_cam * sin_y + x_cam * cos_y

            # Canvas indices
            cols = ((wx - self.map_x_min) / self.MAP_RESOLUTION).astype(np.int32)
            rows = ((wy - self.map_y_min) / self.MAP_RESOLUTION).astype(np.int32)
            in_bounds = (cols >= 0) & (cols < self.map_w_px) & \
                        (rows >= 0) & (rows < self.map_h_px)

            cols = cols[in_bounds]
            rows = rows[in_bounds]
            z_v  = z_vert[in_bounds]

            np.add.at(depth_acc, (rows, cols), z_v)
            np.add.at(count_acc, (rows, cols), 1)

        # Render cells that have data
        filled = count_acc > 0
        mean_depth = np.where(filled, depth_acc / np.maximum(count_acc, 1), 0.0)

        if filled.any():
            d_valid = mean_depth[filled]
            d_min, d_max = d_valid.min(), d_valid.max()
            d_range = d_max - d_min if d_max > d_min else 1.0
            norm = np.zeros_like(mean_depth)
            norm[filled] = (mean_depth[filled] - d_min) / d_range
            norm_u8 = (norm * 255).astype(np.uint8)
            coloured = cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)
            canvas[filled] = coloured[filled]

        # Draw waypoint markers
        for i, (wpx, wpy) in enumerate(self.waypoints):
            col, row = self._world_to_canvas(wpx, wpy)
            if 0 <= col < self.map_w_px and 0 <= row < self.map_h_px:
                captured = i in self.snapshots
                colour = (0, 255, 0) if captured else (100, 100, 100)
                cv2.circle(canvas, (col, row), 8, colour, 2)
                cv2.putText(canvas, str(i+1), (col+10, row+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        # Scale bar
        bar_m = 2.0
        bar_px = int(bar_m / self.MAP_RESOLUTION)
        bx, by = 20, self.map_h_px - 20
        cv2.line(canvas, (bx, by), (bx + bar_px, by), (255, 255, 255), 2)
        cv2.putText(canvas, f'{bar_m:.0f}m', (bx, by - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Legend
        cv2.putText(canvas, 'warm=high  cool=low  grey=unknown',
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        self._log.info(
            f'TerrainMap stitch: {len(self.snapshots)}/9 snapshots, '
            f'{filled.sum()} px filled ({filled.mean()*100:.0f}% canvas)')

        return canvas

    # ── ROS publishing ─────────────────────────────────────────────────────────

    def publish_map_image(self, node: Node, canvas: np.ndarray, stamp):
        """Save stitched map to disk and publish as OccupancyGrid."""
        cv2.imwrite('/tmp/terrain_map_stitched.png', canvas)
        self._log.info('Terrain map saved to /tmp/terrain_map_stitched.png')

        # Publish as OccupancyGrid for RViz
        grey = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        occ = OccupancyGrid()
        occ.header.stamp    = stamp
        occ.header.frame_id = 'map'
        occ.info.resolution = self.MAP_RESOLUTION
        occ.info.width      = self.map_w_px
        occ.info.height     = self.map_h_px
        occ.info.origin.position.x = float(self.map_x_min)
        occ.info.origin.position.y = float(self.map_y_min)
        occ.info.origin.orientation.w = 1.0
        # grey 50=unknown→-1, else scale to 0-100
        data = []
        flat = grey.flatten()
        for v in flat:
            if v == 50:
                data.append(-1)
            else:
                data.append(int(v * 100 / 255))
        occ.data = data
        self._grid_pub.publish(occ)

    def publish(self, node: Node, stamp):
        """Called by the 2Hz map timer — only publishes after stitch is done."""
        pass   # stitched map is published once after sweep completes

    def coverage_pct(self):
        return len(self.snapshots) / max(len(self.waypoints), 1) * 100