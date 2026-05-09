# Cornerstone — Autonomous Safe Landing for UAVs

A ROS 2 package implementing autonomous takeoff, terrain-aware sweep mapping, and safe landing for a PX4-based quadrotor in Gazebo simulation. The system identifies a safe landing zone using depth camera analysis and a 3×3 lawnmower sweep pattern, then commits to the flattest detected patch.

---

## System Requirements

| Dependency | Version |
|---|---|
| Ubuntu | 22.04 |
| ROS 2 | Humble |
| PX4-Autopilot | main / v1.15 |
| Gazebo | Harmonic |
| Python | 3.10 |
| micro-ROS agent | snap |

### Python dependencies

```bash
pip install "numpy<2" opencv-python pyyaml ultralytics
```

> **Note:** `opencv-python` must be uninstalled and replaced with the ROS apt package if `cv_bridge` fails (`pip uninstall opencv-python -y`).

---

## Package Structure

```
cornerstone-drone/
├── launch/
│   ├── simulator.launch.py     # Full sim: PX4 + Gazebo + bridge + QGC
│   ├── sim.launch.py           # Lightweight sim variant
│   └── test.launch.py          # Minimal test launch
├── px4_offboard/
│   ├── takeoff_hold.py         # Simple takeoff and hover node
│   ├── autonomous_land.py      # Basic waypoint fly-and-land node
│   └── safe_land.py            # Full autonomous safe landing (main node)
├── vision/
│   ├── terrain_map.py          # Snapshot-based terrain stitching
│   ├── patch_scorer.py         # Per-patch depth variance scoring
│   ├── obstacle_detection.py   # YOLOv11n obstacle veto
│   ├── visualizer.py           # Debug image rendering
│   ├── stereo_depth.py         # Stereo SGBM depth (real hardware)
│   ├── terrain.py              # TFLite terrain classifier
│   ├── config.yaml             # All tunable parameters
│   ├── yolo11n.pt              # YOLO model weights
│   └── terrain_cls_int8.tflite # Terrain classification model
├── resource/
│   ├── worlds/mars.sdf         # Mars Gale Crater Gazebo world
│   └── models/                 # Mars terrain mesh assets
├── package.xml
└── setup.py
```

---

## Quick Start

### 1. Build the workspace

```bash
cd ~/drone_ws
colcon build --packages-select px4_offboard --symlink-install
source install/setup.bash
```

### 2. Launch the simulator

```bash
ros2 launch px4_offboard simulator.launch.py
```

This starts PX4 SITL + Gazebo (Mars world), the micro-ROS DDS agent, the depth camera ROS-GZ bridge, and QGroundControl. Wait ~50 seconds for everything to initialise.

### 3. Run a node

**Takeoff and hover** (for sensor verification):
```bash
ros2 run px4_offboard takeoff_hold
```

**Fly to fixed coordinate and land**:
```bash
ros2 run px4_offboard autonomous_land
```

**Full autonomous safe landing** (main mission):
```bash
ros2 run px4_offboard safe_land
```

### 4. View debug output

```bash
# Live debug image (depth + terrain map side-by-side)
ros2 run rqt_image_view rqt_image_view
# Select topic: /safe_land/debug_image

# Saved frames (written during sweep)
eog /tmp/safe_land_frame_*.png

# Stitched terrain map (written after sweep completes)
eog /tmp/terrain_map_stitched.png
```

---

## Nodes

### `takeoff_hold`

Minimal node for hardware-in-the-loop testing. Arms, engages offboard mode, climbs to 8 m, and holds position indefinitely.

**Parameters (edit in source):**
- `target_altitude` — NED altitude in metres (default `-8.0`)

---

### `autonomous_land`

Flies to a fixed NED coordinate, then descends until `VehicleLandDetected` fires.

**Parameters (edit in source):**
- `target_x`, `target_y` — NED target coordinates (default `10.0, 5.0`)
- `takeoff_altitude` — NED takeoff altitude (default `-5.0`)
- `descent_speed` — descent rate in m/s (default `0.5`)

---

### `safe_land`

The main autonomous mission node. Full state machine with terrain-aware safe zone selection.

**State machine:**

```
IDLE → TAKEOFF → FLY_TO_COORD → SWEEP → FLY_TO_SAFE → DESCEND → LANDED
                                    ↓ (no safe zone)
                                 SPIRAL SEARCH → RTL (after 5 attempts)
```

**Mission flow:**
1. **IDLE** — primes PX4 offboard mode with 10 heartbeat cycles, then arms
2. **TAKEOFF** — climbs to `takeoff_alt` (−5 m NED) holding XY
3. **FLY_TO_COORD** — flies to a random target within ±100 m at scan altitude (−3 m)
4. **SWEEP** — executes a 3×3 lawnmower grid (4 m spacing) centred on the target; captures one depth snapshot and 10 scoring frames at each of 9 waypoints
5. **FLY_TO_SAFE** — flies to the score-weighted best patch world coordinate
6. **DESCEND** — descends at 0.5 m/s until `VehicleLandDetected`
7. **LANDED** — disarms

**Key parameters (edit in `__init__`):**

| Parameter | Default | Description |
|---|---|---|
| `takeoff_alt` | `-5.0` | NED takeoff altitude (m) |
| `scan_alt` | `-3.0` | Sweep altitude — must be within depth sensor range |
| `target_x/y` | random ±100 | NED landing zone coordinates |
| `sweep_spacing` | `4.0` | Metres between sweep waypoints |
| `frames_per_waypoint` | `10` | Scoring frames collected per waypoint |
| `descent_speed` | `0.5` | Descent rate (m/s) |
| `max_spiral_attempts` | `5` | Spiral search attempts before RTL |

**Published topics:**

| Topic | Type | Description |
|---|---|---|
| `/safe_land/debug_image` | `sensor_msgs/Image` | Side-by-side depth + terrain map |
| `/safe_land/occupancy_map` | `nav_msgs/OccupancyGrid` | Stitched terrain map for RViz |
| `/safe_land/map_cloud` | `sensor_msgs/PointCloud2` | 3D point cloud for RViz |

---

## Vision Pipeline

```
/depth_camera (32FC1)
        │
        ├─── Track A: Terrain Mapping
        │         Back-project pixels → world XY
        │         Capture 1 snapshot per waypoint
        │         Stitch 9 snapshots → top-down map
        │         → /tmp/terrain_map_stitched.png
        │
        └─── Track B: Safe Zone Scoring
                  Divide image into 32×32 patches
                  Per-patch depth std dev → flatness score
                  score = max(0, 1 − std/0.035)
                  Accumulate across all 90 frames
                  argmax → best patch pixel → world (NED)
                  → (safe_x, safe_y)
```

### Camera intrinsics

Derived from depth measurements at 8 m altitude (HFOV ≈ 44.7°, 640×480):

```
fx = fy = 777.59
cx = 320.0,  cy = 240.0
```

---

## Configuration

All tunable parameters live in `vision/config.yaml`. Key sections:

```yaml
patch_scorer:
  patch_size: 32          # pixels per scoring patch

obstacle_detection:
  model_path: "..."       # absolute path to yolo11n.pt — update this
  conf: 0.4               # YOLO confidence threshold

terrain_map:
  flatness_threshold: 0.04  # depth std dev (m) for flat classification
  map_size_m: 20.0          # map coverage area
  resolution: 0.1           # metres per grid cell
```

> **Important:** Update `obstacle_detection.model_path` and `terrain.model_path` to absolute paths on your machine before running.

---

## RViz Visualisation

```bash
ros2 run rviz2 rviz2
```

Add displays:
- **Image** → `/safe_land/debug_image`
- **Map** → `/safe_land/occupancy_map` (set fixed frame to `map`)
- **PointCloud2** → `/safe_land/map_cloud`

---

## Simulator Configuration

Edit `launch/simulator.launch.py` to change:

```python
PX4_AUTOPILOT_DIR = '~/PX4-Autopilot'   # path to PX4 source
WORLD             = 'mars'               # Gazebo world name
MODEL             = 'gz_x500_depth_down' # drone model
SPAWN_POSE        = '-20,-50,5,0,0,0'   # Gazebo spawn XYZ + RPY
DDS_PORT          = '8888'              # micro-ROS agent UDP port
QGC_PATH          = '~/Downloads/...'   # QGroundControl AppImage path
```

The Mars Gale Crater terrain mesh is included in `resource/models/`. It covers approximately Gazebo x∈[−50, 100], y∈[−50, 50] — keep `target_x/y` within this range.

---

## Known Issues

- **NumPy version conflict** — `cv_bridge` requires NumPy < 2. If you see `_ARRAY_API not found`, run `pip install "numpy<2"` and `pip uninstall opencv-python -y`.
- **YOLO startup delay** — first inference takes 200+ ms as the model loads. Subsequent frames are ~50 ms.
- **Scan altitude constraint** — the depth sensor saturates beyond ~3 m. Keep `scan_alt` at −3.0 or lower magnitude.
- **Terrain map stitching** — the stitched map requires the drone camera to be pointing straight down. Any pitch/roll during the sweep causes triangular projection artefacts.

---

## License

TODO