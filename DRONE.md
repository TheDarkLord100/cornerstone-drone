# Cornerstone — Real Drone Platform

This document covers the physical hardware platform, bill of materials, and media links for the Cornerstone autonomous safe landing project.

---

## Presentation & Media

| Resource | Link |
|---|---|
| Project Presentation | [View Slides](https://docs.google.com/presentation/d/1K7SbB6rz-VTCjaUMu1NwcMw__kq3oKHs_KjBb6ONFzU/edit?usp=drive_link) |
| Simulation Demo Video | [Drive Link](https://drive.google.com/file/d/1579sU0P3ycwsvCvbfVRvkOUAUVSqg0GL/view?usp=sharing) |
| Real Drone Flight Video | [Drive Link](https://drive.google.com/file/d/1pDiPfk25IIsg0cCAAr1VD3vqYVS_vWIv/view?usp=sharing) |

---

## Real Drone Overview

The physical platform is a **S500 carbon fibre quadrotor** running PX4 on a Pixhawk 2.4.8 flight controller, with a Raspberry Pi 5 onboard computer handling the vision pipeline and ROS 2 offboard control. The stereo camera pair provides depth estimation via SGBM, and a Benewake TFMini-S LiDAR gives metric altitude above ground for accurate scan height control.

---

## Bill of Materials

| # | Component | Part | Role |
|---|---|---|---|
| 1 | **Frame** | S500 Carbon Fiber Quadcopter Drone Frame Kit | Airframe — 500mm wheelbase |
| 2 | **Propellers** | Pro-Range 1147 (11×4.7) Carbon Fiber — 1CW + 1CCW pair | Thrust generation |
| 3 | **Motors** | DYS D3548-5 900 KV BLDC Motor × 4 | Brushless drive |
| 4 | **Battery** | GenX 14.8V 4S 5200mAh 40C/80C LiPo | Main power source |
| 5 | **ESC** | Hobbywing XRotor FPV G2 4-in-1 65A | Motor speed control |
| 6 | **Flight Controller** | Pixhawk 2.4.8 PX4 32-bit Autopilot | PX4 — EKF2, offboard CTL, mixing |
| 7 | **Shock Absorber** | Anti-Vibration Mount for Pixhawk | FC vibration isolation |
| 8 | **Onboard Computer** | Raspberry Pi 5 — 8GB RAM | ROS 2, vision pipeline, MAVLink |
| 9 | **Stereo Cameras** | Waveshare OV9281-120 Mono Global Shutter 1MP × 2 | Stereo depth estimation |
| 10 | **Power Module** | APM/Pixhawk Power Module BEC 3A XT60 28V 90A | FC power + current sensing |
| 11 | **BEC** | UBEC 8A + 4A Dual Channel | 5V regulated supply for RPi |
| 12 | **RC System** | Flysky FS-i6X 2.4GHz 6CH + FS-iA10B 10CH Receiver | Manual override / RC control |
| 13 | **Telemetry** | 433 MHz 100mW Radio Telemetry Kit | Ground station link (QGC) |
| 14 | **LiDAR** | Benewake TFMini-S 12m Micro LiDAR — UART | AGL distance for scan altitude |
| 15 | **GPS + Compass** | NEO-M8N Ready-to-Sky GPS Module with Magnetometer | Position estimation (EKF2) |

---

## Hardware Architecture

```
                        ┌─────────────────────┐
                        │  PIXHAWK 2.4.8 FC   │
         ┌──────────────│  PX4 · EKF2 · PWM   │──────────────┐
         │  MAVLink     │  offboard control    │  SBUS/IBUS   │
         │  UART        └─────────────────────┘              │
         │                   │          │                     │
         │              Compass       4-in-1 ESC         RC RX (FS-iA10B)
         │              NEO-M8N       65A · PWM              │
         │                            │                  RC TX (FS-i6X)
         │                       4× DYS D3548                │ (RF 2.4GHz)
         │                       900KV Motors
         │
┌─────────────────┐        ┌──────────────┐
│  RASPBERRY Pi 5 │        │  TFMini-S    │
│  8GB RAM        │◄─UART──│  LiDAR 12m   │
│  ROS 2 Humble   │        └──────────────┘
│  Vision pipeline│
│  terrain_map    │        ┌──────────────┐
│  safe_land.py   │◄─CSI───│  OV9281 ×2   │
└─────────────────┘        │  Stereo Cams │
                           └──────────────┘

Power rail:
  GenX 4S 5200mAh LiPo → Power Module → Pixhawk FC
                       → 4-in-1 ESC  → Motors
                       → UBEC 5V     → Raspberry Pi 5
```

---

## Key Hardware Notes

**OV9281 Global Shutter cameras** — the global shutter is important for fast motion; rolling shutter cameras produce skew artefacts during flight that degrade stereo matching. The 120° FOV gives a wide baseline view for depth estimation.

**TFMini-S LiDAR** — connected to Raspberry Pi via UART, not to the Pixhawk. Used by the vision pipeline to verify scan altitude and as a scale anchor for stereo depth. Range: 0.1–12 m, accuracy ±1%.

**Hobbywing 4-in-1 ESC** — single unit drives all four motors, simplifying wiring and reducing weight compared to four individual ESCs.

**UBEC dual channel** — 8A channel powers the Raspberry Pi 5; 4A channel available for peripherals. Isolated from the ESC power rail to prevent noise coupling into the onboard computer.

**Pixhawk shock absorber** — critical for EKF2 stability. Without vibration isolation, high-frequency motor noise contaminates the IMU and causes estimation drift.

---

## Simulation vs Real Hardware

| Aspect | Simulation | Real Hardware |
|---|---|---|
| Depth sensing | Single synthetic depth camera (32FC1) | Stereo OV9281 pair → SGBM depth |
| Altitude | EKF2 barometer | EKF2 barometer + TFMini-S LiDAR |
| Position | GPS (Gazebo ground truth via EKF2) | NEO-M8N GPS + EKF2 |
| Compute | Host PC | Raspberry Pi 5 onboard |
| Camera topic | `/depth_camera` | Stereo CSI pipeline → ROS topic |
| Offboard comms | UDP localhost (micro-ROS) | UART MAVLink (RPi → Pixhawk) |