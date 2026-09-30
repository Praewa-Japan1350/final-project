# -*- coding: utf-8 -*-
"""
Configuration and Constants Module for RoboMaster SLAM.
Contains grid settings, robot speed parameters, sensor thresholds,
and directional maps.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Paths configuration (Outputs are saved in the 'results' folder)
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CODE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Grid & robot exploration configuration
GRID_W, GRID_H = 6, 6
GRID_SIZE_M = 0.60
# Assignment rule: a target may be engaged only when it is at most two tiles
# away. Keep this derived from the configured grid pitch.
MAX_TARGET_RANGE_MM = int(2 * GRID_SIZE_M * 1000)
SPEED = 0.30            # Smooth forward velocity speed (m/s)
ROT_SPEED = 50          # Precise turning speed (deg/s) - retains 90° precision
RECENTER_SPEED = 0.25  # Smooth strafe recentering (m/s)
GIMBAL_SPEED = 280      # Fast & smooth gimbal yaw speed (deg/s)
COLOR_SCAN_HOLD_S = 0.35 # Briefly settle each gimbal view before fresh-frame voting
TOF_ID = 1
# Blaster bullet type: "water" = กระสุนเจล (Gel Beads), "ir" = กระสุนอินฟราเรด (Infrared)
BLASTER_FIRE_TYPE = "water"
# Sensor-adaptor digital output level for the installed IR obstacle modules.
IR_ACTIVE_LEVEL = 0
IR_EVADE_SPEED_MPS = 0.10
IR_EVADE_ACCEL_MPS2 = 0.30
WALL_THRESHOLD_MM = 550 # Distance <= 550mm indicates a foam wall on immediate cell border (open passages >= 700mm)
RECENTER_WALL_MAX_MM = 400 # Use walls within 40cm for recentering
TARGET_WALL_DIST_MM = 200 # Target distance from a single detected wall during recentering (20cm)
SAFE_MIN_MM = 280       # Minimum safe distance to a single side wall (mm)
SAFE_MAX_MM = 300       # Maximum safe distance to a single side wall (mm)
SAFE_FRONT_DIST_MM = 200 # Safe front distance threshold (200mm)
FRONT_BRAKE_DIST_MM = 200 # Immediate front obstacle boundary (200mm)
EMERGENCY_STOP_DIST_MM = 100 # Emergency collision stop threshold (100mm / 10cm - increased from 60mm)
MAX_SHIFT_LATERAL_M = 0.10      # Allow the full correction needed to reach the 20cm wall target
MAX_SHIFT_LONGITUDINAL_M = 0.20 # Allow the full correction needed to reach the 20cm wall target
DEADBAND_M = 0.020              # Deadband tolerance (2.0cm) - ignores minor offsets to prevent jitter
ENABLE_RECENTER = True              # เปิดระบบ Recenter ปรับกึ่งกลางช่อง
ENABLE_LONGITUDINAL_RECENTER = True   # เปิดระบบ Centering หน้า-หลัง เพื่อจัดระยะห่างกำแพง 20cm (TARGET_WALL_DIST_MM)



DIRECTIONS = ["NORTH", "EAST", "SOUTH", "WEST"]
DELTA = {
    "NORTH": (0, 1),
    "EAST": (1, 0),
    "SOUTH": (0, -1),
    "WEST": (-1, 0),
    "NORTHEAST": (0.707, 0.707),
    "SOUTHEAST": (0.707, -0.707),
    "SOUTHWEST": (-0.707, -0.707),
    "NORTHWEST": (-0.707, 0.707),
}
SYMBOL = {"NORTH": "^", "EAST": ">", "SOUTH": "v", "WEST": "<"}

# Simulated foam walls on cell borders (used only with --sim)
SIM_H_WALLS = {(x, 0) for x in range(1, GRID_W + 1)} | {(x, GRID_H) for x in range(1, GRID_W + 1)}
SIM_V_WALLS = {(0, y) for y in range(1, GRID_H + 1)} | {(GRID_W, y) for y in range(1, GRID_H + 1)}
# Sample interior foam walls in simulation
SIM_H_WALLS |= {(1, 2), (2, 4), (3, 3)}
SIM_V_WALLS |= {(2, 1), (2, 2), (3, 4)}

# Simulated ground truth walls for standalone testing
GROUND_TRUTH_H_WALLS = set(SIM_H_WALLS)
GROUND_TRUTH_V_WALLS = set(SIM_V_WALLS)


def set_grid_dimensions(w, h, cell_size_m=None):
    """Dynamically set maze grid width, height, and cell pitch."""
    global GRID_W, GRID_H, GRID_SIZE_M, MAX_TARGET_RANGE_MM, SIM_H_WALLS, SIM_V_WALLS, GROUND_TRUTH_H_WALLS, GROUND_TRUTH_V_WALLS
    GRID_W = max(2, int(w))
    GRID_H = max(2, int(h))
    if cell_size_m is not None:
        GRID_SIZE_M = float(cell_size_m)
        MAX_TARGET_RANGE_MM = int(2 * GRID_SIZE_M * 1000)
    # Rebuild simulated outer perimeter walls for the new grid dimension
    SIM_H_WALLS.clear()
    SIM_H_WALLS |= {(x, 0) for x in range(1, GRID_W + 1)} | {(x, GRID_H) for x in range(1, GRID_W + 1)}
    SIM_V_WALLS.clear()
    SIM_V_WALLS |= {(0, y) for y in range(1, GRID_H + 1)} | {(GRID_W, y) for y in range(1, GRID_H + 1)}
    # Add sample interior walls if within bounds
    for hw in [(1, 2), (2, 4), (3, 3)]:
        if hw[0] <= GRID_W and hw[1] <= GRID_H:
            SIM_H_WALLS.add(hw)
    for vw in [(2, 1), (2, 2), (3, 4)]:
        if vw[0] <= GRID_W and vw[1] <= GRID_H:
            SIM_V_WALLS.add(vw)
    GROUND_TRUTH_H_WALLS.clear()
    GROUND_TRUTH_H_WALLS.update(SIM_H_WALLS)
    GROUND_TRUTH_V_WALLS.clear()
    GROUND_TRUTH_V_WALLS.update(SIM_V_WALLS)


def in_bounds(cell):
    """Check if the cell coordinate (x, y) is within grid bounds."""
    return 1 <= cell[0] <= GRID_W and 1 <= cell[1] <= GRID_H


def adjacent(cell, direction):
    """Return the adjacent cell coordinate in the given cardinal direction."""
    dx, dy = DELTA[direction]
    return cell[0] + dx, cell[1] + dy
