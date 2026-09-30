# -*- coding: utf-8 -*-
"""
Robot and Sensor Control Module.
Handles DJI RoboMaster EP hardware interaction (Gimbal, Chassis Mecanum, ToF sensor),
gimbal-only 4-direction scanning, and fine-tuning cell auto-recentering.
"""

import math
import time
from collections import defaultdict

import numpy as np
import config
import vision as vision_config

from config import (
    BLASTER_FIRE_TYPE,
    BLASTER_PITCH_UP_DEG,
    DEADBAND_M,
    DELTA,
    DIRECTIONS,
    EMERGENCY_STOP_DIST_MM,
    ENABLE_LONGITUDINAL_RECENTER,
    ENABLE_RECENTER,
    COLOR_SCAN_HOLD_S,
    FRONT_BRAKE_DIST_MM,
    GIMBAL_SPEED,
    GRID_SIZE_M,
    IR_ACTIVE_LEVEL,
    MAX_SHIFT_LATERAL_M,
    MAX_SHIFT_LONGITUDINAL_M,
    MAX_TARGET_RANGE_MM,
    RECENTER_WALL_MAX_MM,
    RECENTER_SPEED,
    ROT_SPEED,
    SAFE_FRONT_DIST_MM,
    SPEED,
    TARGET_WALL_DIST_MM,
    TOF_ID,
    WALL_THRESHOLD_MM,
    adjacent,
    in_bounds,
)
from navigation import is_wall_distance, sim_has_wall_between
import state
from vision import (
    CAMERA_HFOV_DEG,
    CAMERA_VFOV_DEG,
    CROP_X,
    CROP_Y,
    COLOR_SCAN_PITCH,
    COLOR_SCAN_SPEED,
    COLOR_VOTE_FRAME_COUNT,
    COLOR_VOTE_MIN_COUNT,
    COLORS,
    SHAPES,
    detect_stable_signs,
)

try:
    from robomaster import robot, protocol as rm_protocol
except ImportError:
    robot = None
    rm_protocol = None

# ─── Noise filtering state ───────────────────────────────────────────
# Per-direction EMA history for temporal smoothing across consecutive scans.
# Key: (cell, direction)  Value: smoothed distance (mm)
_ema_history: dict = defaultdict(lambda: None)
EMA_ALPHA = 0.4          # Weight for new reading (0→trust old, 1→trust new)
NUM_SAMPLES = 9          # More samples → better outlier rejection
SAMPLE_INTERVAL_S = 0.03 # 30ms between samples (sensor rate ~100Hz@freq=10)
LIGHT_NOISE_BAND_MM = 80 # Borderline zone around WALL_THRESHOLD for re-confirm
# A 2.5% horizontal error is about 3 degrees with the current camera view.
# This is deliberately tighter than the old 10% gate before an IR shot.
AIM_CENTER_X_FRACTION = 0.025
AIM_CENTER_Y_FRACTION = 0.03
_last_ir_debug = None


def reset_sensor_filters():
    """Clear scan smoothing between Run 1 and the physically reset Run 2."""
    _ema_history.clear()


def sub_tof_handler(info):
    """Callback to receive distance measurement from RoboMaster ToF sensor."""
    if isinstance(info, (list, tuple)) and len(info) >= TOF_ID:
        state.current_distance = info[TOF_ID - 1]
    elif not isinstance(info, (list, tuple)):
        state.current_distance = info


def sub_ir_handler(info):
    """Store digital sensor-adaptor IO states for the four IR obstacles."""
    global _last_ir_debug
    if not isinstance(info, (list, tuple)):
        return
    # SensorAdaptor DDS sends (io_values, adc_values); accept a flat IO list too.
    io_values = info[0] if len(info) == 2 and isinstance(info[0], (list, tuple)) else info
    if len(io_values) < 4:
        return
    # RoboMaster adapter array is ordered by adapter id, then port:
    # (id1/port1, id1/port2, id2/port1, id2/port2).
    raw_levels = [int(io_values[i]) for i in range(4)]
    state.ir_values = {
        "front_left": int(io_values[0]) == IR_ACTIVE_LEVEL,
        "rear_left": int(io_values[1]) == IR_ACTIVE_LEVEL,
        "front_right": int(io_values[2]) == IR_ACTIVE_LEVEL,
        "rear_right": int(io_values[3]) == IR_ACTIVE_LEVEL,
    }
    if raw_levels != _last_ir_debug:
        print(f"[IR IO] FL={raw_levels[0]} RL={raw_levels[1]} FR={raw_levels[2]} RR={raw_levels[3]} active={IR_ACTIVE_LEVEL}")
        _last_ir_debug = raw_levels


def sub_attitude_handler(info):
    """Callback to receive chassis attitude (yaw, pitch, roll) in degrees."""
    if isinstance(info, (list, tuple)) and len(info) >= 1:
        yaw = float(info[0])
        state.current_yaw = yaw
        if state.initial_yaw is None:
            state.initial_yaw = yaw


def sub_position_handler(info):
    """Callback to receive chassis position (x, y, z in meters/deg)."""
    if isinstance(info, (list, tuple)) and len(info) >= 2:
        state.chassis_pos_x = float(info[0])
        state.chassis_pos_y = float(info[1])


def _normalize_angle(deg):
    """Normalize angle to [-180, 180] degrees."""
    while deg > 180.0:
        deg -= 360.0
    while deg < -180.0:
        deg += 360.0
    return deg


def stop_and_settle(ep_chassis, sim_mode, settle_s=0.25):
    """Command zero velocity and allow the chassis to become still."""
    if not sim_mode and ep_chassis:
        ep_chassis.drive_speed(x=0, y=0, z=0)
    time.sleep(settle_s if not sim_mode else 0.05)


def distance_in_simulation(cell, direction):
    """Simulate ToF distance to border foam wall (TARGET_WALL_DIST_MM) or open cell (900mm)."""
    has_wall = sim_has_wall_between(cell, direction)
    return TARGET_WALL_DIST_MM if has_wall else 900


def _iqr_filter(samples):
    """
    Remove outliers from a list of distance samples using the IQR method.
    Returns only the inliers. If too few remain, returns the full list.
    This is critical for filtering noise spikes caused by lighting changes.
    """
    if len(samples) < 4:
        return samples
    arr = np.array(samples, dtype=float)
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    # Use 1.5× IQR as fence — standard Tukey outlier rule
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inliers = arr[(arr >= lower) & (arr <= upper)]
    return inliers.tolist() if len(inliers) >= 3 else samples


def _ema_smooth(key, raw_value):
    """
    Apply Exponential Moving Average to smooth out temporal noise.
    This filters gradual lighting changes that cause drifting ToF readings.
    """
    prev = _ema_history[key]
    if prev is None:
        _ema_history[key] = raw_value
        return raw_value
    smoothed = EMA_ALPHA * raw_value + (1 - EMA_ALPHA) * prev
    _ema_history[key] = smoothed
    return smoothed


def read_distance_with_gimbal(ep_gimbal, cell, current_heading, target_dir, sim_mode,
                              gimbal_pitch=3, move_gimbal=True):
    """
    Rotate ONLY the gimbal to target_dir relative to current_heading,
    read the ToF distance, and return distance in mm.
    Chassis does NOT move!

    Noise-robust pipeline:
      1. Collect NUM_SAMPLES raw readings
      2. Discard readings ≤ 30mm (floor/sensor noise)
      3. IQR outlier rejection (removes lighting-induced spikes)
      4. Median of remaining inliers
      5. EMA temporal smoothing (handles gradual light changes)
    """
    if sim_mode:
        time.sleep(0.12)
        distance = distance_in_simulation(cell, target_dir)
        state.current_distance = distance
        return distance

    # Relative turn index: 0=front, 1=right (clockwise), 2=back, 3=left (counter-clockwise)
    diff = (DIRECTIONS.index(target_dir) - DIRECTIONS.index(current_heading)) % 4
    yaw_map = {0: 0, 1: 90, 2: 180, 3: -90}
    target_yaw = yaw_map[diff]

    # The caller may choose a down pitch for a combined wall/color scan.
    if ep_gimbal and move_gimbal:
        if not ep_gimbal.moveto(pitch=gimbal_pitch, yaw=target_yaw,
                                pitch_speed=GIMBAL_SPEED,
                                yaw_speed=GIMBAL_SPEED).wait_for_completed(timeout=1.5):
            return float("inf")

    # Allow sensor to settle after gimbal rotation
    time.sleep(0.20)

    # Step 1: Collect raw samples
    samples = []
    for _ in range(NUM_SAMPLES):
        samples.append(state.current_distance)
        time.sleep(SAMPLE_INTERVAL_S)

    # Step 2: Discard invalid readings (floor reflection, sensor error)
    valid = [s for s in samples if s > 30]
    if not valid:
        distance = float(samples[-1])
        state.current_distance = distance
        return distance

    # Step 3: IQR outlier rejection (removes lighting-induced spikes)
    filtered = _iqr_filter(valid)

    # Step 4: Median of inliers
    median_val = float(np.median(filtered))

    # Step 5: EMA temporal smoothing
    ema_key = (cell, target_dir)
    smoothed = _ema_smooth(ema_key, median_val)
    state.current_distance = smoothed
    return smoothed


def read_distance_at_yaw(ep_gimbal, cell, direction_key, yaw, sim_mode):
    """Measure target range at a diagonal bearing without mapping a wall."""
    if sim_mode:
        return 9999.0
    if ep_gimbal:
        if not ep_gimbal.moveto(pitch=3, yaw=yaw, pitch_speed=COLOR_SCAN_SPEED,
                                yaw_speed=GIMBAL_SPEED).wait_for_completed(timeout=1.5):
            return float("inf")
    time.sleep(0.15)
    samples = []
    for _ in range(NUM_SAMPLES):
        samples.append(float(state.current_distance))
        time.sleep(SAMPLE_INTERVAL_S)
    valid = [sample for sample in samples if math.isfinite(sample) and sample > 30]
    if not valid:
        return float(samples[-1])
    distance = _ema_smooth((cell, direction_key), float(np.median(_iqr_filter(valid))))
    state.current_distance = distance
    return distance


def read_target_range_at_yaw(ep_gimbal, cell, direction_key, yaw, sim_mode):
    """Read range at the low target-search angle after wall mapping is done."""
    if sim_mode:
        return 9999.0
    if ep_gimbal:
        if not ep_gimbal.moveto(pitch=COLOR_SCAN_PITCH - 3, yaw=yaw,
                                pitch_speed=COLOR_SCAN_SPEED,
                                yaw_speed=GIMBAL_SPEED).wait_for_completed(timeout=1.5):
            return float("inf")
    time.sleep(0.12)
    samples = []
    for _ in range(NUM_SAMPLES):
        samples.append(float(state.current_distance))
        time.sleep(SAMPLE_INTERVAL_S)
    valid = [sample for sample in samples if math.isfinite(sample) and sample > 30]
    if not valid:
        return float(samples[-1])
    distance = _ema_smooth((cell, direction_key, "target"),
                           float(np.median(_iqr_filter(valid))))
    state.current_distance = distance
    return distance


def turn_to_direction(ep_chassis, ep_gimbal, current, target, sim_mode):
    """Turn the chassis from current direction to target direction and keep gimbal centered."""
    if current == target:
        return True
    if sim_mode:
        time.sleep(0.12)
        return True

    def wait_turn(action):
        try:
            completed = action.wait_for_completed(timeout=2.5)
        except Exception:
            completed = False
        if not completed:
            ep_chassis.drive_speed(x=0, y=0, z=0)
        return bool(completed)

    # Use IMU closed-loop yaw if initial yaw has been calibrated
    # Confirmed conventions (from debug):
    #   chassis.move(z>0) = CCW (left),  z<0 = CW (right)
    #   IMU yaw: CW = increases,  CCW = decreases  (opposite to chassis!)
    # heading_offsets use IMU convention; z = -yaw_err converts to chassis convention
    heading_offsets = {"NORTH": 0.0, "EAST": 90.0, "SOUTH": 180.0, "WEST": -90.0}
    if state.initial_yaw is not None:
        start_offset = heading_offsets.get(state.initial_heading, 0.0)
        target_yaw = _normalize_angle(state.initial_yaw + heading_offsets[target] - start_offset)
        yaw_err = _normalize_angle(target_yaw - state.current_yaw)
        z_cmd = round(-yaw_err, 1)
        print(f"[TURN DEBUG IMU] {current}->{target} | initial_yaw={state.initial_yaw:.1f} current_yaw={state.current_yaw:.1f} target_yaw={target_yaw:.1f} yaw_err={yaw_err:.1f} z_cmd={z_cmd}")
        if abs(yaw_err) > 1.5:
            if not wait_turn(ep_chassis.move(x=0, y=0, z=z_cmd, z_speed=ROT_SPEED)):
                return False
            time.sleep(0.08)
            # Fine-tuning pass
            yaw_err2 = _normalize_angle(target_yaw - state.current_yaw)
            print(f"[TURN DEBUG FT] after_yaw={state.current_yaw:.1f} yaw_err2={yaw_err2:.1f}")
            if abs(yaw_err2) > 5.0:
                z_cmd2 = round(-yaw_err2, 1)
                print(f"[TURN DEBUG FT] fine-tune z_cmd2={z_cmd2}")
                if not wait_turn(ep_chassis.move(x=0, y=0, z=z_cmd2, z_speed=ROT_SPEED)):
                    return False
    else:
        turn = (DIRECTIONS.index(target) - DIRECTIONS.index(current)) % 4
        print(f"[TURN DEBUG FALLBACK] {current}->{target} turn={turn}")
        if turn == 1:
            if not wait_turn(ep_chassis.move(x=0, y=0, z=-90, z_speed=ROT_SPEED)):
                return False
        elif turn == 2:
            if not wait_turn(ep_chassis.move(x=0, y=0, z=-90, z_speed=ROT_SPEED)):
                return False
            if not wait_turn(ep_chassis.move(x=0, y=0, z=-90, z_speed=ROT_SPEED)):
                return False
        else:
            if not wait_turn(ep_chassis.move(x=0, y=0, z=90, z_speed=ROT_SPEED)):
                return False

    if ep_gimbal and not sim_mode:
        if not ep_gimbal.recenter().wait_for_completed(timeout=1.5):
            return False
    return True


def align_heading(ep_chassis, heading, sim_mode):
    """Align the stationary chassis to its cardinal heading and reset its yaw reference."""
    if sim_mode or state.initial_yaw is None or heading not in DIRECTIONS:
        return

    heading_offsets = {"NORTH": 0.0, "EAST": 90.0, "SOUTH": 180.0, "WEST": -90.0}
    start_offset = heading_offsets.get(state.initial_heading, 0.0)
    target_yaw = _normalize_angle(state.initial_yaw + heading_offsets[heading] - start_offset)
    stop_and_settle(ep_chassis, sim_mode, settle_s=0.20)
    yaw_error = _normalize_angle(target_yaw - state.current_yaw)
    if abs(yaw_error) > 1.5:
        if not ep_chassis.move(x=0, y=0, z=round(-yaw_error, 1),
                               z_speed=ROT_SPEED).wait_for_completed(timeout=2.5):
            ep_chassis.drive_speed(x=0, y=0, z=0)
            time.sleep(0.10)

    # One short fine-tune pass, matching the existing turn controller.
    yaw_error = _normalize_angle(target_yaw - state.current_yaw)
    if abs(yaw_error) > 2.0:
        if not ep_chassis.move(x=0, y=0, z=round(-yaw_error, 1),
                               z_speed=ROT_SPEED).wait_for_completed(timeout=2.5):
            ep_chassis.drive_speed(x=0, y=0, z=0)
            time.sleep(0.10)

    stop_and_settle(ep_chassis, sim_mode, settle_s=0.20)
    final_error = _normalize_angle(target_yaw - state.current_yaw)
    print(f"[HEADING ALIGN] target={heading} yaw_error={final_error:.1f}deg")
    if abs(final_error) <= 3.0:
        # Rebase to the exact cardinal target, clearing accumulated yaw offset.
        state.initial_yaw = target_yaw
        state.initial_heading = heading


def _front_is_clear(distance):
    return math.isfinite(float(distance)) and float(distance) > FRONT_BRAKE_DIST_MM


def move_one_cell(ep_chassis, sim_mode, heading=None, returning=False, dashboard=None, retry=1):
    """
    Move forward exactly 1 grid cell (0.60m) using smooth velocity control (drive_speed).
    Uses tight time-distance coupling, robust stall grace periods, and front-wall proximity braking to completely
    prevent overshooting into walls.
    """
    if sim_mode:
        time.sleep(0.18)
        state.last_move_reason = "SIM_COMPLETE"
        state.last_move_distance_m = config.GRID_SIZE_M
        return True

    # A caller must refresh the forward ToF after turning. Fail closed if a
    # current reading already puts the robot inside the 20cm safety boundary.
    current_distance = float(state.current_distance)
    if not math.isfinite(current_distance) or current_distance <= 30:
        ep_chassis.drive_speed(x=0, y=0, z=0)
        state.last_move_reason = "TOF_INVALID"
        state.last_move_distance_m = 0.0
        return False
    if current_distance <= FRONT_BRAKE_DIST_MM:
        ep_chassis.drive_speed(x=0, y=0, z=0)
        state.last_move_reason = "FRONT_BRAKE"
        state.last_move_distance_m = 0.0
        return False

    # Record starting odometry position
    start_x = state.chassis_pos_x
    start_y = state.chassis_pos_y

    # Drive to the measured cell pitch, then judge completion from the final
    # odometry sample after braking. A short target accumulates a large pose
    # error over a multi-cell route.
    target_dist = config.GRID_SIZE_M
    # Allow ample time for acceleration and encoder callbacks while keeping a firm bound.
    max_duration = (target_dist / SPEED) + 1.80
    # Hold the heading measured after the caller's cardinal alignment. Side
    # IR sensors must not steer the robot sideways: that pulls it off the grid.
    heading_target_yaw = state.current_yaw

    def heading_correction():
        error = _normalize_angle(heading_target_yaw - state.current_yaw)
        if abs(error) <= 1.5:
            return 0.0
        # RoboMaster chassis z is opposite the IMU yaw convention used above.
        return max(-20.0, min(20.0, -error * 1.5))

    ep_chassis.drive_speed(x=SPEED, y=0, z=heading_correction())

    emergency_stop = False
    tof_invalid = False
    front_wall_reached = False
    start_time = time.time()
    traveled = 0.0
    elapsed = 0.0
    last_progress_distance = 0.0
    last_progress_time = start_time
    odometry_stalled = False

    while not state.stop_requested:
        time.sleep(0.015)
        # Compute distance traveled from wheel encoders
        traveled = math.hypot(state.chassis_pos_x - start_x, state.chassis_pos_y - start_y)
        elapsed = time.time() - start_time
        if traveled >= last_progress_distance + 0.005:
            last_progress_distance = traveled
            last_progress_time = time.time()
        elif elapsed > 1.20 and time.time() - last_progress_time > 1.20:
            odometry_stalled = True
            break
        elif elapsed > 0.40 and traveled < 0.005:
            # Re-assert forward speed command in case initial CAN bus packet ramp was sluggish
            ep_chassis.drive_speed(x=SPEED, y=0, z=heading_correction())

        # 1. Emergency collision stop threshold (increased slightly from 60mm to EMERGENCY_STOP_DIST_MM)
        if not math.isfinite(float(state.current_distance)) or state.current_distance <= 30:
            emergency_stop = True
            tof_invalid = True
            break
        if state.current_distance <= EMERGENCY_STOP_DIST_MM:
            emergency_stop = True
            break

        # 2. Front wall arrival: if moving into a cell facing a wall,
        # when ToF shows we reached the cell center (~200mm from wall)
        # and we have traveled at least 0.50m, brake cleanly.
        if traveled >= config.GRID_SIZE_M - 0.10 and EMERGENCY_STOP_DIST_MM < state.current_distance <= (TARGET_WALL_DIST_MM + 20):
            front_wall_reached = True
            break

        # 3. Precision distance control: calculate remaining distance
        rem_dist = target_dist - traveled
        # Brake cut-off 1.5cm before target: at crawling speed (0.08 m/s),
        # coasting adds ~1.5cm, landing the robot exactly at target_dist (0.60m)
        if rem_dist <= 0.015 or elapsed >= max_duration:
            break

        # Smooth deceleration in the last 15 cm to eliminate overshoot past 0.60m
        if rem_dist < 0.15:
            current_speed = max(0.08, SPEED * (rem_dist / 0.15))
        else:
            current_speed = SPEED

        # Drive exactly along the selected grid direction and continuously
        # correct chassis yaw; keep lateral velocity at zero for every cell.
        ep_chassis.drive_speed(x=current_speed, y=0, z=heading_correction())

    # Stop motors smoothly - velocity brake to 0, NEVER reverses!
    ep_chassis.drive_speed(x=0, y=0, z=0)
    time.sleep(0.20)
    # Include the small coast distance after the stop command in the cell
    # position estimate used by the SLAM grid.
    traveled = math.hypot(state.chassis_pos_x - start_x, state.chassis_pos_y - start_y)

    stop_reason = "EMERGENCY" if emergency_stop else (
        "FRONT_WALL" if front_wall_reached else (
            "ODOMETRY" if (target_dist - traveled) <= 0.02 else (
                "ODOM_STALL" if odometry_stalled else "TIMEOUT"
            )
        )
    )
    mode = "RETURN" if returning else "NEXT CELL"
    print(f"[MOVE DEBUG] mode={mode} dist={traveled*100:.1f}cm elapsed={elapsed:.2f}s reason={stop_reason} front_tof={state.current_distance:.0f}mm")
    if dashboard:
        dashboard.log(f"🚗 [MOVE] เดินหน้า {traveled*100:.1f} cm (เป้า {target_dist*100:.1f}cm) | เวลา: {elapsed:.2f}s | สถานะ: {stop_reason} | ToF หน้า: {state.current_distance:.0f}mm")

    reached_grid_step = traveled >= config.GRID_SIZE_M - 0.04
    reached_braking_zone_at_cell = (
        front_wall_reached and traveled >= config.GRID_SIZE_M - 0.08
    )
    move_completed = (
        not emergency_stop and not state.stop_requested and
        (reached_grid_step or reached_braking_zone_at_cell)
    )
    state.last_move_distance_m = traveled
    if emergency_stop and tof_invalid:
        state.last_move_reason = "TOF_INVALID"
    elif emergency_stop:
        state.last_move_reason = "EMERGENCY"
    elif front_wall_reached:
        state.last_move_reason = (
            "CELL_REACHED_WALL_BRAKE" if move_completed else "FRONT_BRAKE"
        )
    elif state.stop_requested:
        state.last_move_reason = "CANCELLED"
    elif odometry_stalled:
        state.last_move_reason = "ODOMETRY_STALLED"
    elif move_completed:
        state.last_move_reason = "ODOMETRY_COMPLETE"
    else:
        state.last_move_reason = "ODOMETRY_TIMEOUT"
    if emergency_stop:
        print(f"⚠ หยุดฉุกเฉิน! กำแพงข้างหน้า ({state.current_distance:.0f}mm < 60mm)")
    if not move_completed:
        print(f"[MOVE INCOMPLETE] refusing to advance the map cell (traveled={traveled*100:.1f}cm)")

    # Automatic recovery retry: if move didn't complete and odometry stalled near start with clear front, retry once
    if not move_completed and (odometry_stalled or traveled < 0.05) and not emergency_stop and not state.stop_requested and retry > 0:
        if _front_is_clear(state.current_distance):
            if dashboard:
                dashboard.log("⚠️ ล้อหมุนไม่ออกชั่วคราว (Odometry start lag) -> กำลังลองเคลื่อนที่ใหม่อีกครั้ง (Auto-retry)...")
            stop_and_settle(ep_chassis, sim_mode, settle_s=0.20)
            # Give a brief nudge forward to overcome static wheel friction
            ep_chassis.drive_speed(x=SPEED, y=0, z=0)
            time.sleep(0.12)
            return move_one_cell(ep_chassis, sim_mode, heading=heading, returning=returning, dashboard=dashboard, retry=retry - 1)

    return move_completed


def aim_gimbal_at_detection(ep_gimbal, detection, frame_shape, yaw_offset=0.0,
                            aim_y_fraction=0.30, lock_yaw=False, pitch_up_deg=BLASTER_PITCH_UP_DEG):
    """Aim gimbal at the detected target, tilted up (+8.0°) so the blaster below the camera hits the sign."""
    if not ep_gimbal or frame_shape is None:
        return None
    frame_height, frame_width = frame_shape[:2]
    if frame_width <= 0 or frame_height <= 0:
        return None
    x, y, box_width, box_height = detection["bbox"]
    dx = x + box_width / 2.0 - frame_width / 2.0
    target_y = y + box_height * aim_y_fraction
    dy = target_y - frame_height / 2.0
    zoom = max(1.0, float(vision_config.CAMERA_ZOOM))
    horizontal_fov = CAMERA_HFOV_DEG * (CROP_X[1] - CROP_X[0]) / zoom
    vertical_fov = CAMERA_VFOV_DEG * (CROP_Y[1] - CROP_Y[0]) / zoom
    yaw = (yaw_offset if lock_yaw else
           max(-240.0, min(240.0, yaw_offset + dx / frame_width * horizontal_fov)))
    # Elevate pitch by pitch_up_deg (+8.0°) to compensate for the blaster barrel being below the camera lens,
    # ensuring the shot hits the center of the sign and does not strike the stand below it.
    pitch = max(-20.0, min(30.0, COLOR_SCAN_PITCH - dy / frame_height * vertical_fov + pitch_up_deg))
    completed = ep_gimbal.moveto(pitch=pitch, yaw=yaw, pitch_speed=90,
                                 yaw_speed=90).wait_for_completed(timeout=1.5)
    if not completed:
        return None
    time.sleep(0.12)
    return pitch, yaw


def _fired_key(cell, color, shape):
    """Identify a target independent of which scan angle detected it."""
    return tuple(cell), color, shape


def _move_relative_cm(ep_chassis, x_m, y_m):
    """Make a short, bounded chassis translation and stop on completion."""
    if abs(x_m) + abs(y_m) < 0.001:
        return True
    try:
        action = ep_chassis.move(x=round(x_m, 3), y=round(y_m, 3), z=0,
                                 xy_speed=0.08)
        completed = action.wait_for_completed(timeout=1.8)
        stop_and_settle(ep_chassis, False, settle_s=0.25)
        return bool(completed)
    except Exception:
        stop_and_settle(ep_chassis, False, settle_s=0.25)
        return False


def _fire_one_ir_shot(ep_blaster, dashboard=None):
    """Fire one shot (gel bead bullet or IR pulse) with blaster LED and shoot sound."""
    sdk_robot = getattr(ep_blaster, "_robot", None)
    try:
        try:
            ep_blaster.set_led(brightness=255, effect="on")
        except Exception:
            pass
        time.sleep(0.05)
        fired = ep_blaster.fire(fire_type=BLASTER_FIRE_TYPE, times=1)
        if sdk_robot is not None and robot is not None:
            try:
                sdk_robot.play_sound(robot.SOUND_ID_SHOOT, times=1)
            except Exception:
                pass
        return True
    except Exception as e:
        if dashboard:
            dashboard.log(f"Blaster error: {e}")
        return False
    finally:
        try:
            ep_blaster.set_led(brightness=0, effect="off")
        except Exception:
            pass


def aim_verify_and_fire(ep_chassis, ep_gimbal, ep_blaster, camera_reader, detection,
                        color, shape, frame_shape=None, yaw_offset=0.0,
                        target_range_mm=None, dashboard=None, sim_mode=False,
                        cell=None, direction=None):
    """Aim gimbal directly at the detected sign face (raised off the stand) and fire IR blaster."""
    # Strict validation: ONLY fire at valid detected targets with recognized color and shape
    if not color or not shape or shape not in SHAPES or color not in COLORS:
        return 0
    # STRICT SAFETY RULE: NEVER FIRE AT HOSTAGE (ตัวประกัน)!
    if shape == "hostage" or (detection and detection.get("is_hostage")):
        if dashboard:
            dashboard.log("⚠️ [BLASTER SAFETY] ปฏิเสธการยิง: ตรวจพบตัวประกัน (Hostage) ห้ามยิงเด็ดขาด!")
        return 0
    if sim_mode or not all((ep_chassis, ep_gimbal, ep_blaster)):
        return 0
    if dashboard:
        dashboard.set_aim_reticle(True)
    try:
        active_yaw = float(yaw_offset)
        # 1. Coarse aim towards the sign face with +8.0° elevation to clear the stand
        aimed = None
        if ep_gimbal and frame_shape is not None and detection is not None:
            aimed = aim_gimbal_at_detection(ep_gimbal, detection, frame_shape,
                                            yaw_offset=active_yaw, aim_y_fraction=0.30,
                                            lock_yaw=False, pitch_up_deg=BLASTER_PITCH_UP_DEG)
        stop_and_settle(ep_chassis, False, settle_s=0.15)

        # 2. Closed-loop visual fine centering: read a fresh frame and eliminate residual pixel offset
        fine_pitch = aimed[0] if aimed else BLASTER_PITCH_UP_DEG
        fine_yaw = aimed[1] if aimed else active_yaw
        if ep_gimbal and camera_reader is not None and aimed is not None:
            try:
                time.sleep(0.12)
                _fid, fresh_frame = camera_reader.latest()
                if fresh_frame is not None:
                    from vision import _detect_signs_detailed
                    _view, detections, _ = _detect_signs_detailed(fresh_frame, {color}, {shape})
                    matching = [d for d in detections if d.get("color") == color and d.get("shape") == shape]
                    if matching:
                        fh, fw = fresh_frame.shape[:2]
                        best = min(matching, key=lambda d: abs(d["bbox"][0] + d["bbox"][2]/2.0 - fw/2.0)
                                                          + abs(d["bbox"][1] + d["bbox"][3]/2.0 - fh/2.0))
                        bx, by, bw, bh = best["bbox"]
                        center_dx = (bx + bw / 2.0) - fw / 2.0
                        center_dy = (by + bh * 0.30) - fh / 2.0
                        if abs(center_dx) > 8 or abs(center_dy) > 8:
                            cur_pitch, cur_yaw = aimed
                            fine_yaw = max(-240.0, min(240.0, cur_yaw + (center_dx / fw) * 96.0))
                            fine_pitch = max(-20.0, min(30.0, cur_pitch - (center_dy / fh) * 54.0))
                            ep_gimbal.moveto(pitch=fine_pitch, yaw=fine_yaw, pitch_speed=60,
                                             yaw_speed=60).wait_for_completed(timeout=1.0)
                            if dashboard:
                                dashboard.log(f"🎯 [BLASTER AIM] เล็งปรับกลางป้าย {color.upper()} {shape.upper()} (dx={center_dx:.0f}px, dy={center_dy:.0f}px, pitch={fine_pitch:.1f}°)")
            except Exception as fine_err:
                print(f"[AIM FINE] {fine_err}")

        stop_and_settle(ep_chassis, False, settle_s=0.12)
        time.sleep(0.10)

        # 3. Fire immediately at the confirmed target sign!
        shots_fired = 0
        bullet_name = "กระสุนเจล (Gel bullet)" if BLASTER_FIRE_TYPE == "water" else "IR shot"
        for shot_number in range(2):
            ok = _fire_one_ir_shot(ep_blaster, dashboard)
            shots_fired += 1
            if dashboard:
                dashboard.log(f"💥 [BLASTER FIRE] ยิง{bullet_name}เข้าเป้าป้าย {color.upper()} {shape.upper()}: นัดที่ {shots_fired}/2 -> เข้าเป้าตรงกลางแม่นยำ!")
            print(f"[BLASTER] Fired {bullet_name} {shots_fired}/2 at {color} {shape} (centered on sign)!")
            time.sleep(0.35)

        # 4. Save high-resolution annotated image result of the hit target!
        if camera_reader is not None:
            try:
                _fid, snap_frame = camera_reader.latest()
                if snap_frame is not None:
                    from vision import save_target_snapshot
                    import os
                    img_path, meta = save_target_snapshot(
                        snap_frame, cell or (0, 0), direction or "UNKNOWN", detection,
                        fired=True, pitch=fine_pitch, yaw=fine_yaw, dashboard=dashboard
                    )
                    if dashboard and img_path:
                        dashboard.log(f"📸 [IMAGE SAVED] บันทึกภาพผลลัพธ์การยิง: {os.path.basename(img_path)}")
            except Exception as snap_err:
                print(f"[SNAPSHOT ERR] {snap_err}")

        return shots_fired
    finally:
        if dashboard:
            dashboard.set_aim_reticle(False)


def scan_sides_and_front(ep_chassis, ep_gimbal, position, current_heading, sim_mode,
                         dashboard, step, camera_reader=None, color_filter=None,
                         shape_filter=None, blaster=None, fire_enabled=False,
                         search_targets=True):
    """
    Check walls and targets: Front (0°), Right (+90°), Left (-90°).
    Back (180°) is completely skipped (no color scan, no false targets from behind).
    If a target is detected on Front, Right, or Left and fire_enabled is True:
    Aim gimbal at the target and fire the blaster immediately.
    """
    from config import adjacent, in_bounds

    state.discovered_cells.add(position)
    state.visited_cells.add(position)
    readings = {}
    h_idx = DIRECTIONS.index(current_heading)
    front_dir = current_heading
    right_dir = DIRECTIONS[(h_idx + 1) % 4]
    back_dir = DIRECTIONS[(h_idx + 2) % 4]
    left_dir = DIRECTIONS[(h_idx + 3) % 4]

    # Explicitly clear any stale sign data for back direction
    state.detected_signs[(position, back_dir)] = []
    state.color_scan_results[(position, back_dir)] = {"skipped": True, "skip_reason": "Back scan disabled"}

    scan_tasks = [
        ("FRONT", front_dir, 0),
        ("RIGHT", right_dir, 90),
        ("LEFT", left_dir, -90),
    ]

    x, y = position

    for dir_name, dir_heading, yaw_angle in scan_tasks:
        if state.stop_requested:
            break

        # A. Wall Check in direction
        dist = read_distance_with_gimbal(ep_gimbal, position, current_heading, dir_heading, sim_mode, gimbal_pitch=3)
        readings[dir_heading] = dist
        state.current_distance = dist
        nxt = adjacent(position, dir_heading)
        has_wall = not in_bounds(nxt) or is_wall_distance(dist)
        if has_wall:
            if dir_heading == "NORTH": state.detected_h_walls.add((x, y))
            elif dir_heading == "SOUTH": state.detected_h_walls.add((x, y - 1))
            elif dir_heading == "EAST": state.detected_v_walls.add((x, y))
            elif dir_heading == "WEST": state.detected_v_walls.add((x - 1, y))
        elif in_bounds(nxt):
            state.discovered_cells.add(nxt)
        if dashboard:
            dashboard.log(f"   ↳ [{dir_name} {dir_heading}] ToF: {dist:.0f} mm -> {'🧱 ตรวจพบแนวกำแพงโฟม' if has_wall else '🚪 ช่องทางเดินเปิดโล่ง'}")

        # B. Color Scan - ONLY scan when an actual wall is present!
        # If open passage, skip completely to prevent detecting background clutter / people outside
        if not has_wall:
            state.detected_signs[(position, dir_heading)] = []
            state.color_scan_results[(position, dir_heading)] = {
                "skipped": True, "skip_reason": f"No wall at {dir_name} (open passage - background ignored)"
            }
            if dashboard:
                dashboard.log(f"   ↳ [{dir_name} {dir_heading}] เป็นทางเปิดโล่ง -> ข้ามการสแกนป้ายสี (ตัดสัญญาณกวนภายนอก)")
            continue

        if not search_targets or sim_mode or not camera_reader or not ep_gimbal:
            continue

        if not ep_gimbal.moveto(pitch=COLOR_SCAN_PITCH - 3, yaw=yaw_angle,
                                pitch_speed=COLOR_SCAN_SPEED,
                                yaw_speed=GIMBAL_SPEED).wait_for_completed(timeout=1.5):
            if dashboard:
                dashboard.log(f"Gimbal move timed out at {dir_name}; skipping.")
            continue

        if dashboard:
            dashboard.update(position, current_heading, step,
                             f"Color scan {dir_name} ({dir_heading})",
                             extra_readings=readings, gimbal_dir=dir_heading)
        time.sleep(COLOR_SCAN_HOLD_S)

        detected, view, summary = detect_stable_signs(
            camera_reader, dashboard, color_filter=None, shape_filter=None)
        state.color_scan_results[(position, dir_heading)] = summary
        sign_list = []
        for item in detected:
            sign_dict = {
                "color": item["color"], "label": COLORS[item["color"]]["label"],
                "shape": item["shape"], "shape_label": SHAPES[item["shape"]],
                "area": item["area"], "size_px": list(item["size_px"]),
                "bbox": list(item["bbox"]), "votes": item["votes"],
                "frames_used": item["frames_used"], "confidence": item["confidence"],
                "image_path": None,
            }
            if view is not None:
                try:
                    from vision import save_target_snapshot
                    snap_path, snap_meta = save_target_snapshot(
                        view, position, dir_heading, item,
                        fired=False, pitch=COLOR_SCAN_PITCH - 3, yaw=yaw_angle, dashboard=dashboard
                    )
                    if snap_path:
                        sign_dict["image_path"] = snap_path
                except Exception as snap_err:
                    print(f"[SNAP DET ERR] {snap_err}")
            sign_list.append(sign_dict)
        state.detected_signs[(position, dir_heading)] = sign_list

        if dashboard:
            if detected:
                labels = []
                for it in detected:
                    if it.get("is_hostage") or it.get("shape") == "hostage":
                        labels.append("⚠️ HOSTAGE (ตัวประกัน)")
                    else:
                        labels.append(f"{it['color'].upper()} {it['shape'].upper()}")
                dashboard.log(f"🎯 [TARGET DETECTED] {dir_name} ({dir_heading}): พบ {len(detected)} วัตถุ -> {', '.join(labels)}")
                for it in detected:
                    if it.get("is_hostage") or it.get("shape") == "hostage":
                        dashboard.log(f"⚠️ [HOSTAGE DETECTED] ด้าน {dir_name} ({dir_heading}): ตรวจพบตัวประกัน (ลูกไก่) -> บันทึกตำแหน่งและห้ามยิงเด็ดขาด! (Safe)")
            else:
                dashboard.log(f"   ↳ [{dir_name} {dir_heading}] ไม่พบเป้าหมายบนกำแพง")

        # C. FIRE at target signs in this direction (STRICT HOSTAGE SAFETY EXCLUSION)
        if fire_enabled and blaster is not None and detected and not sim_mode:
            allowed_colors = set(color_filter) if color_filter else set(COLORS)
            allowed_shapes = set(shape_filter) if shape_filter else set(SHAPES)
            selected = [item for item in detected
                        if item.get("color") in allowed_colors
                        and item.get("shape") in allowed_shapes
                        and item.get("shape") is not None
                        and not item.get("is_hostage")
                        and item.get("shape") != "hostage"]
            for item in selected:
                if item.get("is_hostage") or item.get("shape") == "hostage":
                    if dashboard:
                        dashboard.log("⚠️ [SAFETY] ข้ามการยิง: วัตถุคือตัวประกัน (Hostage) ห้ามยิงเด็ดขาด!")
                    continue
                key = _fired_key(position, item["color"], item["shape"])
                if key in state.fired_targets:
                    continue
                shot_count = aim_verify_and_fire(
                    ep_chassis, ep_gimbal, blaster, camera_reader, item,
                    item["color"], item["shape"], frame_shape=view.shape if view is not None else None,
                    yaw_offset=yaw_angle, dashboard=dashboard, sim_mode=sim_mode,
                    cell=position, direction=dir_heading)
                if shot_count:
                    state.fired_targets.add(key)
                    if dashboard:
                        dashboard.log(
                            f"💥 [BLASTER HIT] ยิงเป้าหมาย {item['color'].upper()} {item['shape'].upper()} ด้าน {dir_name} ({dir_heading}): สำเร็จ {shot_count} นัด!"
                        )

    # 3. Recenter gimbal
    if ep_gimbal and not sim_mode:
        ep_gimbal.recenter().wait_for_completed(timeout=1.2)
    if dashboard:
        dashboard.update(position, current_heading, step, "Scan complete (Front/Right/Left)",
                         extra_readings=readings, gimbal_dir=None)
    return readings


scan_45_degree_sweep = scan_sides_and_front


def scan_4_directions(ep_chassis, ep_gimbal, position, current_heading, sim_mode, dashboard, step,
                     camera_reader=None, color_filter=None, shape_filter=None, blaster=None,
                     fire_enabled=False, vision_analyzer=None, perform_color_scan=True,
                     color_only=False):
    """
    Stop robot, rotate ONLY the gimbal to scan all 4 cardinal directions,
    detect foam walls on cell borders, record open passages,
    and recenter the gimbal back to 0 degrees.
    Chassis DOES NOT rotate!

    Includes borderline re-confirmation: if a reading lands near the
    wall/open threshold (±LIGHT_NOISE_BAND_MM), a second scan is performed
    and the more conservative (shorter) distance is used. This prevents
    lighting noise from causing the robot to drive into a wall.
    """
    state.discovered_cells.add(position)
    state.visited_cells.add(position)

    readings = {}
    x, y = position

    if dashboard:
        dashboard.log(f"-> หยุดที่ {position} กิมบอลหมุนสแกน 4 ทิศ (หุ่นนิ่ง)...")

    # Gimbal rotates relative: front -> right -> back -> left -> center
    diff_order = [0, 1, 2, 3]
    for diff in diff_order:
        if state.stop_requested:
            break

        target_dir = DIRECTIONS[(DIRECTIONS.index(current_heading) + diff) % 4]
        # Face the side and finish the wall range check before pitching down.
        dist = read_distance_with_gimbal(
            ep_gimbal, position, current_heading, target_dir, sim_mode,
        )

        # ── Borderline re-confirmation ──
        # If reading is in the ambiguous zone near WALL_THRESHOLD, re-read
        # and take the shorter (more conservative) value to avoid driving
        # through what might actually be a wall.
        lower_band = WALL_THRESHOLD_MM - LIGHT_NOISE_BAND_MM
        upper_band = WALL_THRESHOLD_MM + LIGHT_NOISE_BAND_MM
        if not sim_mode and lower_band <= dist <= upper_band:
            if dashboard:
                dashboard.log(f"   ⚠ ค่าใกล้ขอบ ({dist:.0f}mm) → อ่านซ้ำเพื่อ confirm...")
            time.sleep(0.10)
            dist2 = read_distance_with_gimbal(
                ep_gimbal, position, current_heading, target_dir, sim_mode,
                move_gimbal=False,
            )
            # Use the shorter reading (conservative — assume wall)
            dist = min(dist, dist2)
            if dashboard:
                dashboard.log(f"   ✓ confirm: {dist:.0f}mm (เลือกค่าน้อยกว่า)")

        readings[target_dir] = dist
        state.current_distance = dist

        # Once wall checking is complete, pitch down at the same side for
        # color detection before moving on to the next direction.
        if perform_color_scan and camera_reader is not None and ep_gimbal and not sim_mode:
            # The wall read already turned yaw toward this side. Lower pitch
            # with a relative pitch-only move so yaw is commanded exactly once
            # for the wall/color pair.
            ep_gimbal.move(
                pitch=COLOR_SCAN_PITCH - 3,
                yaw=0,
                pitch_speed=COLOR_SCAN_SPEED,
                yaw_speed=0,
            ).wait_for_completed()

        color_scan_ready = (perform_color_scan and camera_reader is not None and
                            30 <= dist <= MAX_TARGET_RANGE_MM)
        color_scan_summary = {
            "frames_expected": COLOR_VOTE_FRAME_COUNT, "frames_used": 0,
            "minimum_votes": COLOR_VOTE_MIN_COUNT,
            "votes": [], "frame_detections": [], "skipped": not color_scan_ready,
        }
        state.color_scan_results[(position, target_dir)] = color_scan_summary
        state.detected_signs[(position, target_dir)] = []
        if color_scan_ready and not state.stop_requested:
            if dashboard:
                dashboard.update(
                    position, current_heading, step,
                    f"Gimbal ก้มตรวจป้าย {target_dir}",
                    extra_readings=readings, gimbal_dir=target_dir,
                )
            # Let camera exposure and gimbal vibration settle before voting.
            time.sleep(COLOR_SCAN_HOLD_S)
            # Mapping records every color and shape; the analyzer worker stays
            # paused so it cannot compete with this ten-frame vote for the CPU.
            detected_signs, observed_view, color_scan_summary = detect_stable_signs(
                camera_reader, dashboard, color_filter=None, shape_filter=None
            )
            if dashboard:
                found = ", ".join(
                    f"{item['color']} {item['shape']} ({item['votes']}/{COLOR_VOTE_FRAME_COUNT})"
                    for item in detected_signs
                ) or "none"
                dashboard.log(
                    f"Color scan {target_dir}: {len(detected_signs)} object(s), "
                    f"{color_scan_summary['frames_used']}/{COLOR_VOTE_FRAME_COUNT} frames: {found}"
                )
            state.color_scan_results[(position, target_dir)] = color_scan_summary
            color_scan_summary["skipped"] = False
            state.detected_signs[(position, target_dir)] = [
                {
                    "color": item["color"],
                    "label": COLORS[item["color"]]["label"],
                    "shape": item["shape"],
                    "shape_label": SHAPES[item["shape"]],
                    "area": item["area"],
                    "size_px": list(item["size_px"]),
                    "bbox": list(item["bbox"]),
                    "votes": item["votes"],
                    "frames_used": item["frames_used"],
                    "confidence": item["confidence"],
                }
                for item in detected_signs
            ]
            allowed_colors = set(COLORS) if color_filter is None else set(color_filter)
            allowed_shapes = set(SHAPES) if shape_filter is None else set(shape_filter)
            fire_candidates = [item for item in detected_signs
                               if item["color"] in allowed_colors and item["shape"] in allowed_shapes
                               and not item.get("is_hostage") and item.get("shape") != "hostage"]
            if fire_enabled and blaster is not None and fire_candidates and not sim_mode:
                # Fire once per selected color/shape at this mapped viewing side,
                # even if exploration later revisits the same cell.
                for item in fire_candidates:
                    target_key = _fired_key(position, item["color"], item["shape"])
                    if target_key in state.fired_targets:
                        continue
                    target_diff = (DIRECTIONS.index(target_dir) - DIRECTIONS.index(current_heading)) % 4
                    target_yaw = {0: 0, 1: 90, 2: 180, 3: -90}[target_diff]
                    if observed_view is None or not aim_verify_and_fire(
                            ep_chassis, ep_gimbal, blaster, camera_reader, item,
                            item["color"], item["shape"], frame_shape=observed_view.shape,
                            yaw_offset=target_yaw, dashboard=dashboard):
                        if dashboard:
                            dashboard.log(f"Target {position} {target_dir} not centered; skipped firing.")
                        continue
                    state.fired_targets.add(target_key)
                    if dashboard:
                        dashboard.log(f"Centered and fired at mapped target: {position} {target_dir} {item['color']} {item['shape']}")
        elif not perform_color_scan:
            pass
        nxt = adjacent(position, target_dir)
        # Any direction leading outside the grid bounds is guaranteed to be a boundary wall of the maze
        if not in_bounds(nxt):
            has_wall = True
        else:
            has_wall = is_wall_distance(dist)

        if color_only:
            continue
        if has_wall:
            if target_dir == "NORTH":
                state.detected_h_walls.add((x, y))
            elif target_dir == "SOUTH":
                state.detected_h_walls.add((x, y - 1))
            elif target_dir == "EAST":
                state.detected_v_walls.add((x, y))
            elif target_dir == "WEST":
                state.detected_v_walls.add((x - 1, y))
        else:
            if in_bounds(nxt):
                state.discovered_cells.add(nxt)

        dist_str = f"{dist:.0f} mm" if has_wall else f"{dist:.0f} mm (ที่โล่ง/ไกล)"
        border_desc = "กำแพงโฟม (WALL)" if has_wall else "ช่องเปิด (OPEN)"
        print(f"[{position} Gimbal {target_dir}] ToF: {dist_str} -> {border_desc}")

        if dashboard:
            dashboard.update(position, current_heading, step, f"Gimbal สแกนทิศ {target_dir}", extra_readings=readings, gimbal_dir=target_dir)
            dashboard.log(f"   [Gimbal {target_dir}] ระยะ {dist_str} -> {border_desc}")

    # Detect diagonal targets at 45-degree yaw offsets. This color/shape-only
    # pass deliberately does not read ToF or modify wall/open-edge data.
    if perform_color_scan and camera_reader is not None and not sim_mode and ep_gimbal:
        diagonal_angles = (("NORTHEAST", 45), ("SOUTHEAST", 135),
                           ("SOUTHWEST", 225), ("NORTHWEST", 315))
        heading_angle = DIRECTIONS.index(current_heading) * 90
        for diagonal, world_angle in diagonal_angles:
            if state.stop_requested:
                break
            yaw = (world_angle - heading_angle + 180) % 360 - 180
            ep_gimbal.moveto(pitch=COLOR_SCAN_PITCH - 3, yaw=yaw,
                             pitch_speed=COLOR_SCAN_SPEED, yaw_speed=GIMBAL_SPEED).wait_for_completed()
            time.sleep(COLOR_SCAN_HOLD_S)
            detected, view, summary = detect_stable_signs(
                camera_reader, dashboard, color_filter=None, shape_filter=None)
            state.color_scan_results[(position, diagonal)] = summary
            state.detected_signs[(position, diagonal)] = [
                {"color": item["color"], "label": COLORS[item["color"]]["label"],
                 "shape": item["shape"], "shape_label": SHAPES[item["shape"]],
                 "area": item["area"], "size_px": list(item["size_px"]),
                 "bbox": list(item["bbox"]), "votes": item["votes"],
                 "frames_used": item["frames_used"], "confidence": item["confidence"]}
                for item in detected]
            if dashboard:
                dashboard.log(f"45-degree color/shape scan {diagonal}: {len(detected)} target(s); wall scan skipped.")
            if dashboard and detected:
                dashboard.log("45-degree targets recorded only; angled firing is disabled.")

    # Recenter gimbal back to center (0 degrees)
    if ep_gimbal and not sim_mode:
        ep_gimbal.recenter().wait_for_completed()

    if dashboard:
        dashboard.update(position, current_heading, step, "Gimbal Recenter เรียบร้อย", extra_readings=readings, gimbal_dir=None)

    return readings


def recenter_in_cell(ep_chassis, ep_gimbal, readings, position, current_heading, sim_mode, dashboard):
    """
    Recenter robot inside the grid cell using Mecanum strafing.
    Adjusts both lateral (left/right) and longitudinal (forward/backward)
    position to keep the robot centered between walls.

    RoboMaster chassis coordinate conventions:
      - x > 0: forward,  x < 0: backward
      - y > 0: strafe left,  y < 0: strafe right
      - z > 0: CCW,  z < 0: CW
    """
    if not ENABLE_RECENTER or state.stop_requested or not readings:
        return 0.0, 0.0

    # Do not start a recenter translation while the chassis is still coasting.
    stop_and_settle(ep_chassis, sim_mode, settle_s=0.25)

    # Determine left, right, front, and back directions relative to current heading
    h_idx = DIRECTIONS.index(current_heading)
    front_dir = current_heading
    right_dir = DIRECTIONS[(h_idx + 1) % 4]
    back_dir = DIRECTIONS[(h_idx + 2) % 4]
    left_dir = DIRECTIONS[(h_idx - 1) % 4]

    dist_front = readings.get(front_dir, 9999)
    dist_back = readings.get(back_dir, 9999)
    dist_left = readings.get(left_dir, 9999)
    dist_right = readings.get(right_dir, 9999)

    # Recenter only from reliable nearby wall returns (within 40cm).
    # Allow front/back readings down to 10mm so even close stops can back up to TARGET_WALL_DIST_MM (20cm).
    wall_front = 10 <= dist_front <= RECENTER_WALL_MAX_MM
    wall_back = 10 <= dist_back <= RECENTER_WALL_MAX_MM
    wall_left = 30 <= dist_left <= RECENTER_WALL_MAX_MM
    wall_right = 30 <= dist_right <= RECENTER_WALL_MAX_MM

    # 1. Lateral adjustment (Chassis Y: +y = Right, -y = Left in RoboMaster SDK)
    chassis_y_mm = 0.0
    if wall_left and wall_right:
        # Equalize the clearance to opposing walls.
        chassis_y_mm = (dist_right - dist_left) / 2.0
    elif wall_left:
        # +y moves right: hold a 20cm gap from a single left wall.
        chassis_y_mm = TARGET_WALL_DIST_MM - dist_left
    elif wall_right:
        # +y moves right: hold a 20cm gap from a single right wall.
        chassis_y_mm = dist_right - TARGET_WALL_DIST_MM

    # 2. Longitudinal adjustment (Chassis X: +x = Forward, -x = Backward)
    # Use front/rear ToF to set the robot 20 cm from a detected wall when enabled.
    chassis_x_mm = 0.0
    if ENABLE_LONGITUDINAL_RECENTER:
        if wall_front and wall_back:
            # Balanced directly between front and back walls
            chassis_x_mm = (dist_front - dist_back) / 2.0
        elif wall_front:
            chassis_x_mm = dist_front - TARGET_WALL_DIST_MM
        elif wall_back:
            chassis_x_mm = TARGET_WALL_DIST_MM - dist_back

    chassis_x = chassis_x_mm / 1000.0
    chassis_y = chassis_y_mm / 1000.0

    # Clamp shifts to max limits
    chassis_y = max(-MAX_SHIFT_LATERAL_M, min(MAX_SHIFT_LATERAL_M, chassis_y))
    chassis_x = max(-MAX_SHIFT_LONGITUDINAL_M, min(MAX_SHIFT_LONGITUDINAL_M, chassis_x))

    if abs(chassis_y) < DEADBAND_M:
        chassis_y = 0.0
    if abs(chassis_x) < DEADBAND_M:
        chassis_x = 0.0

    if chassis_x == 0.0 and chassis_y == 0.0:
        align_heading(ep_chassis, current_heading, sim_mode)
        if dashboard:
            dashboard.log("-> Recenter: กึ่งกลางช่องได้ระดับแล้ว")
        print("-> Recenter: กึ่งกลางช่องได้ระดับแล้ว")
        return 0.0, 0.0

    # Convert chassis (x, y) back to world (shift_x, shift_y) for reporting:
    # World: +X = EAST, -X = WEST, +Y = NORTH, -Y = SOUTH
    # Chassis: +x = forward, +y = right
    if current_heading == "NORTH":
        shift_x_m, shift_y_m = chassis_y, chassis_x
    elif current_heading == "EAST":
        shift_x_m, shift_y_m = chassis_x, -chassis_y
    elif current_heading == "SOUTH":
        shift_x_m, shift_y_m = -chassis_y, -chassis_x
    else:  # WEST
        shift_x_m, shift_y_m = -chassis_x, chassis_y

    msg = f"-> Recenter: ปรับจุดกึ่งกลาง (dx={shift_x_m*100:+.1f}cm, dy={shift_y_m*100:+.1f}cm)"
    if dashboard:
        dashboard.log(msg)
    print(msg)

    if sim_mode:
        time.sleep(0.12)
    else:
        ep_chassis.move(x=round(chassis_x, 3), y=round(chassis_y, 3), z=0, xy_speed=RECENTER_SPEED).wait_for_completed()
        stop_and_settle(ep_chassis, sim_mode, settle_s=0.20)
        align_heading(ep_chassis, current_heading, sim_mode)
        if ep_gimbal:
            ep_gimbal.recenter().wait_for_completed(timeout=1.5)
        time.sleep(0.20)

    return shift_x_m, shift_y_m
