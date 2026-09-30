# -*- coding: utf-8 -*-
"""
Assignment 2: RoboMaster Autonomous SLAM in Foam Wall Maze.
Main entry point orchestrating exploration, sensor fusion, path planning,
and graphical visualization.

Usage:
    python slam.py --sim
    python slam.py
    python slam.py --sim --no-dashboard --start-x 1 --start-y 1 --start-heading NORTH
"""

from config import GRID_SIZE_M
import argparse
from collections import deque
import json
import math
import os
import threading
import time
import tkinter as tk

import config
from config import (DIRECTIONS, GIMBAL_SPEED, GRID_H, GRID_SIZE_M, GRID_W,
                    SAFE_FRONT_DIST_MM, adjacent, in_bounds)
from dashboard import Dashboard
from evaluation import save_outputs
from navigation import find_path_to_nearest_unvisited, astar_path
from navigation import is_wall_between
from robot_control import (
    align_heading,
    aim_verify_and_fire,
    move_one_cell,
    read_distance_with_gimbal,
    recenter_in_cell,
    reset_sensor_filters,
    robot,
    scan_45_degree_sweep,
    read_target_range_at_yaw,
    stop_and_settle,
    sub_attitude_handler,
    sub_ir_handler,
    sub_position_handler,
    sub_tof_handler,
    turn_to_direction,
)
import state
from vision import (COLOR_SCAN_PITCH, COLOR_SCAN_SPEED, COLORS, SHAPES,
                    CameraFrameReader, VisionFrameAnalyzer, detect_stable_signs)

MAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps")
MISSION_MAP_PATH = os.path.join(MAP_DIR, "mission_map.json")
LEGACY_MISSION_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mission_map.json")


def _front_is_clear(distance):
    return math.isfinite(float(distance)) and float(distance) > SAFE_FRONT_DIST_MM


def _direction_yaw(heading, target_direction):
    """Return relative gimbal yaw for a cardinal or diagonal map direction."""
    world_angles = {"NORTH": 0, "NORTHEAST": 45, "EAST": 90,
                    "SOUTHEAST": 135, "SOUTH": 180, "SOUTHWEST": 225,
                    "WEST": 270, "NORTHWEST": 315}
    heading_angle = world_angles[heading]
    target_angle = world_angles[target_direction]
    return (target_angle - heading_angle + 180) % 360 - 180


def _record_pose(step, position, heading):
    """Record a confirmed robot position for the round trajectory image."""
    previous = state.trajectory[-1]["cell"] if state.trajectory else None
    state.trajectory.append({
        "step": step, "timestamp": round(time.time(), 2),
        "grid_x": position[0], "grid_y": position[1],
        "x_m": round((position[0] - 0.5) * config.GRID_SIZE_M, 3),
        "y_m": round((position[1] - 0.5) * config.GRID_SIZE_M, 3),
        "heading": heading, "tof_mm": state.current_distance,
        "cell": position, "previous": previous,
    })


def clean_old_results():
    """
    Clear old logs, map images, CSVs, and target photos before starting a new mapping run.
    Ensures a fresh, clean slate for Round 1 mapping without clutter from previous sessions.
    NOTE: Called ONLY when starting 'mapping' mode, NEVER when starting 'targets' mode (Round 2).
    """
    import glob
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    targets_dir = os.path.join(results_dir, "targets")
    os.makedirs(targets_dir, exist_ok=True)

    # 1. Clear target snapshot images
    for p in glob.glob(os.path.join(targets_dir, "*.*")):
        try:
            os.remove(p)
        except Exception:
            pass

    # 2. Clear CSVs, TXT logs, and PNGs in results/
    for ext in ("*.csv", "*.txt", "*.png"):
        for p in glob.glob(os.path.join(results_dir, ext)):
            try:
                os.remove(p)
            except Exception:
                pass

    # 3. Clear root-level result images and leftover CSVs if any
    code_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in ("final_slam_map.png", "robot_trajectory.png", "robot_trajectory_round1.png",
                  "robot_trajectory_round2.png", "slam_map.png", "exploration_log.csv",
                  "signs_data.csv", "trajectory_log.csv", "trajectory_log_round1.csv",
                  "trajectory_log_round2.csv", "visited_cells.csv", "wall_data.csv"):
        p = os.path.join(code_dir, fname)
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    # 4. Clear maps/mission_map.json
    if os.path.exists(MISSION_MAP_PATH):
        try:
            os.remove(MISSION_MAP_PATH)
        except Exception:
            pass
    if os.path.exists(LEGACY_MISSION_MAP_PATH):
        try:
            os.remove(LEGACY_MISSION_MAP_PATH)
        except Exception:
            pass
    print("🧹 [CLEANUP] เคลียร์ไฟล์และข้อมูลของรอบเก่าทั้งหมดเรียบร้อยแล้ว (Cleaned old session data)")


def _save_mission_map(start_config):
    payload = {
        "version": 3,
        "grid_size_m": config.GRID_SIZE_M,
        "grid_width": config.GRID_W,
        "grid_height": config.GRID_H,
        "start": list(start_config),
        "h_walls": sorted([list(edge) for edge in state.detected_h_walls]),
        "v_walls": sorted([list(edge) for edge in state.detected_v_walls]),
        "discovered": sorted([list(cell) for cell in state.discovered_cells]),
        "visited": sorted([list(cell) for cell in state.visited_cells]),
        "targets": [
            {"cell": list(cell), "direction": direction, "signs": signs}
            for (cell, direction), signs in state.detected_signs.items() if signs
        ],
        "color_scans": [
            {"cell": list(cell), "direction": direction, **scan}
            for (cell, direction), scan in state.color_scan_results.items()
        ],
    }
    os.makedirs(MAP_DIR, exist_ok=True)
    temp_path = MISSION_MAP_PATH + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
    os.replace(temp_path, MISSION_MAP_PATH)


def _load_mission_map():
    map_path = MISSION_MAP_PATH if os.path.exists(MISSION_MAP_PATH) else LEGACY_MISSION_MAP_PATH
    with open(map_path, "r", encoding="utf-8") as source:
        data = json.load(source)
    gw = int(data.get("grid_width", config.GRID_W))
    gh = int(data.get("grid_height", config.GRID_H))
    cs = float(data.get("grid_size_m", config.GRID_SIZE_M))
    config.set_grid_dimensions(gw, gh, cs)
    return data


def _shortest_cell_path(start, goal, current_heading=None):
    """A* with turn penalty สำหรับหาเส้นทางสั้นที่สุดระหว่างสองจุด"""
    return astar_path(start, goal, current_heading=current_heading)


def _run_target_round(ep_chassis, ep_gimbal, ep_blaster, camera_reader, start_pos,
                      heading, sim_mode, dashboard, step, color_filter, shape_filter,
                      vision_analyzer=None, deadline=None):
    """Greedy nearest-target ordering with A* shortest paths (turn penalty) on the saved SLAM map."""
    color_filter = color_filter if color_filter else set(COLORS)
    shape_filter = shape_filter if shape_filter else set(SHAPES)
    targets = []
    seen_signs = set()
    for (cell, direction), signs in state.detected_signs.items():
        selected = [sign for sign in signs
                    if (not color_filter or sign["color"] in color_filter)
                    and (not shape_filter or sign["shape"] in shape_filter)
                    and not sign.get("is_hostage")
                    and sign.get("shape") != "hostage"]
        unique = []
        for sign in selected:
            key = (tuple(cell), sign["color"], sign["shape"])
            if key not in seen_signs:
                seen_signs.add(key)
                unique.append(sign)
        if unique:
            targets.append({"cell": tuple(cell), "direction": direction, "signs": unique})
    position = start_pos
    _record_pose(step, position, heading)
    align_heading(ep_chassis, heading, sim_mode)
    if not targets:
        if dashboard:
            dashboard.log("⚠️ ไม่พบเป้าหมายที่ตรวจจับได้จากรอบแรก หรือยิงครบหมดแล้ว")
        return position, heading
    if dashboard:
        dashboard.log(f"🎯 [RUN 2 TARGETS] พบเป้าหมายจากรอบแรกทั้งหมด {len(targets)} จุด พร้อมนำทางไปยิงทีละเป้าหมาย!")
    while targets and not state.stop_requested and (deadline is None or time.monotonic() < deadline):
        targets = [target for target in targets
                   if any((tuple(target["cell"]), sign["color"], sign["shape"])
                          not in state.fired_targets for sign in target["signs"])]
        if not targets:
            break
        routes = [(_shortest_cell_path(position, target["cell"], current_heading=heading), target) for target in targets]
        routes = [(path, target) for path, target in routes if path]
        if not routes:
            if dashboard:
                dashboard.log("Remaining targets are unreachable through saved open passages.")
            break
        path, target = min(routes, key=lambda pair: (len(pair[0]), pair[1]["cell"], pair[1]["direction"]))
        target_cell = target["cell"]
        target_dir = target["direction"]
        signs_desc = ", ".join(f"{s['color'].upper()} {s['shape'].upper()}" for s in target["signs"])
        path_str = " ➔ ".join(str(c) for c in path)
        if dashboard:
            dashboard.log(f"⭐ [A* PATH] คำนวณเส้นทาง A* ที่ดีที่สุด (ระยะทางสั้นสุด + Turn Penalty):")
            dashboard.log(f"   ▸ ไปยังเป้าหมาย: ช่อง {target_cell} ด้าน {target_dir} ({signs_desc})")
            dashboard.log(f"   ▸ เส้นทาง A*: {path_str} (รวม {len(path)-1} ก้าว)")

        for next_cell in path[1:]:
            if state.stop_requested or (deadline is not None and time.monotonic() >= deadline):
                break
            target_heading = next(d for d in DIRECTIONS if adjacent(position, d) == next_cell)
            turn_to_direction(ep_chassis, ep_gimbal, heading, target_heading, sim_mode)
            align_heading(ep_chassis, target_heading, sim_mode)
            if state.stop_requested:
                break
            heading = target_heading
            front_dist = read_distance_with_gimbal(ep_gimbal, position, heading, heading, sim_mode)
            move_ok = False
            if _front_is_clear(front_dist):
                move_ok = move_one_cell(
                    ep_chassis, sim_mode, heading=heading, returning=True
                )
            else:
                state.last_move_reason = "PREFLIGHT_WALL" if front_dist > 30 else "TOF_INVALID"
                state.last_move_distance_m = 0.0
            if not move_ok:
                if dashboard:
                    dashboard.log(
                        f"Target route stopped at {position}: {state.last_move_reason}, "
                        f"ToF {front_dist:.0f}mm, moved {state.last_move_distance_m * 100:.1f}cm."
                    )
                state.stop_requested = True
                break
            position = next_cell
            step += 1
            _record_pose(step, position, heading)
            if dashboard:
                dashboard.update(position, heading, step, f"A*: ไปยังเป้าหมาย {target_cell}")
        if state.stop_requested or (deadline is not None and time.monotonic() >= deadline):
            break

        # Use the bearing stored during mapping as a starting point.  Signs are
        # allowed at every 45-degree sector: the gimbal performs the final
        # image-based correction, so the chassis can remain centered in its cell.
        target_heading = target["direction"]
        target_yaw = _direction_yaw(heading, target_heading)
        # Keep the chassis at the mapped cell center during target acquisition.
        # Center the sign optically with the gimbal instead of translating the
        # robot toward a single wall and drifting off the grid.
        if ep_gimbal and not sim_mode:
            ep_gimbal.moveto(pitch=COLOR_SCAN_PITCH - 3, yaw=target_yaw, pitch_speed=COLOR_SCAN_SPEED,
                             yaw_speed=GIMBAL_SPEED).wait_for_completed(timeout=1.5)
        # Hold the chassis still before collecting the final target frames.
        stop_and_settle(ep_chassis, sim_mode, settle_s=0.35)
        target_range = read_target_range_at_yaw(ep_gimbal, position, target_heading, target_yaw, sim_mode)
        if not math.isfinite(target_range) or target_range <= 30:
            target_range = 600.0

        if dashboard:
            dashboard.log(f"🎯 [RUN 2 TARGET] เล็งเป้าหมายที่ช่อง {target['cell']} ด้าน {target_heading}...")
        observed = []
        observed_view = None
        if camera_reader is not None:
            observed, observed_view, _vote_summary = detect_stable_signs(
                camera_reader, dashboard, color_filter=color_filter, shape_filter=shape_filter
            )
        wanted = {(item["color"], item["shape"]) for item in target["signs"]}
        confirmed_pairs = {(item["color"], item["shape"]) for item in observed} & wanted
        if sim_mode:
            confirmed_pairs = wanted
            for color, shape in sorted(confirmed_pairs):
                state.fired_targets.add((tuple(position), color, shape))
                if dashboard:
                    dashboard.log(f"🎯 [SIM TARGET HIT] ยิงกระสุนเจลเสมือนใส่เป้าหมาย {color.upper()} {shape.upper()} ที่ช่อง {target['cell']} ด้าน {target_heading} สำเร็จ!")
        elif confirmed_pairs:
            if not sim_mode and ep_blaster is not None:
                for color, shape in sorted(confirmed_pairs):
                    if shape == "hostage":
                        if dashboard:
                            dashboard.log("⚠️ [SAFETY] ข้ามเป้าหมาย: ตรวจพบตัวประกัน (Hostage) ห้ามยิงเด็ดขาด!")
                        continue
                    matching = [item for item in observed
                                if item["color"] == color and item["shape"] == shape
                                and not item.get("is_hostage") and item.get("shape") != "hostage"]
                    if not matching or observed_view is None:
                        continue
                    candidate = min(
                        matching,
                        key=lambda item: abs(item["bbox"][0] + item["bbox"][2] / 2 - observed_view.shape[1] / 2)
                             + abs(item["bbox"][1] + item["bbox"][3] / 2 - observed_view.shape[0] / 2),
                    )
                    shot_count = aim_verify_and_fire(
                            ep_chassis, ep_gimbal, ep_blaster, camera_reader, candidate,
                            color, shape, frame_shape=observed_view.shape,
                            yaw_offset=target_yaw, target_range_mm=target_range,
                            dashboard=dashboard,
                            sim_mode=sim_mode,
                            cell=position, direction=target_heading)
                    if shot_count:
                        state.fired_targets.add((tuple(position), color, shape))
                        if dashboard:
                            dashboard.log(f"💥 [RUN 2 HIT] ยิงกระสุนเจล 2 นัด เข้าเป้าหมาย {color.upper()} {shape.upper()} ตรงกลางแม่นยำ!")
                    elif dashboard:
                        dashboard.log(f"{color} {shape} not centered after gimbal correction; skipped shot.")
            if dashboard:
                dashboard.log(f"Confirmed {len(confirmed_pairs)} mapped target(s) at {target['cell']} {target_heading}.")
        elif dashboard:
            dashboard.log(f"⚠️ ไม่พบเป้าหมายสดด้าน {target_heading} (อาจติดมุมแสง) ข้ามไปเป้าหมายถัดไป")
        if ep_gimbal and not sim_mode:
            ep_gimbal.recenter().wait_for_completed(timeout=1.5)
        targets.remove(target)
        step += 1

    # End of target round: return to start cell with A*
    if not state.stop_requested and position != start_pos:
        return_path = _shortest_cell_path(position, start_pos, current_heading=heading)
        if return_path and len(return_path) > 1:
            if dashboard:
                dashboard.log(f"🏁 [A* COMPLETE] ยิงเป้าหมายครบถ้วนแล้ว! กำลังเดิน A* กลับจุดเริ่มต้น {start_pos}...")
            for next_cell in return_path[1:]:
                if state.stop_requested or (deadline is not None and time.monotonic() >= deadline):
                    break
                target_heading = next(d for d in DIRECTIONS if adjacent(position, d) == next_cell)
                turn_to_direction(ep_chassis, ep_gimbal, heading, target_heading, sim_mode)
                align_heading(ep_chassis, target_heading, sim_mode)
                heading = target_heading
                front_dist = read_distance_with_gimbal(ep_gimbal, position, heading, heading, sim_mode)
                if not _front_is_clear(front_dist) or not move_one_cell(ep_chassis, sim_mode, heading=heading, returning=True):
                    break
                position = next_cell
                step += 1
                _record_pose(step, position, heading)
                if dashboard:
                    dashboard.update(position, heading, step, f"A*: เดินกลับจุดเริ่มต้น {start_pos}")
            if not state.stop_requested:
                align_heading(ep_chassis, heading, sim_mode)
                if dashboard:
                    dashboard.log(f"🏁 หุ่นยนต์กลับถึงจุดเริ่มต้น {start_pos} เรียบร้อยแล้ว!")
    return position, heading


def run_exploration(sim_mode, start_config, dashboard, ground_truth=None, mission_config=None,
                    finish_dashboard=True):
    """
    Main exploration loop ensuring the robot visits every reachable cell in the maze.
    Orchestrates gimbal scanning, Mecanum recentering, BFS path planning, and movement.
    """
    mission_config = mission_config or {"run_mode": "mapping", "colors": set(COLORS), "shapes": set(SHAPES)}
    run_mode = mission_config.get("run_mode", "mapping")
    color_filter = set(mission_config.get("colors", set()))
    shape_filter = set(mission_config.get("shapes", set()))

    gw = mission_config.get("grid_w")
    gh = mission_config.get("grid_h")
    cs = mission_config.get("cell_size_m")
    if gw is not None and gh is not None:
        config.set_grid_dimensions(gw, gh, cs)

    saved_map = None
    if run_mode in ("targets", "astar_only", "astar"):
        try:
            saved_map = _load_mission_map()
            if start_config is None:
                start_config = tuple(saved_map["start"])
            if dashboard:
                dashboard.log(f"🚀 [A* LOAD] โหลดแผนที่สำเร็จ! จุดเริ่ม {start_config} | ช่องที่เคยเดิน {len(saved_map.get('visited', []))} ช่อง | เป้าหมาย {len(saved_map.get('targets', []))} จุด")
        except Exception as exc:
            if state.detected_h_walls or state.detected_v_walls or state.visited_cells:
                if dashboard:
                    dashboard.log(f"🚀 [A* LOAD] ใช้ข้อมูลแผนที่จากหน่วยความจำ/CSV ({len(state.visited_cells)} ช่อง, เป้าหมาย {sum(len(s) for s in state.detected_signs.values())} จุด)")
            else:
                if dashboard:
                    dashboard.log(f"Run 2 (A*) cannot load mapped targets: {exc}")
                dashboard.run_finished() if dashboard else None
                return False
    start_pos = (start_config[0], start_config[1])
    heading = start_config[2]
    position = start_pos

    # Reset all runtime state for a fresh run
    state.reset_state(start_pos)
    reset_sensor_filters()
    state.initial_heading = heading
    if saved_map is not None:
        state.detected_h_walls.update(tuple(edge) for edge in saved_map["h_walls"])
        state.detected_v_walls.update(tuple(edge) for edge in saved_map["v_walls"])
        state.discovered_cells.update(tuple(cell) for cell in saved_map["discovered"])
        state.visited_cells.update(tuple(cell) for cell in saved_map["visited"])
        for target in saved_map["targets"]:
            key = (tuple(target["cell"]), target["direction"])
            state.detected_signs[key] = target["signs"]
            if dashboard:
                for sign in target["signs"]:
                    img_p = sign.get("image_path")
                    if img_p and os.path.exists(img_p):
                        meta = {
                            "color": sign.get("color", ""),
                            "shape": sign.get("shape", ""),
                            "cell": tuple(target["cell"]),
                            "direction": target["direction"],
                            "fired": (tuple(target["cell"]), sign.get("color"), sign.get("shape")) in state.fired_targets,
                            "area": sign.get("area", 0),
                            "time": time.strftime("%H:%M:%S")
                        }
                        dashboard.add_target_result(img_p, meta)
        for scan in saved_map.get("color_scans", []):
            key = (tuple(scan["cell"]), scan["direction"])
            state.color_scan_results[key] = {k: v for k, v in scan.items()
                                             if k not in ("cell", "direction")}

    ep_robot = ep_chassis = ep_gimbal = ep_sensor = ep_ir_adapter = ep_camera = camera_reader = ep_blaster = None
    vision_analyzer = None
    mission_ready = False

    try:
        if not sim_mode:
            if robot is None:
                raise RuntimeError("ติดตั้ง RoboMaster SDK ด้วย python -m pip install robomaster")
            if dashboard:
                dashboard.log("กำลังเชื่อมต่อ RoboMaster...")
            ep_robot = robot.Robot()
            ep_robot.initialize(conn_type="ap")
            ep_chassis, ep_gimbal, ep_sensor, ep_blaster = ep_robot.chassis, ep_robot.gimbal, ep_robot.sensor, ep_robot.blaster
            ep_camera = ep_robot.camera
            ep_ir_adapter = ep_robot.sensor_adaptor
            # Skip the blocking startup recenter: every scan/aim command below
            # sets an absolute yaw and pitch before using the camera.
            ep_sensor.sub_distance(freq=10, callback=sub_tof_handler)
            ep_ir_adapter.sub_adapter(freq=50, callback=sub_ir_handler)
            ep_chassis.sub_attitude(freq=20, callback=sub_attitude_handler)
            ep_chassis.sub_position(freq=20, callback=sub_position_handler)
            try:
                if ep_camera.start_video_stream(display=False, resolution="720p"):
                    camera_reader = CameraFrameReader(ep_camera)
                    camera_reader.start()
                    vision_analyzer = VisionFrameAnalyzer(camera_reader, interval_s=0.12)
                    vision_analyzer.set_filters(None, None)
                    vision_analyzer.start()
                    if dashboard:
                        dashboard.set_vision_source(camera_reader, vision_analyzer)
                    if dashboard:
                        dashboard.log("เปิดกล้อง RoboMaster แล้ว")
                else:
                    ep_camera = None
                    if dashboard:
                        dashboard.log("เปิดกล้องไม่สำเร็จ ระบบจะสแกนเฉพาะ ToF")
            except Exception as camera_error:
                print(f"[CAMERA] start failed: {camera_error}")
                ep_camera = None
                if dashboard:
                    dashboard.log("เปิดกล้องไม่สำเร็จ ระบบจะสแกนเฉพาะ ToF")
            time.sleep(0.80)  # Allow IMU attitude stream to fully settle before calibrating initial_yaw
            if dashboard:
                dashboard.log("เชื่อมต่อ RoboMaster สำเร็จ!")

        print(f"เริ่มสำรวจจากตำแหน่ง {position} ทิศ {heading}")
        if dashboard:
            dashboard.log(f"เริ่มสำรวจจาก {position} ทิศ {heading}")

        step = 0
        round_started = time.monotonic()
        is_mapping_round = run_mode in ("mapping", "slam_only", "slam")
        round_limit_s = getattr(config, "ROUND1_LIMIT_S", 900) if is_mapping_round else getattr(config, "ROUND2_LIMIT_S", 600)
        round_label = "round1" if is_mapping_round else "round2"

        # Ensure timer is active on dashboard if not already started by GUI button
        if dashboard and hasattr(dashboard, "start_timer") and not dashboard._timer_running:
            dashboard.start_timer(
                round_key="round1" if is_mapping_round else "round2",
                round_label="รอบ 1 (SLAM)" if is_mapping_round else "รอบ 2 (A*)"
            )

        if dashboard:
            if is_mapping_round:
                dashboard.log("=" * 60)
                dashboard.log("🗺️ [ROUND 1 - SLAM] เริ่มต้นรอบที่ 1: เดินสำรวจและสร้างแผนที่ด้วย SLAM")
                dashboard.log(f"   ▸ เวลาสูงสุด: {round_limit_s // 60} นาที | จุดเริ่มต้น {position} ทิศ {heading}")
                dashboard.log("=" * 60)
            else:
                dashboard.log("=" * 60)
                dashboard.log("⭐ [ROUND 2 - A*] เริ่มต้นรอบที่ 2: นำทางค้นหาและยิงเป้าหมายด้วย A* Algorithm")
                dashboard.log(f"   ▸ เวลาสูงสุด: {round_limit_s // 60} นาที | จุดเริ่มต้น {position} ทิศ {heading}")
                dashboard.log("=" * 60)

        # Start 10-second periodic background autosave worker thread
        autosave_stop_event = threading.Event()
        def _autosave_worker():
            while not autosave_stop_event.wait(10.0):
                if state.stop_requested:
                    break
                try:
                    if state.visited_cells and is_mapping_round:
                        _save_mission_map(start_config)
                        save_outputs("autosave", generate_plot=False)
                        msg = f"💾 [AUTOSAVE 10s] บันทึกแผนที่ ({len(state.visited_cells)} ช่อง) และ Log การเดินอัตโนมัติ"
                        if dashboard:
                            dashboard.log(msg)
                        else:
                            print(msg)
                except Exception as auto_err:
                    print(f"[AUTOSAVE WARNING] {auto_err}")

        autosave_thread = threading.Thread(target=_autosave_worker, daemon=True)
        autosave_thread.start()

        if not is_mapping_round:
            position, heading = _run_target_round(
                ep_chassis, ep_gimbal, ep_blaster, camera_reader, position, heading,
                sim_mode, dashboard, step, color_filter, shape_filter,
                vision_analyzer=vision_analyzer, deadline=round_started + round_limit_s,
            )
            if time.monotonic() >= round_started + round_limit_s:
                state.stop_requested = True

        # ติดตามช่องที่เคยสแกนหาเป้าหมายไปแล้ว เพื่อไม่ให้เสียเวลาสแกนซ้ำเมื่อเดินผ่าน
        scanned_target_cells = set()

        while (not state.stop_requested and is_mapping_round
               and time.monotonic() - round_started < round_limit_s):
            # Stop and let chassis motion settle before every gimbal scan
            stop_and_settle(ep_chassis, sim_mode, settle_s=0.18)
            align_heading(ep_chassis, heading, sim_mode)
            if state.stop_requested:
                break

            is_already_visited = position in scanned_target_cells
            scanned_target_cells.add(position)

            if dashboard:
                dashboard.log(f"📍 [STEP {step}] หุ่นยนต์อยู่ที่ ({position[0]},{position[1]}) ทิศ {heading} | สำรวจแล้ว {len(state.visited_cells)}/{config.GRID_W*config.GRID_H} ช่อง ({len(state.visited_cells)/(config.GRID_W*config.GRID_H)*100:.1f}%)")

            if is_already_visited:
                # ตรงที่เดิน visit ไปแล้ว ไม่ต้องหาเป้าหมายซ้ำ! (Fast Backtrack Pass)
                if dashboard:
                    dashboard.log(f"⚡ [FAST PASS] ช่อง {position} เคยสำรวจแล้ว -> ไม่ต้องหาเป้าหมายซ้ำ (เดินผ่านเร็ว)")
                readings = scan_45_degree_sweep(
                    ep_chassis, ep_gimbal, position, heading, sim_mode, dashboard, step,
                    camera_reader=None, color_filter=None, shape_filter=None,
                    blaster=None, fire_enabled=False, search_targets=False,
                )
            else:
                if dashboard:
                    dashboard.log(f"🔭 [SCAN START] เริ่มสแกน 3 ทิศทาง (หน้า, ขวา, ซ้าย) จากช่อง {position}")
                readings = scan_45_degree_sweep(
                    ep_chassis, ep_gimbal, position, heading, sim_mode, dashboard, step,
                    camera_reader=camera_reader, color_filter=color_filter,
                    shape_filter=shape_filter, blaster=ep_blaster, fire_enabled=True,
                    search_targets=True,
                )

            state.visited_cells.add(position)
            if is_mapping_round:
                _save_mission_map(start_config)

            if state.stop_requested:
                break

            # ปรับกึ่งกลางเฉพาะช่องใหม่ เพื่อความรวดเร็วในการเคลื่อนที่
            if not is_already_visited:
                recenter_in_cell(ep_chassis, ep_gimbal, readings, position, heading, sim_mode, dashboard)
            if state.stop_requested:
                break

            # 3. Log trajectory
            previous = state.trajectory[-1]["cell"] if state.trajectory else None
            state.trajectory.append({
                "step": step,
                "timestamp": round(time.time(), 2),
                "grid_x": position[0],
                "grid_y": position[1],
                "x_m": round((position[0] - 0.5) * config.GRID_SIZE_M, 3),
                "y_m": round((position[1] - 0.5) * config.GRID_SIZE_M, 3),
                "heading": heading,
                "tof_mm": min(readings.values()) if readings else 9999,
                "cell": position,
                "previous": previous,
                "stack_depth": len(state.exploration_stack),
            })

            # 4. Find path to nearest unvisited cell through open passages
            path = find_path_to_nearest_unvisited(position, state.visited_cells, current_heading=heading)
            if not path or len(path) < 2:
                if dashboard:
                    dashboard.log("✅ สำรวจครบทุกช่องที่เดินได้แล้ว (All reachable cells visited)!")
                break

            next_cell = path[1]
            target_heading = next(d for d in DIRECTIONS if adjacent(position, d) == next_cell)

            if dashboard:
                target_desc = "สำรวจช่องใหม่" if next_cell not in state.visited_cells else "เดินย้อนทางเดิม (Backtrack)"
                dashboard.log(f"🧭 [NAV PLAN] เป้าหมายถัดไป: {next_cell} [{target_desc}] | หัน {target_heading} | เส้นทาง {len(path)-1} ก้าว")
                dashboard.update(position, target_heading, step, f"กำลังเคลื่อนที่ไป {next_cell}")

            # 5. Turn chassis to target heading at the center of the cell
            # Because recentering just completed, the robot has full clearance from all walls (~9cm+)
            if heading != target_heading and dashboard:
                dashboard.log(f"🔄 [NAV TURN] หมุนตัวจาก {heading} -> {target_heading} ที่จุดกึ่งกลางช่อง...")
            turn_to_direction(ep_chassis, ep_gimbal, heading, target_heading, sim_mode)
            align_heading(ep_chassis, target_heading, sim_mode)
            heading = target_heading
            # Previously visited cells are path retraces: keep a straight
            # heading correction active while the forward IR/ToF prevents impact.
            front_dist = read_distance_with_gimbal(ep_gimbal, position, heading, heading, sim_mode)
            if not _front_is_clear(front_dist):
                state.last_move_reason = "PREFLIGHT_WALL" if math.isfinite(float(front_dist)) and front_dist > 30 else "TOF_INVALID"
                state.last_move_distance_m = 0.0
                move_ok = False
            else:
                move_ok = move_one_cell(
                    ep_chassis, sim_mode, heading=heading,
                    returning=next_cell in state.visited_cells,
                    dashboard=dashboard,
                )

            if move_ok:
                position = next_cell
                state.exploration_stack.append(position)
                state.visited_cells.add(position)
                state.discovered_cells.add(position)
                # Publish the new pose as soon as the completed cell move is
                # confirmed. The following ToF/color scan can take several
                # seconds; waiting for it left the GUI marker one cell behind
                # the physical chassis during that entire scan.
                if dashboard:
                    dashboard.update(
                        position, heading, step,
                        f"Cell move confirmed; scanning from {position}",
                    )
            else:
                obstacle_stop = state.last_move_reason in {
                    "PREFLIGHT_WALL", "FRONT_BRAKE", "EMERGENCY"
                }
                # Grid cells are 60cm center-to-center. Once measured odometry
                # crosses half that distance from the cell center, the robot is
                # physically on the next side of the cell boundary. Count that
                # cell even if the 20cm emergency brake stopped the rest of the
                # commanded move, then map the detected wall from that cell.
                crossed_into_next_cell = (
                    state.last_move_reason != "TOF_INVALID"
                    and state.last_move_reason != "CANCELLED"
                    and state.last_move_distance_m >= config.GRID_SIZE_M / 2
                )
                if crossed_into_next_cell:
                    position = next_cell
                    if not state.exploration_stack or state.exploration_stack[-1] != position:
                        state.exploration_stack.append(position)
                    state.visited_cells.add(position)
                    state.discovered_cells.add(position)

                # A valid ToF stop is wall evidence. Anchor that wall to the
                # cell the chassis actually occupies after the measured move.
                if obstacle_stop:
                    x, y = position
                    if heading == "NORTH":
                        state.detected_h_walls.add((x, y))
                    elif heading == "SOUTH":
                        state.detected_h_walls.add((x, y - 1))
                    elif heading == "EAST":
                        state.detected_v_walls.add((x, y))
                    elif heading == "WEST":
                        state.detected_v_walls.add((x - 1, y))

                if dashboard:
                    dashboard.log(
                        f"Move ended ({state.last_move_reason}, odometry "
                        f"{state.last_move_distance_m*100:.1f}cm); map position {position}."
                    )
                    dashboard.update(
                        position, heading, step,
                        f"Position synchronized from odometry: {position}",
                    )
                # A wall/emergency brake is a completed cell observation, not
                # a mission failure. Continue surveying from the measured cell;
                # stop only when pose data are invalid or a non-wall move fails.
                if not obstacle_stop and not crossed_into_next_cell:
                    state.stop_requested = True

            step += 1

        # Finish at the saved start pose so Run 2 can begin immediately from
        # the same physical cell without relying on a stale pose assumption.
        if (is_mapping_round and not state.stop_requested
                and time.monotonic() - round_started < round_limit_s and position != start_pos):
            return_path = _shortest_cell_path(position, start_pos, current_heading=heading)
            if return_path:
                if dashboard:
                    dashboard.log("Mapping complete; returning to the saved start cell for Run 2 (A*).")
                for next_cell in return_path[1:]:
                    if time.monotonic() >= round_started + round_limit_s:
                        state.stop_requested = True
                        break
                    target_heading = next(d for d in DIRECTIONS if adjacent(position, d) == next_cell)
                    turn_to_direction(ep_chassis, ep_gimbal, heading, target_heading, sim_mode)
                    align_heading(ep_chassis, target_heading, sim_mode)
                    heading = target_heading
                    front_dist = read_distance_with_gimbal(ep_gimbal, position, heading, heading, sim_mode)
                    if not _front_is_clear(front_dist) or not move_one_cell(
                            ep_chassis, sim_mode, heading=heading, returning=True):
                        state.stop_requested = True
                        if dashboard:
                            dashboard.log("Safety stop while returning to the saved start cell.")
                        break
                    position = next_cell
                    step += 1
                    if dashboard:
                        dashboard.update(position, heading, step, "Returning to Run 2 start cell")
                if not state.stop_requested:
                    turn_to_direction(ep_chassis, ep_gimbal, heading, start_config[2], sim_mode)
                    align_heading(ep_chassis, start_config[2], sim_mode)
                    heading = start_config[2]

        total_detected_walls = len(state.detected_h_walls) + len(state.detected_v_walls)
        total_targets = sum(len(s) for s in state.detected_signs.values())
        total_grid_cells = config.GRID_W * config.GRID_H
        print("\n=======================================================")
        print("🏁 สรุปผลการทำงาน (MISSION SUMMARY)")
        print("=======================================================")
        print(f"• ช่องที่สำรวจ : {len(state.visited_cells)} / {total_grid_cells} ช่อง")
        print(f"• แนวกำแพงโฟม  : ตรวจพบ {total_detected_walls} แนวกำแพง")
        print(f"• เป้าที่ตรวจพบ: {total_targets} เป้า")
        print(f"• จุดเริ่มต้น  : {start_pos} | จุดสิ้นสุด: {position}")
        print("=======================================================\n")
        elapsed = time.monotonic() - round_started
        if elapsed >= round_limit_s:
            if dashboard:
                dashboard.log(f"Time limit reached ({round_limit_s // 60} minutes); ending {round_label}.")
        save_outputs(round_label)

        if is_mapping_round:
            _save_mission_map(start_config)
            mission_ready = True
            if dashboard:
                dashboard.log(f"Saved SLAM map and target detections to {MISSION_MAP_PATH}")

        if dashboard:
            if hasattr(dashboard, "on_round_completed"):
                dashboard.on_round_completed(round_label)
            status_text = "หยุดการทำงาน" if state.stop_requested else f"สำรวจครบทุกช่องแล้ว ({len(state.visited_cells)} ช่อง)"
            dashboard.update(position, heading, step, status_text)
            elapsed_sec = int(time.monotonic() - round_started)
            dashboard.log("=" * 60)
            dashboard.log("🏁 สรุปผลการทำงาน (MISSION SUMMARY REPORT)")
            dashboard.log("=" * 60)
            dashboard.log(f"• เวลาที่ใช้รอบนี้ ({round_label}) : {elapsed_sec // 60} นาที {elapsed_sec % 60} วินาที")
            if hasattr(dashboard, "_round_times"):
                r1_t = dashboard._round_times.get("round1")
                r2_t = dashboard._round_times.get("round2")
                tot_t = dashboard._round_times.get("total")
                if r1_t is not None:
                    dashboard.log(f"  ▸ เวลา รอบ 1 (SLAM)   : {dashboard._format_time_str(r1_t)}")
                if r2_t is not None:
                    dashboard.log(f"  ▸ เวลา รอบ 2 (A*)     : {dashboard._format_time_str(r2_t)}")
                if tot_t is not None:
                    dashboard.log(f"  ▸ เวลารวมทั้งหมด      : {dashboard._format_time_str(tot_t)}")
            dashboard.log(f"• ช่องที่สำรวจสำเร็จ: {len(state.visited_cells)}/{total_grid_cells} ช่อง ({len(state.visited_cells)/total_grid_cells*100:.1f}%)")
            dashboard.log(f"• แนวกำแพงโฟม      : ตรวจพบ {total_detected_walls} แนว (แนวนอน: {len(state.detected_h_walls)}, แนวตั้ง: {len(state.detected_v_walls)})")
            dashboard.log(f"• เป้าหมายที่ตรวจพบ : {total_targets} เป้า")
            dashboard.log(f"• เป้าหมายที่ยิงสำเร็จ: {len(state.fired_targets)} เป้า")
            dashboard.log(f"• ระยะทางเดินทั้งหมด: {len(state.trajectory) * config.GRID_SIZE_M:.2f} เมตร ({len(state.trajectory)} ก้าว)")
            dashboard.log("• บันทึกไฟล์ผลลัพธ์  : โฟลเดอร์ results/ และ final_slam_map.png")
            dashboard.log("=" * 60)
            dashboard.show_final_map_window(round_label)

    except KeyboardInterrupt:
        print("\n⚠️ ผู้ใช้กดหยุดฉุกเฉิน (KeyboardInterrupt / Ctrl+C)")
        state.stop_requested = True
        if state.visited_cells:
            _save_mission_map(start_config)
            save_outputs(round_label)
            print(f"💾 บันทึกแผนที่ ({len(state.visited_cells)} ช่อง, {sum(len(s) for s in state.detected_signs.values())} เป้า) ลง {MISSION_MAP_PATH} สำเร็จแล้ว พร้อมรันรอบ 2 ได้ทันที!")
            if dashboard:
                dashboard.log("💾 [AUTOSAVE] บันทึกแผนที่และภาพเป้าหมายลงไฟล์เรียบร้อยแล้ว (พร้อมสำหรับ Run 2)")
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()
        if state.visited_cells:
            try:
                _save_mission_map(start_config)
                save_outputs(round_label)
            except Exception:
                pass
        if dashboard:
            dashboard.log(f"เกิดข้อผิดพลาด: {e}")
    finally:
        try:
            autosave_stop_event.set()
        except Exception:
            pass
        if state.visited_cells:
            try:
                _save_mission_map(start_config)
                save_outputs(round_label)
            except Exception:
                pass
        if dashboard and finish_dashboard:
            dashboard.run_finished()
        if ep_sensor:
            try:
                ep_sensor.unsub_distance()
            except Exception:
                pass
        if ep_ir_adapter:
            try:
                ep_ir_adapter.unsub_adapter()
            except Exception:
                pass
        if ep_gimbal:
            try:
                ep_gimbal.recenter().wait_for_completed()
            except Exception:
                pass
        if vision_analyzer:
            vision_analyzer.stop()
        if camera_reader:
            camera_reader.stop()
        if ep_camera:
            try:
                ep_camera.stop_video_stream()
            except Exception:
                pass
        if ep_robot:
            try:
                ep_robot.close()
            except Exception:
                pass
    return mission_ready and position == start_pos


def main():
    parser = argparse.ArgumentParser(description="RoboMaster Autonomous SLAM - Assignment 2")
    parser.add_argument("--sim", action="store_true", help="จำลองโดยไม่เชื่อมต่อหุ่นยนต์จริง")
    parser.add_argument("--no-dashboard", action="store_true", help="ไม่เปิดหน้าต่าง dashboard")
    parser.add_argument("--start-x", type=int, default=1, help="พิกัด X เริ่มต้น (1 ถึง 6)")
    parser.add_argument("--start-y", type=int, default=1, help="พิกัด Y เริ่มต้น (1 ถึง 6)")
    parser.add_argument("--start-heading", choices=DIRECTIONS, default="NORTH", help="ทิศทางเริ่มต้น")
    parser.add_argument("--mode", choices=["all", "slam", "astar", "mapping", "targets"], default="all",
                        help="โหมด: slam (รอบ 1 SLAM), astar (รอบ 2 A*), all (รอบ 1 SLAM ➔ รอบ 2 A*)")
    args = parser.parse_args()

    default_start = (args.start_x, args.start_y, args.start_heading)

    if args.no_dashboard:
        if args.mode in ("slam", "mapping"):
            clean_old_results()
            run_exploration(args.sim, default_start, None, mission_config={"run_mode": "slam_only", "colors": set(COLORS), "shapes": set(SHAPES)})
        elif args.mode in ("astar", "targets"):
            run_exploration(args.sim, default_start, None, mission_config={"run_mode": "astar_only", "colors": set(COLORS), "shapes": set(SHAPES)})
        else:  # "all"
            clean_old_results()
            mission_ready = run_exploration(args.sim, default_start, None, mission_config={"run_mode": "slam_only", "colors": set(COLORS), "shapes": set(SHAPES)})
            if mission_ready and not state.stop_requested:
                run_exploration(
                    args.sim, default_start, None,
                    mission_config={"run_mode": "astar_only", "colors": set(COLORS), "shapes": set(SHAPES)},
                )
    else:
        dashboard = None

        def on_start(start_cfg, mission_cfg):
            try:
                run_mode = mission_cfg.get("run_mode", "auto_all")
                if run_mode in ("slam_only", "slam"):
                    clean_old_results()
                    if dashboard:
                        dashboard.reset_session_ui()
                        dashboard.log("🧹 ล้างข้อมูลรอบเก่าทั้งหมดเรียบร้อยแล้ว เริ่มต้น [รอบ 1: เดินสำรวจ SLAM]...")
                    run_exploration(
                        args.sim, start_cfg, dashboard,
                        mission_config=dict(mission_cfg, run_mode="slam_only"), finish_dashboard=True,
                    )
                elif run_mode in ("astar_only", "astar", "targets"):
                    if dashboard:
                        dashboard.log("⭐ เริ่มต้น [รอบ 2: เดินนำทางและยิงเป้าหมายด้วย A* Algorithm]...")
                    run_exploration(
                        args.sim, start_cfg, dashboard,
                        mission_config=dict(mission_cfg, run_mode="astar_only"), finish_dashboard=True,
                    )
                else:  # "auto_all", "mapping", "all"
                    clean_old_results()
                    if dashboard:
                        dashboard.reset_session_ui()
                        dashboard.log("🧹 ล้างข้อมูลรอบเก่าทั้งหมดเรียบร้อยแล้ว เริ่มต้นภารกิจ [รอบ 1 SLAM ➔ รอบ 2 A* ต่อเนื่อง]...")
                    ready = run_exploration(
                        args.sim, start_cfg, dashboard,
                        mission_config=dict(mission_cfg, run_mode="slam_only"), finish_dashboard=False,
                    )
                    if ready and not state.stop_requested:
                        if dashboard:
                            dashboard.on_round_completed("round1")
                            dashboard.log("=" * 60)
                            dashboard.log("🎉 จบรอบที่ 1 (SLAM) สำเร็จ! กำลังต่อ [รอบที่ 2: A* Algorithm] อัตโนมัติ...")
                            dashboard.log("=" * 60)
                            dashboard.start_timer("round2", "รอบ 2 (A*)")
                        target_config = dict(mission_cfg, run_mode="astar_only")
                        run_exploration(
                            args.sim, start_cfg, dashboard,
                            mission_config=target_config, finish_dashboard=True,
                        )
                    else:
                        if dashboard:
                            dashboard.on_round_completed("round1")
                        dashboard.run_finished()
            except Exception as e:
                print(f"[MISSION ERROR] {e}")
                import traceback
                traceback.print_exc()
                if dashboard:
                    dashboard.log(f"❌ เกิดข้อผิดพลาดในการรันภารกิจ: {e}")
                    dashboard.run_finished()

        def on_stop():
            state.stop_requested = True

        dashboard = Dashboard(
            on_start=on_start,
            on_stop=on_stop,
            sim_mode=args.sim,
            default_start=default_start,
        )
        try:
            dashboard.root.mainloop()
        except KeyboardInterrupt:
            print("\n⚠️ ผู้ใช้กด Ctrl+C ในหน้าจอ Terminal...")
            state.stop_requested = True
            if state.visited_cells:
                try:
                    cfg = dashboard.get_start_config() if dashboard else default_start
                    _save_mission_map(cfg)
                    save_outputs("emergency_stop")
                    print(f"💾 บันทึกแผนที่ ({len(state.visited_cells)} ช่อง) และ Log การเดินเรียบร้อยแล้ว!")
                except Exception as se:
                    print(f"Error saving on Ctrl+C: {se}")


if __name__ == "__main__":
    main()
