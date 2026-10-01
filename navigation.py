# -*- coding: utf-8 -*-
"""
Navigation and Path Planning Module.
Provides wall checking functions, BFS fallback, and A* shortest path algorithm
with turn penalty for autonomous exploration of the maze.
"""

import heapq
from collections import deque
from config import DIRECTIONS, SIM_H_WALLS, SIM_V_WALLS, WALL_THRESHOLD_MM, adjacent, in_bounds
import state

# Turn penalty: ค่าปรับเมื่อหุ่นยนต์ต้องเลี้ยว (ใช้เป็น Tie-breaker เท่านั้น ไม่ให้แซงระยะก้าว)
TURN_PENALTY_90 = 0.05    # หมุน 90° (เลี้ยวซ้าย/ขวา)
TURN_PENALTY_180 = 0.10   # หมุนกลับหลัง 180°
STEP_COST = 100.0         # ต้นทุนหลักต่อก้าว (รับประกันว่าเส้นทางก้าวน้อยกว่าจะชนะเสมอ 100% ไม่เดินอ้อม)


def is_wall_distance(dist):
    """
    Check if ToF reading corresponds to a real foam wall on cell border (25mm - 380mm).
    Accepts full physical reflection range down to 25mm so close walls are never ignored.
    """
    return 30 <= dist <= WALL_THRESHOLD_MM  # Slightly higher lower bound to ignore noise spikes


def is_wall_between(cell, direction):
    """Check if a detected foam wall exists on the border of cell in the given direction."""
    x, y = cell
    if direction == "NORTH":
        return (x, y) in state.detected_h_walls
    elif direction == "SOUTH":
        return (x, y - 1) in state.detected_h_walls
    elif direction == "EAST":
        return (x, y) in state.detected_v_walls
    elif direction == "WEST":
        return (x - 1, y) in state.detected_v_walls
    return True


def sim_has_wall_between(cell, direction):
    """Simulated check for foam walls on borders (used during --sim mode)."""
    x, y = cell
    if direction == "NORTH":
        return (x, y) in SIM_H_WALLS
    elif direction == "SOUTH":
        return (x, y - 1) in SIM_H_WALLS
    elif direction == "EAST":
        return (x, y) in SIM_V_WALLS
    elif direction == "WEST":
        return (x - 1, y) in SIM_V_WALLS
    return True


def _turn_cost(from_dir, to_dir):
    """คำนวณค่าปรับการเลี้ยว (turn penalty) ระหว่างสองทิศทาง"""
    if from_dir is None or from_dir == to_dir:
        return 0.0
    idx_from = DIRECTIONS.index(from_dir)
    idx_to = DIRECTIONS.index(to_dir)
    diff = abs(idx_from - idx_to)
    if diff == 2:
        return TURN_PENALTY_180  # หมุนกลับ 180°
    else:
        return TURN_PENALTY_90   # เลี้ยว 90°


def _heuristic(cell, goal):
    """Manhattan distance heuristic สำหรับ A*"""
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def astar_path(start, goal, current_heading=None):
    """
    A* algorithm หาเส้นทางสั้นที่สุดจาก start ไป goal โดยยึดจำนวนก้าวน้อยที่สุดเป็นหลัก
    และใช้ turn penalty เป็นเพียงตัวตัดสินเมื่อจำนวนก้าวเท่ากัน (ไม่เดินอ้อม)

    Parameters:
        start: tuple (x, y) จุดเริ่มต้น
        goal: tuple (x, y) จุดหมาย
        current_heading: str ทิศทางปัจจุบัน เช่น "NORTH", "EAST", ...

    Returns:
        list of (x,y) tuples เส้นทาง หรือ None ถ้าไปไม่ถึง
    """
    if start == goal:
        return [start]

    # Priority queue: (f_cost, counter, cell, heading)
    counter = 0
    open_set = [(0.0 + _heuristic(start, goal) * STEP_COST, counter, start, current_heading)]
    g_cost = {(start, current_heading): 0.0}
    parent = {(start, current_heading): None}

    while open_set:
        f, _cnt, curr, curr_heading = heapq.heappop(open_set)

        if curr == goal:
            # Reconstruct path
            path = []
            key = (curr, curr_heading)
            while key is not None:
                path.append(key[0])
                key = parent[key]
            return list(reversed(path))

        for d in DIRECTIONS:
            if is_wall_between(curr, d):
                continue  # กำแพงขวาง

            nxt = adjacent(curr, d)
            if not in_bounds(nxt):
                continue

            # ต้นทุนเดิน 1 ก้าว (STEP_COST=100) + ค่าปรับเลี้ยวเล็กน้อย
            move_cost = STEP_COST + _turn_cost(curr_heading, d)
            new_g = g_cost[(curr, curr_heading)] + move_cost

            key_nxt = (nxt, d)
            if key_nxt not in g_cost or new_g < g_cost[key_nxt]:
                g_cost[key_nxt] = new_g
                f_new = new_g + _heuristic(nxt, goal) * STEP_COST
                counter += 1
                heapq.heappush(open_set, (f_new, counter, nxt, d))
                parent[key_nxt] = (curr, curr_heading)

    return None  # ไปไม่ถึง


def find_path_to_nearest_unvisited(start, visited, current_heading=None):
    """
    ค้นหาเส้นทางสั้นที่สุด (ระยะก้าวน้อยที่สุดเสมอ) ไปยังช่องที่ยังไม่ได้เยี่ยมชม
    1. ตรวจสอบช่องข้างเคียงติดกัน (1 ก้าว) ก่อนทันที หากมีช่องเปิดที่ยังไม่สำรวจ จะเลือกเดินเข้าทันที ไม่เดินอ้อม
    2. หากช่องรอบตัวสำรวจหมดแล้ว จะใช้ Dijkstra ที่คำนวณจำนวนก้าวน้อยที่สุดเป็นหลัก (Step Cost 100)
       รับประกันว่าจะเลือกช่องที่ใกล้ตัวที่สุดเสมอ ไม่เดินอ้อมไกล
    """
    # 1. ตรวจสอบช่องข้างเคียง (Immediate Neighbors) 1 ก้าวทันที
    if current_heading is not None:
        h_idx = DIRECTIONS.index(current_heading)
        check_dirs = [
            current_heading,
            DIRECTIONS[(h_idx + 1) % 4],
            DIRECTIONS[(h_idx - 1) % 4],
            DIRECTIONS[(h_idx + 2) % 4],
        ]
    else:
        check_dirs = DIRECTIONS

    for d in check_dirs:
        if not is_wall_between(start, d):
            nxt = adjacent(start, d)
            if in_bounds(nxt) and nxt not in visited:
                return [start, nxt]

    # 2. ค้นหาเส้นทางที่สั้นที่สุด (ก้าวน้อยที่สุด) ไปยังช่องที่ยังไม่เคยสำรวจ
    counter = 0
    open_set = [(0.0, counter, start, current_heading)]
    g_cost = {(start, current_heading): 0.0}
    parent = {(start, current_heading): None}

    while open_set:
        f, _cnt, curr, curr_heading = heapq.heappop(open_set)

        # Goal: ช่องที่ยังไม่ได้เยี่ยมชม (Unvisited frontier cell)
        if curr not in visited:
            path = []
            key = (curr, curr_heading)
            while key is not None:
                path.append(key[0])
                key = parent[key]
            return list(reversed(path))

        # ขยายเส้นทางเฉพาะจากช่องที่เคยสำรวจกำแพงแล้ว (visited) เท่านั้น
        for d in DIRECTIONS:
            if is_wall_between(curr, d):
                continue

            nxt = adjacent(curr, d)
            if not in_bounds(nxt):
                continue

            # ต้นทุนก้าวละ 100.0 + ค่าปรับเลี้ยวเล็กน้อย (0.05/0.10)
            # รับประกันว่าเส้นทางที่จำนวนก้าวน้อยกว่าจะชนะเสมอ 100% ไม่เดินอ้อม
            move_cost = STEP_COST + _turn_cost(curr_heading, d)
            new_g = g_cost[(curr, curr_heading)] + move_cost

            key_nxt = (nxt, d)
            if key_nxt not in g_cost or new_g < g_cost[key_nxt]:
                g_cost[key_nxt] = new_g
                counter += 1
                heapq.heappush(open_set, (new_g, counter, nxt, d))
                parent[key_nxt] = (curr, curr_heading)

    return None


def _bfs_nearest_unvisited(start, visited):
    """
    BFS fallback: หาเส้นทางสั้นสุดไปช่องที่ยังไม่ได้เยี่ยมชม (ไม่มี turn penalty)
    """
    queue = deque([start])
    parent = {start: None}

    while queue:
        curr = queue.popleft()

        if curr not in visited:
            path = []
            c = curr
            while c is not None:
                path.append(c)
                c = parent[c]
            return list(reversed(path))

        for d in DIRECTIONS:
            if is_wall_between(curr, d):
                continue

            nxt = adjacent(curr, d)
            if in_bounds(nxt) and nxt not in parent:
                parent[nxt] = curr
                queue.append(nxt)

    return None
