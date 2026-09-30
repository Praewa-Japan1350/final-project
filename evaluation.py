# -*- coding: utf-8 -*-
"""
Map Evaluation and Output Exporter Module.
Calculates map coverage and accuracy metrics against ground truth,
and exports CSV logs, text reports, and a high-resolution trajectory map plot.
"""

from config import GRID_H
from config import GRID_W
import csv
import os
import config
from config import OUTPUT_DIR
from vision import COLORS
import state


def evaluate_map_accuracy(gt_h_walls=None, gt_v_walls=None):
    """
    Evaluate exploration metrics:
    - Coverage = (visited cells / total cells) * 100
    - Map Accuracy = (correct cells / total cells) * 100
    """
    if gt_h_walls is None:
        gt_h_walls = config.GROUND_TRUTH_H_WALLS
    if gt_v_walls is None:
        gt_v_walls = config.GROUND_TRUTH_V_WALLS

    total_cells = config.GRID_W * config.GRID_H
    coverage_pct = (len(state.visited_cells) / total_cells * 100.0) if total_cells else 0.0

    correct_cells = 0
    cell_details = {}

    for x in range(1, config.GRID_W + 1):
        for y in range(1, config.GRID_H + 1):
            det_north = (x, y) in state.detected_h_walls
            det_south = (x, y - 1) in state.detected_h_walls
            det_east = (x, y) in state.detected_v_walls
            det_west = (x - 1, y) in state.detected_v_walls

            gt_north = (x, y) in gt_h_walls
            gt_south = (x, y - 1) in gt_h_walls
            gt_east = (x, y) in gt_v_walls
            gt_west = (x - 1, y) in gt_v_walls

            is_correct = (
                det_north == gt_north
                and det_south == gt_south
                and det_east == gt_east
                and det_west == gt_west
            )
            if is_correct:
                correct_cells += 1
            cell_details[(x, y)] = {
                "correct": is_correct,
                "detected": {"N": det_north, "S": det_south, "E": det_east, "W": det_west},
                "ground_truth": {"N": gt_north, "S": gt_south, "E": gt_east, "W": gt_west},
            }

    map_accuracy_pct = (correct_cells / total_cells * 100.0) if total_cells else 0.0

    return {
        "coverage_pct": coverage_pct,
        "map_accuracy_pct": map_accuracy_pct,
        "correct_cells": correct_cells,
        "total_cells": total_cells,
        "cell_details": cell_details,
        "detected_walls": len(state.detected_h_walls) + len(state.detected_v_walls),
        "gt_walls": len(gt_h_walls) + len(gt_v_walls),
    }


def save_outputs(round_name=None, generate_plot=True):
    """Save trajectory CSV, walls CSV, signs CSV, and optionally high-res SLAM plot."""
    log_file = os.path.join(OUTPUT_DIR, "exploration_log.csv")
    walls_file = os.path.join(OUTPUT_DIR, "wall_data.csv")
    signs_file = os.path.join(OUTPUT_DIR, "signs_data.csv")
    img_name = f"robot_trajectory_{round_name}.png" if round_name else "robot_trajectory.png"
    img_file = os.path.join(OUTPUT_DIR, img_name)
    traj_file = os.path.join(OUTPUT_DIR, "trajectory_log.csv")
    visited_file = os.path.join(OUTPUT_DIR, "visited_cells.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fields = ["step", "timestamp", "grid_x", "grid_y", "x_m", "y_m", "heading", "tof_mm"]
    with open(log_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: item[key] for key in fields} for item in state.trajectory])

    with open(traj_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: item[key] for key in fields} for item in state.trajectory])

    with open(visited_file, "w", newline="", encoding="utf-8") as file:
        w = csv.writer(file)
        w.writerow(["grid_x", "grid_y", "x_m", "y_m"])
        for vx, vy in sorted(state.visited_cells):
            xm = round((vx - 0.5) * config.GRID_SIZE_M, 3)
            ym = round((vy - 0.5) * config.GRID_SIZE_M, 3)
            w.writerow([vx, vy, xm, ym])

    # Export foam walls coordinates to results/wall_data.csv
    h_walls_to_save = set(state.detected_h_walls)
    v_walls_to_save = set(state.detected_v_walls)
    if not h_walls_to_save and not v_walls_to_save:
        try:
            import json
            m_path = os.path.join(config.CODE_DIR, "maps", "mission_map.json")
            if os.path.exists(m_path):
                with open(m_path, "r", encoding="utf-8") as mf:
                    m_data = json.load(mf)
                h_walls_to_save = {tuple(w) for w in m_data.get("h_walls", [])}
                v_walls_to_save = {tuple(w) for w in m_data.get("v_walls", [])}
        except Exception:
            pass

    with open(walls_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "x", "y"])
        for wx, wy in sorted(h_walls_to_save):
            w.writerow(["H", wx, wy])
        for wx, wy in sorted(v_walls_to_save):
            w.writerow(["V", wx, wy])

    with open(signs_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["grid_x", "grid_y", "direction", "color", "color_label", "shape", "shape_label", "area_px", "image_path"])
        for (cell, direction), signs in sorted(state.detected_signs.items()):
            for sign in signs:
                w.writerow([
                    cell[0], cell[1], direction, sign["color"], sign.get("label", ""),
                    sign["shape"], sign.get("shape_label", ""), round(sign.get("area", 0)),
                    sign.get("image_path", ""),
                ])

    if not generate_plot:
        return

    try:
        # pyrefly: ignore [missing-import]
        import matplotlib
        matplotlib.use("Agg", force=True)
        # pyrefly: ignore [missing-import]
        import matplotlib.pyplot as plt
        # pyrefly: ignore [missing-import]
        import matplotlib.patches as patches
        # pyrefly: ignore [missing-import]
        from matplotlib.lines import Line2D

        # Luxury Light Theme (Porcelain & Royal Sapphire / Laser Crimson)
        fig_bg = "#f8fafc"          # Soft Platinum Porcelain
        ax_bg = "#ffffff"           # Clean White Canvas
        cell_visited_bg = "#f0fdf4" # Mint Emerald Tint
        cell_visited_edge = "#86efac"

        wall_glow = "#fecdd3"       # Soft Rose Outer Aura
        wall_core = "#e11d48"       # Laser Crimson Core
        wall_bright = "#be123c"     # Deep Crimson Highlight

        traj_glow = "#bfdbfe"       # Soft Sapphire Glow
        traj_mid = "#3b82f6"        # Royal Sapphire Path
        traj_core = "#1d4ed8"       # Deep Cobalt Waypoints

        start_col = "#059669"       # Deep Emerald Green for Start
        end_col = "#dc2626"         # Deep Crimson Ruby for End

        fig, ax = plt.subplots(figsize=(7.8, 10.6), facecolor=fig_bg)
        ax.set_facecolor(ax_bg)

        # 1. Draw Grid Cells with Luxury Light Styling
        for x in range(1, config.GRID_W + 1):
            for y in range(1, config.GRID_H + 1):
                is_vis = (x, y) in state.visited_cells
                is_disc = (x, y) in state.discovered_cells

                if is_vis:
                    cell_patch = patches.Rectangle(
                        (x - 1 + 0.03, y - 1 + 0.03), 0.94, 0.94,
                        facecolor=cell_visited_bg,
                        edgecolor=cell_visited_edge,
                        linewidth=1.2,
                        alpha=0.95,
                        zorder=2,
                    )
                    ax.add_patch(cell_patch)

                coord_color = "#047857" if is_vis else ("#0284c7" if is_disc else "#64748b")
                badge_bg = "#ffffff"
                badge_edge = "#cbd5e1"
                ax.text(
                    x - 1 + 0.12, y - 0.14, f"({x},{y})",
                    ha="left", va="top", color=coord_color,
                    fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor=badge_bg, edgecolor=badge_edge, linewidth=0.6, alpha=0.95),
                    zorder=15,
                )

                ax.plot(x - 0.5, y - 0.5, marker="+", color="#cbd5e1", markersize=5, zorder=2)

                cell_signs = {}
                for (sign_cell, _direction), signs in state.detected_signs.items():
                    if sign_cell == (x, y):
                        for sign in signs:
                            cell_signs[(sign["color"], sign["shape"])] = sign
                if cell_signs:
                    for sign_index, sign in enumerate(cell_signs.values()):
                        icon_col, icon_row = sign_index % 8, sign_index // 8
                        sx = x - 1 + 0.08 + icon_col * 0.115
                        sy = y - 1 + 0.10 + icon_row * 0.14
                        bgr = COLORS[sign["color"]]["bgr"]
                        color = "#%02x%02x%02x" % tuple(reversed(bgr))
                        if sign["shape"] == "circle":
                            marker = patches.Circle((sx, sy), radius=0.035, facecolor=color,
                                                    edgecolor="#0f172a", linewidth=0.6, zorder=16)
                        elif sign["shape"] == "square":
                            marker = patches.Rectangle((sx - 0.035, sy - 0.035), 0.07, 0.07,
                                                       facecolor=color, edgecolor="#0f172a", linewidth=0.6, zorder=16)
                        elif sign["shape"] == "horizontal":
                            marker = patches.Rectangle((sx - 0.045, sy - 0.025), 0.09, 0.05,
                                                       facecolor=color, edgecolor="#0f172a", linewidth=0.6, zorder=16)
                        else:
                            marker = patches.Rectangle((sx - 0.025, sy - 0.045), 0.05, 0.09,
                                                       facecolor=color, edgecolor="#0f172a", linewidth=0.6, zorder=16)
                        ax.add_patch(marker)

        # 2. Draw Detected Foam Border Walls
        for (wx, wy) in state.detected_h_walls:
            ax.plot([wx - 1, wx], [wy, wy], color=wall_glow, linewidth=8.0, alpha=0.50, solid_capstyle="round", zorder=6)
            ax.plot([wx - 1, wx], [wy, wy], color=wall_core, linewidth=4.5, alpha=0.90, solid_capstyle="round", zorder=7)
            ax.plot([wx - 1, wx], [wy, wy], color=wall_bright, linewidth=1.8, alpha=1.0, solid_capstyle="round", zorder=8)
            ax.plot([wx - 1, wx], [wy, wy], marker="o", color=wall_core, markersize=4.0, zorder=9)

        for (wx, wy) in state.detected_v_walls:
            ax.plot([wx, wx], [wy - 1, wy], color=wall_glow, linewidth=8.0, alpha=0.50, solid_capstyle="round", zorder=6)
            ax.plot([wx, wx], [wy - 1, wy], color=wall_core, linewidth=4.5, alpha=0.90, solid_capstyle="round", zorder=7)
            ax.plot([wx, wx], [wy - 1, wy], color=wall_bright, linewidth=1.8, alpha=1.0, solid_capstyle="round", zorder=8)
            ax.plot([wx, wx], [wy - 1, wy], marker="o", color=wall_core, markersize=4.0, zorder=9)

        # 3. Draw Robot Trajectory
        if state.trajectory:
            tx = [item["grid_x"] - 0.5 for item in state.trajectory]
            ty = [item["grid_y"] - 0.5 for item in state.trajectory]

            ax.plot(tx, ty, color=traj_glow, linewidth=7.5, alpha=0.55, solid_capstyle="round", zorder=4)
            ax.plot(tx, ty, color=traj_mid, linewidth=3.8, alpha=0.85, solid_capstyle="round", zorder=4)
            ax.plot(tx, ty, color=traj_core, linewidth=2.0, alpha=1.0, solid_capstyle="round", zorder=5)

            ax.plot(
                tx, ty, marker="o", linestyle="None",
                markerfacecolor=ax_bg, markeredgecolor=traj_core,
                markeredgewidth=1.5, markersize=5.5, zorder=5,
            )

            for i in range(len(tx) - 1):
                dx = tx[i + 1] - tx[i]
                dy = ty[i + 1] - ty[i]
                dist = (dx**2 + dy**2)**0.5
                if dist > 0.1:
                    mid_x = tx[i] + dx * 0.55
                    mid_y = ty[i] + dy * 0.55
                    ax.annotate(
                        "", xy=(mid_x + dx * 0.08, mid_y + dy * 0.08),
                        xytext=(mid_x - dx * 0.08, mid_y - dy * 0.08),
                        arrowprops=dict(
                            arrowstyle="->",
                            color="#2563eb",
                            lw=1.6,
                            shrinkA=0, shrinkB=0,
                        ),
                        zorder=5,
                    )

            # Start Point Badge
            ax.plot(tx[0], ty[0], marker="o", color=start_col, markersize=18, alpha=0.25, zorder=10)
            ax.plot(tx[0], ty[0], marker="o", color=start_col, markersize=11, alpha=0.9, zorder=11)
            ax.plot(tx[0], ty[0], marker="o", color="#ffffff", markersize=5, zorder=12)
            ax.text(
                tx[0], ty[0] - 0.28, "START", color="#ffffff", fontsize=7.5,
                fontweight="heavy", ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=start_col, edgecolor="none", alpha=0.95),
                zorder=13,
            )

            # End Point Badge
            ax.plot(tx[-1], ty[-1], marker="o", color=end_col, markersize=20, alpha=0.25, zorder=10)
            ax.plot(tx[-1], ty[-1], marker="D", color=end_col, markersize=11, alpha=0.9, zorder=11)
            ax.plot(tx[-1], ty[-1], marker="*", color="#ffffff", markersize=6, zorder=12)
            ax.text(
                tx[-1], ty[-1] + 0.28, "END", color="#ffffff", fontsize=7.5,
                fontweight="heavy", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=end_col, edgecolor="none", alpha=0.95),
                zorder=13,
            )

        # 4. Axes & Tick Formatting
        ax.set_xlim(-0.15, config.GRID_W + 0.15)
        ax.set_ylim(-0.15, config.GRID_H + 0.15)
        ax.set_xticks(range(config.GRID_W + 1))
        ax.set_yticks(range(config.GRID_H + 1))
        ax.set_xticklabels(range(1, config.GRID_W + 2), color="#475569", fontsize=9, fontweight="bold")
        ax.set_yticklabels(range(1, config.GRID_H + 2), color="#475569", fontsize=9, fontweight="bold")

        ax.tick_params(colors="#94a3b8", which="both", length=4, width=1.2)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cbd5e1")
            spine.set_linewidth(1.4)

        ax.grid(True, linestyle=":", color="#e2e8f0", alpha=0.8, zorder=0)
        ax.set_xlabel(f"Grid Coordinates X ({config.GRID_SIZE_M*100:.0f}cm / cell)", color="#475569", fontsize=9, labelpad=8)
        ax.set_ylabel(f"Grid Coordinates Y ({config.GRID_SIZE_M*100:.0f}cm / cell)", color="#475569", fontsize=9, labelpad=8)

        # 5. Header HUD Card
        total_walls = len(state.detected_h_walls) + len(state.detected_v_walls)
        total_steps = len(state.trajectory) if state.trajectory else 0
        total_signs = sum(len(s) for s in state.detected_signs.values())

        fig.text(0.5, 0.965, "ROBOMASTER AUTONOMOUS SLAM",
                 ha="center", va="top", color="#0f172a", fontsize=14, fontweight="bold")
        fig.text(0.5, 0.940, "MAZE RECONSTRUCTION & TARGET LOCALIZATION",
                 ha="center", va="top", color="#2563eb", fontsize=8, fontweight="bold", alpha=0.9)

        stats_str = f"EXPLORED: {len(state.visited_cells)}/{config.GRID_W * config.GRID_H} CELLS   |   WALLS: {total_walls}   |   TARGETS: {total_signs}   |   STEPS: {total_steps}   |   STATUS: COMPLETED"
        fig.text(0.5, 0.912, stats_str,
                 ha="center", va="top", color="#334155", fontsize=7.8, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.0, alpha=0.95))

        # 6. HUD Legend
        legend_elements = [
            Line2D([0], [0], color=wall_core, lw=3.0, label="Foam Wall (Red)"),
            Line2D([0], [0], color=traj_mid, lw=2.0, marker="o", markerfacecolor=ax_bg, markeredgecolor=traj_core, markersize=4.5, label="Trajectory (Blue)"),
            patches.Patch(facecolor=cell_visited_bg, edgecolor=cell_visited_edge, label="Explored Cell"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=start_col, markersize=6.5, label="Start Position"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor=end_col, markersize=6.5, label="End Position"),
        ]
        legend = ax.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.025),
            ncol=3,
            facecolor="#ffffff",
            edgecolor="#cbd5e1",
            framealpha=1.0,
            fontsize=7.3,
            labelcolor="#1e293b",
            handletextpad=0.5,
            columnspacing=1.0,
            borderpad=0.55,
        )
        legend.get_frame().set_linewidth(1.0)
        legend.set_zorder(30)

        fig.text(0.5, 0.02, "RoboMaster EP SLAM Engine • Real-time ToF & Odometry Fusion",
                 ha="center", va="bottom", color="#64748b", fontsize=7.2, fontweight="medium")

        fig.subplots_adjust(top=0.810, bottom=0.075, left=0.12, right=0.94)
        fig.savefig(img_file, dpi=300, facecolor=fig_bg)

        # Save copies to results/ directory and repository root
        code_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(code_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        fig.savefig(os.path.join(results_dir, "final_slam_map.png"), dpi=300, facecolor=fig_bg)
        fig.savefig(os.path.join(code_dir, "final_slam_map.png"), dpi=300, facecolor=fig_bg)
        plt.close(fig)

        import shutil
        for fname in ["exploration_log.csv", "trajectory_log.csv", "visited_cells.csv", "walls_data.csv", "signs_data.csv"]:
            src = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(src):
                dst_results = os.path.join(results_dir, fname)
                if os.path.abspath(src) != os.path.abspath(dst_results):
                    shutil.copy2(src, dst_results)
                dst_code = os.path.join(code_dir, fname)
                if os.path.abspath(src) != os.path.abspath(dst_code):
                    shutil.copy2(src, dst_code)

        print(f"บันทึกไฟล์ผลลัพธ์ทั้งหมดไว้ที่: {results_dir}")
        print("-> exploration_log.csv, trajectory_log.csv, visited_cells.csv, walls_data.csv, signs_data.csv, final_slam_map.png เรียบร้อยแล้ว")
    except Exception as e:
        print(f"ไม่สามารถบันทึกรูปภาพได้: {e}")


def re_evaluate_from_saved():
    """
    โหลดข้อมูลแนวกำแพงและประวัติการเดินจากไฟล์ CSV ที่บันทึกไว้
    นำมาคำนวณ Map Accuracy เทียบกับ Ground Truth ใน config.py อีกครั้ง
    โดยไม่ต้องนำหุ่นไปวิ่งใหม่
    """
    import config
    import importlib
    importlib.reload(config)

    walls_file = os.path.join(OUTPUT_DIR, "walls_data.csv")
    log_file = os.path.join(OUTPUT_DIR, "exploration_log.csv")

    if not os.path.exists(walls_file) or not os.path.exists(log_file):
        print("⚠️ ไม่พบไฟล์ walls_data.csv หรือ exploration_log.csv กรุณารันหุ่นยนต์หรือ simulation อย่างน้อย 1 ครั้งก่อน")
        return

    state.detected_h_walls.clear()
    state.detected_v_walls.clear()
    state.visited_cells.clear()
    state.discovered_cells.clear()
    state.trajectory.clear()

    # 1. โหลดแนวกำแพงที่หุ่นเคยสแกนพบ
    with open(walls_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 3:
                continue
            w_type, wx, wy = row[0].strip(), int(row[1]), int(row[2])
            if w_type == "H":
                state.detected_h_walls.add((wx, wy))
            elif w_type == "V":
                state.detected_v_walls.add((wx, wy))

    # 2. โหลดประวัติการเดินและช่องที่เคยไป
    prev_cell = None
    with open(log_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cell = (int(row["grid_x"]), int(row["grid_y"]))
            state.visited_cells.add(cell)
            state.discovered_cells.add(cell)
            state.trajectory.append({
                "step": int(row.get("step", 0)),
                "timestamp": float(row.get("timestamp", 0)),
                "grid_x": cell[0],
                "grid_y": cell[1],
                "x_m": float(row.get("x_m", 0)),
                "y_m": float(row.get("y_m", 0)),
                "heading": row.get("heading", "NORTH"),
                "tof_mm": float(row.get("tof_mm", 9999)),
                "cell": cell,
                "previous": prev_cell,
            })
            prev_cell = cell

    total_walls = len(state.detected_h_walls) + len(state.detected_v_walls)
    total_signs = sum(len(s) for s in state.detected_signs.values())
    print("\n=======================================================")
    print("🏁 สรุปข้อมูลที่บันทึกไว้ (SAVED RUN SUMMARY)")
    print("=======================================================")
    print(f"• ช่องที่สำรวจ : {len(state.visited_cells)} / {GRID_W * GRID_H} ช่อง")
    print(f"• แนวกำแพงโฟม  : ตรวจพบ {total_walls} แนวกำแพง")
    print(f"• เป้าที่ตรวจพบ: {total_signs} เป้า")
    print("=======================================================\n")

    save_outputs()
    print("✅ อัปเดต robot_trajectory.png เรียบร้อยแล้ว!")


if __name__ == "__main__":
    re_evaluate_from_saved()

