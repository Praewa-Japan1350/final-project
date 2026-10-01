# -*- coding: utf-8 -*-
"""
Tkinter GUI Dashboard Module for RoboMaster SLAM.
Provides real-time visualization of the grid map, detected foam walls,
robot pose, gimbal scanning direction, and telemetry control panels.
"""

import os
import csv
import threading
import queue
import tkinter as tk
import time
import traceback

import cv2
from PIL import Image, ImageTk

import config
from config import DELTA, DIRECTIONS, SYMBOL, in_bounds
import vision
from vision import COLORS
import state


class Dashboard:
    def __init__(self, on_start=None, on_stop=None, sim_mode=False, default_start=(1, 1, "NORTH")):
        self.root = tk.Tk()
        self.root.title("RoboMaster Autonomous SLAM - Dashboard")
        self.root.geometry("1400x820")
        self.root.minsize(1120, 680)
        self.root.configure(bg="#f1f5f9")
        self.closed = False
        self.is_running = False
        self.on_start = on_start
        self.on_stop = on_stop
        self.sim_mode = sim_mode
        self.run_mode = "mapping"
        self._camera_lock = threading.Lock()
        self._camera_pending_frame = None
        self._camera_callback_pending = False
        self._aim_reticle_active = False
        self._camera_busy = False
        self._last_camera_frame = None
        self._last_camera_display = 0.0
        self._last_camera_source_frame_id = -1
        self.vision_reader = None
        self.vision_analyzer = None
        self._last_filter_frame_id = -1
        self._filter_photo = None
        self._ui_queue = queue.Queue(maxsize=1024)
        self._ui_coalesce_lock = threading.Lock()
        self._ui_coalesced = {}
        self._ui_coalesced_scheduled = set()
        self._last_camera_error = None

        # Stopwatch & Round Timing State
        self._timer_running = False
        self._timer_start_time = None
        self._mission_start_time = None
        self._active_round_key = None       # "round1" or "round2"
        self._active_round_label = ""
        self._round_times = {"round1": None, "round2": None, "total": None}
        self._timer_tick_id = None

        self.pad = 28
        max_dim = max(config.GRID_W, config.GRID_H)
        max_canvas_side = 460
        self.cell_size = max(40, min(100, int((max_canvas_side - 2 * self.pad) / max_dim)))
        canvas_w = self.pad * 2 + self.cell_size * config.GRID_W
        canvas_h = self.pad * 2 + self.cell_size * config.GRID_H

        # Variables for start configuration
        self.start_x_var = tk.IntVar(value=default_start[0])
        self.start_y_var = tk.IntVar(value=default_start[1])
        self.start_heading_var = tk.StringVar(value=default_start[2])

        self.start_x_var.trace_add("write", lambda *args: self.redraw_preview())
        self.start_y_var.trace_add("write", lambda *args: self.redraw_preview())
        self.start_heading_var.trace_add("write", lambda *args: self.redraw_preview())

        # Main Layout: Left Column (Grid Map & Live Camera), Right Column (Control & Status Panel)
        self.map_container = tk.Frame(self.root, bg="#f1f5f9")
        self.map_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=(14, 8), pady=12)

        self.canvas = tk.Canvas(
            self.map_container,
            width=canvas_w,
            height=canvas_h,
            bg="#ffffff",
            highlightthickness=1,
            highlightbackground="#cbd5e1",
            cursor="hand2",
        )
        self.canvas.pack(anchor="nw")
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Live Camera Frame below the Map Canvas
        self.cam_frame = tk.LabelFrame(
            self.map_container,
            text=" 📷 ภาพจากกล้องสด (Camera Feed) ",
            bg="#ffffff",
            fg="#0f172a",
            font=("Segoe UI", 9, "bold"),
            padx=4,
            pady=4,
            highlightbackground="#cbd5e1",
            highlightthickness=1,
        )
        self.cam_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.camera_label = tk.Label(
            self.cam_frame,
            text="📷 กำลังรอกล้อง RoboMaster...",
            bg="#0f172a",
            fg="#94a3b8",
            highlightbackground="#cbd5e1",
            highlightthickness=1,
        )
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        panel = tk.Frame(self.root, bg="#f8fafc", padx=16, pady=14, highlightbackground="#e2e8f0", highlightthickness=1)
        panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 16), pady=16)

        # Header Title Card
        title_text = "ROBOMASTER SLAM · SIMULATION" if sim_mode else "ROBOMASTER SLAM · ROBOT EXPLORER"
        tk.Label(
            panel,
            text=title_text,
            bg="#f8fafc",
            fg="#0f172a",
            font=("Segoe UI", 15, "bold"),
        ).pack(anchor="w", pady=(0, 2))
        tk.Label(
            panel,
            text="Assignment 2 · 6×6 Maze Mapping & Autonomous Blaster System",
            bg="#f8fafc",
            fg="#2563eb",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        # Maze Dimensions & Start Configuration Box
        self.config_frame = tk.LabelFrame(
            panel,
            text=" ⚙️ กำหนดขนาดแผนที่ & จุดเริ่มต้น (Dimensions & Start Config) ",
            bg="#ffffff",
            fg="#0f172a",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=8,
            highlightbackground="#cbd5e1",
            highlightthickness=1,
        )
        self.config_frame.pack(fill=tk.X, pady=(0, 8))

        # Row 1: Maze Grid Dimensions (W x H) and Pitch
        size_row = tk.Frame(self.config_frame, bg="#ffffff")
        size_row.pack(fill=tk.X, pady=(0, 4))

        tk.Label(size_row, text="ขนาดตาราง กว้าง(W):", bg="#ffffff", fg="#334155", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 3))
        self.grid_w_var = tk.IntVar(value=config.GRID_W)
        self.spin_w = tk.Spinbox(
            size_row, from_=2, to=12, textvariable=self.grid_w_var, width=3,
            bg="#f8fafc", fg="#0f172a", font=("Segoe UI", 9, "bold"),
            buttonbackground="#e2e8f0", justify=tk.CENTER, relief=tk.SOLID, bd=1,
            command=self._on_grid_dimensions_changed,
        )
        self.spin_w.pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(size_row, text="ยาว(H):", bg="#ffffff", fg="#334155", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 3))
        self.grid_h_var = tk.IntVar(value=config.GRID_H)
        self.spin_h = tk.Spinbox(
            size_row, from_=2, to=12, textvariable=self.grid_h_var, width=3,
            bg="#f8fafc", fg="#0f172a", font=("Segoe UI", 9, "bold"),
            buttonbackground="#e2e8f0", justify=tk.CENTER, relief=tk.SOLID, bd=1,
            command=self._on_grid_dimensions_changed,
        )
        self.spin_h.pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(size_row, text="ขนาดช่อง (ม.):", bg="#ffffff", fg="#334155", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 3))
        self.grid_pitch_var = tk.DoubleVar(value=config.GRID_SIZE_M)
        self.spin_pitch = tk.Spinbox(
            size_row, from_=0.30, to=1.50, increment=0.05, format="%.2f",
            textvariable=self.grid_pitch_var, width=5,
            bg="#f8fafc", fg="#0f172a", font=("Segoe UI", 9, "bold"),
            buttonbackground="#e2e8f0", justify=tk.CENTER, relief=tk.SOLID, bd=1,
            command=self._on_grid_dimensions_changed,
        )
        self.spin_pitch.pack(side=tk.LEFT, padx=(0, 4))

        self.grid_w_var.trace_add("write", lambda *args: self._on_grid_dimensions_changed())
        self.grid_h_var.trace_add("write", lambda *args: self._on_grid_dimensions_changed())
        self.grid_pitch_var.trace_add("write", lambda *args: self._on_grid_dimensions_changed())

        # Row 2: Start Coordinates and Heading
        coord_row = tk.Frame(self.config_frame, bg="#ffffff")
        coord_row.pack(fill=tk.X, pady=2)

        tk.Label(coord_row, text="Start X:", bg="#ffffff", fg="#334155", font=("Segoe UI", 9, "bold")).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        self.spin_x = tk.Spinbox(
            coord_row,
            from_=1,
            to=config.GRID_W,
            textvariable=self.start_x_var,
            width=3,
            bg="#f8fafc",
            fg="#0f172a",
            font=("Segoe UI", 9, "bold"),
            buttonbackground="#e2e8f0",
            justify=tk.CENTER,
            relief=tk.SOLID,
            bd=1,
        )
        self.spin_x.pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(coord_row, text="Start Y:", bg="#ffffff", fg="#334155", font=("Segoe UI", 9, "bold")).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        self.spin_y = tk.Spinbox(
            coord_row,
            from_=1,
            to=config.GRID_H,
            textvariable=self.start_y_var,
            width=3,
            bg="#f8fafc",
            fg="#0f172a",
            font=("Segoe UI", 9, "bold"),
            buttonbackground="#e2e8f0",
            justify=tk.CENTER,
            relief=tk.SOLID,
            bd=1,
        )
        self.spin_y.pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(coord_row, text="ทิศทาง:", bg="#ffffff", fg="#334155", font=("Segoe UI", 9, "bold")).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        self.heading_menu = tk.OptionMenu(coord_row, self.start_heading_var, *DIRECTIONS)
        self.heading_menu.config(
            bg="#f8fafc",
            fg="#0f172a",
            font=("Segoe UI", 8, "bold"),
            activebackground="#e2e8f0",
            activeforeground="#0f172a",
            highlightthickness=1,
            highlightbackground="#cbd5e1",
            relief=tk.SOLID,
            bd=1,
        )
        self.heading_menu["menu"].config(bg="#ffffff", fg="#0f172a")
        self.heading_menu.pack(side=tk.LEFT)

        self.size_info_label = tk.Label(
            self.config_frame,
            text=f"ขนาดพื้นที่: {config.GRID_W} × {config.GRID_H} ช่อง ({config.GRID_W * config.GRID_SIZE_M:.2f} × {config.GRID_H * config.GRID_SIZE_M:.2f} m) · คลิกช่องในแผนที่เพื่อเลือกจุดเริ่มต้น",
            bg="#ffffff",
            fg="#64748b",
            font=("Segoe UI", 8),
        )
        self.size_info_label.pack(anchor="w", pady=(4, 0))

        # Target Selection & Filter Box
        filters = tk.LabelFrame(panel, text=" 🎯 เลือกเป้าหมาย & ตั้งค่าการตรวจจับ (Target Selection) ",
                                bg="#ffffff", fg="#0f172a",
                                font=("Segoe UI", 10, "bold"),
                                padx=10, pady=6,
                                highlightbackground="#cbd5e1", highlightthickness=1)
        filters.pack(fill=tk.X, pady=(0, 8))
        self.color_vars = {name: tk.BooleanVar(value=True) for name in COLORS}
        self.shape_vars = {name: tk.BooleanVar(value=True) for name in
                           ("circle", "square", "horizontal", "vertical")}
        self.vision_controls = []
        row = tk.Frame(filters, bg="#ffffff")
        row.pack(fill=tk.X, pady=1)
        tk.Label(row, text="สีเป้าหมาย: ", bg="#ffffff", fg="#334155", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        for name, var in self.color_vars.items():
            control = tk.Checkbutton(row, text=name, variable=var, bg="#ffffff", fg="#0f172a",
                                     selectcolor="#f1f5f9", activebackground="#ffffff",
                                     font=("Segoe UI", 9))
            control.pack(side=tk.LEFT, padx=3)
            self.vision_controls.append(control)
        row = tk.Frame(filters, bg="#ffffff")
        row.pack(fill=tk.X, pady=1)
        tk.Label(row, text="รูปร่างเป้า: ", bg="#ffffff", fg="#334155", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        for name, var in self.shape_vars.items():
            control = tk.Checkbutton(row, text=name, variable=var, bg="#ffffff", fg="#0f172a",
                                     selectcolor="#f1f5f9", activebackground="#ffffff",
                                     font=("Segoe UI", 9))
            control.pack(side=tk.LEFT, padx=3)
            self.vision_controls.append(control)
        # Buttons Frame
        # Buttons Frame - Row 1: Primary Round 1 & Round 2 Controls
        btn_frame = tk.Frame(panel, bg="#f8fafc")
        btn_frame.pack(fill=tk.X, pady=(0, 4))

        self.btn_slam = tk.Button(
            btn_frame,
            text="🗺️ รอบ 1: เดินแบบ SLAM",
            bg="#2563eb",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=8,
            pady=6,
            cursor="hand2",
            command=lambda: self.handle_start_click("slam_only"),
        )
        self.btn_slam.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.btn_astar = tk.Button(
            btn_frame,
            text="⭐ รอบ 2: เดินแบบ A*",
            bg="#7c3aed",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            activebackground="#6d28d9",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=8,
            pady=6,
            cursor="hand2",
            command=lambda: self.handle_start_click("astar_only"),
        )
        self.btn_astar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.btn_stop = tk.Button(
            btn_frame,
            text="⏹ หยุดฉุกเฉิน (Stop)",
            bg="#e11d48",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            activebackground="#be123c",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=6,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.handle_stop_click,
        )
        self.btn_stop.pack(side=tk.RIGHT, padx=(2, 0))

        # Buttons Frame - Row 2: Continuous (Round 1 ➔ Round 2), Recovery/Resume, & CSV Map Upload
        btn_row2 = tk.Frame(panel, bg="#f8fafc")
        btn_row2.pack(fill=tk.X, pady=(0, 6))

        self.btn_auto_all = tk.Button(
            btn_row2,
            text="🚀 รันต่อเนื่อง (รอบ 1➔2)",
            bg="#059669",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            activebackground="#047857",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=6,
            pady=5,
            cursor="hand2",
            command=lambda: self.handle_start_click("auto_all"),
        )
        self.btn_auto_all.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.btn_resume = tk.Button(
            btn_row2,
            text="🔄 กู้คืน/รันต่อ (Resume)",
            bg="#0284c7",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            activebackground="#0369a1",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=6,
            pady=5,
            cursor="hand2",
            command=lambda: self.handle_start_click("resume"),
        )
        self.btn_resume.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.btn_load_csv = tk.Button(
            btn_row2,
            text="📂 โหลด CSV",
            bg="#ea580c",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            activebackground="#c2410c",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=6,
            pady=5,
            cursor="hand2",
            command=self._load_csv_map,
        )
        self.btn_load_csv.pack(side=tk.RIGHT, padx=(2, 0))

        # Compatibility aliases
        self.btn_start = self.btn_slam
        self.btn_target = self.btn_astar

        # CSV Status Label Row
        csv_status_row = tk.Frame(panel, bg="#f8fafc")
        csv_status_row.pack(fill=tk.X, pady=(0, 4))
        self.csv_status_label = tk.Label(
            csv_status_row, text="สถานะแผนที่: ใช้การสำรวจสด หรือคลิก 'โหลดแผนที่ CSV'",
            bg="#f8fafc", fg="#64748b", font=("Segoe UI", 8),
        )
        self.csv_status_label.pack(side=tk.LEFT, padx=(2, 0))

        # Timer / Stopwatch Card for Each Round
        self.timer_card = tk.LabelFrame(
            panel,
            text=" ⏱️ ระบบจับเวลาภารกิจแต่ละรอบ (Mission & Round Stopwatch) ",
            bg="#ffffff",
            fg="#0f172a",
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=5,
            highlightbackground="#cbd5e1",
            highlightthickness=1,
        )
        self.timer_card.pack(fill=tk.X, pady=(0, 6))

        # Top row: Big digital timer + active state badge + reset button
        timer_top = tk.Frame(self.timer_card, bg="#ffffff")
        timer_top.pack(fill=tk.X, pady=(0, 4))

        self.lbl_timer_main = tk.Label(
            timer_top,
            text="00:00.0",
            bg="#0f172a",
            fg="#38bdf8",
            font=("Consolas", 15, "bold"),
            padx=8,
            pady=2,
            relief=tk.FLAT,
        )
        self.lbl_timer_main.pack(side=tk.LEFT, padx=(0, 8))

        self.lbl_timer_status = tk.Label(
            timer_top,
            text="⚪ พร้อมจับเวลา (กดรันเพื่อเริ่มทันที)",
            bg="#ffffff",
            fg="#64748b",
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        )
        self.lbl_timer_status.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.btn_timer_reset = tk.Button(
            timer_top,
            text="🔄 รีเซ็ตเวลา",
            command=self.reset_timer,
            bg="#f1f5f9",
            fg="#475569",
            activebackground="#e2e8f0",
            activeforeground="#0f172a",
            font=("Segoe UI", 8, "bold"),
            relief=tk.SOLID,
            bd=1,
            padx=6,
            pady=2,
            cursor="hand2",
        )
        self.btn_timer_reset.pack(side=tk.RIGHT)

        # Bottom row: Recorded time for each round
        timer_rounds = tk.Frame(self.timer_card, bg="#ffffff")
        timer_rounds.pack(fill=tk.X)

        self.lbl_round1_time = tk.Label(
            timer_rounds,
            text="🗺️ รอบ 1 (SLAM): --:--.-",
            bg="#f8fafc",
            fg="#64748b",
            font=("Segoe UI", 8, "bold"),
            padx=6,
            pady=3,
            relief=tk.SOLID,
            bd=1,
        )
        self.lbl_round1_time.pack(side=tk.LEFT, padx=(0, 4), fill=tk.X, expand=True)

        self.lbl_round2_time = tk.Label(
            timer_rounds,
            text="⭐ รอบ 2 (A*): --:--.-",
            bg="#f8fafc",
            fg="#64748b",
            font=("Segoe UI", 8, "bold"),
            padx=6,
            pady=3,
            relief=tk.SOLID,
            bd=1,
        )
        self.lbl_round2_time.pack(side=tk.LEFT, padx=(0, 4), fill=tk.X, expand=True)

        self.lbl_total_time = tk.Label(
            timer_rounds,
            text="⌛ เวลารวม: --:--.-",
            bg="#f8fafc",
            fg="#64748b",
            font=("Segoe UI", 8, "bold"),
            padx=6,
            pady=3,
            relief=tk.SOLID,
            bd=1,
        )
        self.lbl_total_time.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Status Label Card
        self.status = tk.Label(
            panel,
            justify=tk.LEFT,
            anchor="nw",
            bg="#ffffff",
            fg="#0f172a",
            font=("Consolas", 9),
            padx=10,
            pady=6,
            highlightbackground="#cbd5e1",
            highlightthickness=1,
            bd=0,
        )
        self.status.pack(fill=tk.X, pady=(0, 6))

        # Results Toolbar Card (Target gallery, Final Map, Results folder)
        results_toolbar = tk.Frame(panel, bg="#f8fafc")
        results_toolbar.pack(fill=tk.X, pady=(0, 6))
        tk.Label(
            results_toolbar, text="📊 เมนูผลลัพธ์ & รายงาน (Results & Reports)",
            bg="#f8fafc", fg="#0f172a", font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.LEFT)

        self.btn_open_folder = tk.Button(
            results_toolbar, text="📁 Results", command=self.open_results_folder,
            bg="#475569", fg="white", font=("Segoe UI", 8, "bold"), relief=tk.FLAT,
            padx=6, pady=3, cursor="hand2"
        )
        self.btn_open_folder.pack(side=tk.RIGHT, padx=(3, 0))

        self.btn_map_view = tk.Button(
            results_toolbar, text="🗺️ แผนที่รอบ 1 & 2", command=self.show_final_map_window,
            bg="#059669", fg="white", font=("Segoe UI", 8, "bold"), relief=tk.FLAT,
            padx=6, pady=3, cursor="hand2"
        )
        self.btn_map_view.pack(side=tk.RIGHT, padx=(3, 0))

        self.btn_gallery = tk.Button(
            results_toolbar, text="🖼️ ภาพเป้าหมาย (0)", command=self.show_target_gallery,
            bg="#7c3aed", fg="white", font=("Segoe UI", 8, "bold"), relief=tk.FLAT,
            padx=6, pady=3, cursor="hand2"
        )
        self.btn_gallery.pack(side=tk.RIGHT, padx=(3, 0))

        self.btn_clear_data = tk.Button(
            results_toolbar, text="🧹 เคลียร์ข้อมูล", command=self.clear_data_and_logs,
            bg="#d97706", fg="white", font=("Segoe UI", 8, "bold"), relief=tk.FLAT,
            padx=6, pady=3, cursor="hand2"
        )
        self.btn_clear_data.pack(side=tk.RIGHT, padx=(3, 0))

        self._camera_photo = None
        self.root.after(200, self._refresh_camera)
        self.root.after(30, self._drain_ui_queue)
        self.root.after(30000, self._periodic_cleanup)

        # Target Results Gallery Storage & Window
        self.target_results = []
        self._latest_target_photo = None
        self.gallery_window = None

        # Latest Target Snapshot Result Card
        self.target_card = tk.LabelFrame(
            panel,
            text=" 🎯 ผลลัพธ์ภาพเป้าหมายล่าสุด (Latest Target Snapshot) ",
            bg="#ffffff",
            fg="#0f172a",
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=4,
            highlightbackground="#cbd5e1",
            highlightthickness=1,
        )
        self.target_card.pack(fill=tk.X, pady=(0, 6))

        target_row = tk.Frame(self.target_card, bg="#ffffff")
        target_row.pack(fill=tk.X)

        self.target_thumb_label = tk.Label(
            target_row,
            text="ยังไม่พบเป้าหมาย\n(จะแสดงภาพเมื่อตรวจพบ/ยิง)",
            bg="#f1f5f9",
            fg="#64748b",
            font=("Segoe UI", 8),
            width=26,
            height=4,
            relief=tk.SOLID,
            bd=1,
        )
        self.target_thumb_label.pack(side=tk.LEFT, padx=(0, 8))

        self.target_info_label = tk.Label(
            target_row,
            text="เป้าหมาย: (รอตรวจจับ)\nพิกัด: -  |  ทิศทาง: -\nสถานะ: ยังไม่มีการยิง\nไฟล์ภาพ: -",
            bg="#ffffff",
            fg="#1e293b",
            font=("Consolas", 8),
            justify=tk.LEFT,
            anchor="w",
        )
        self.target_info_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Log Text Box Card
        log_frame = tk.LabelFrame(
            panel,
            text=" 📋 บันทึกการทำงาน (Mission Event Logs) ",
            bg="#ffffff",
            fg="#0f172a",
            font=("Segoe UI", 9, "bold"),
            padx=4,
            pady=4,
            highlightbackground="#cbd5e1",
            highlightthickness=1,
        )
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.logs = tk.Text(
            log_frame,
            height=6,
            bg="#ffffff",
            fg="#1e293b",
            font=("Consolas", 9),
            bd=0,
            padx=6,
            pady=6,
            selectbackground="#bfdbfe",
        )
        self.logs.pack(fill=tk.BOTH, expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.report_callback_exception = self._report_tk_callback_exception

        # Initial render of grid and start preview
        self.redraw_preview()
        self.log("ยินดีต้อนรับสู่ระบบ RoboMaster Autonomous SLAM (Assignment 2)")
        self.log("ระบบพร้อมทำงาน: รองรับแผนที่เขาวงกต 6×6 และการสแกนยิงเป้าหมาย (หน้า, ขวา, ซ้าย)")

    def _load_csv_map(self):
        """
        โหลดไฟล์ CSV เพื่อสร้างแผนที่ใน GUI
        รองรับ 2 รูปแบบ:
        
        รูปแบบ 1 - Wall List (แต่ละแถวเป็นกำแพง):
            type,x,y
            h,1,2        (กำแพงแนวนอนที่ตำแหน่ง x=1, y=2)
            v,3,4        (กำแพงแนวตั้งที่ตำแหน่ง x=3, y=4)
            target,2,3,NORTH,red,circle  (เป้าหมายที่ช่อง (2,3) ด้าน NORTH)
        
        รูปแบบ 2 - Grid Matrix (ตาราง NxM):
            แถวแรก = header (ไม่สนใจ)
            แต่ละช่อง: 0=ว่าง, 1=กำแพงขวา(v), 2=กำแพงบน(h), 3=ทั้งสอง
        """
        if self.is_running:
            self.log("⚠️ ไม่สามารถโหลด CSV ขณะภารกิจกำลังทำงาน")
            return

        from tkinter import filedialog
        filepath = filedialog.askopenfilename(
            title="เลือกไฟล์ CSV แผนที่ (Select Map CSV File)",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
            initialdir=os.path.dirname(os.path.abspath(__file__)),
        )
        if not filepath:
            return

        try:
            h_walls = set()
            v_walls = set()
            targets = []
            visited = set()
            discovered = set()
            grid_w = None
            grid_h = None

            with open(filepath, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                rows = list(reader)

            if not rows:
                self.log("⚠️ ไฟล์ CSV ว่างเปล่า!")
                return

            # ตรวจสอบรูปแบบ: ถ้าคอลัมน์แรกเป็น type indicator (h, v, target, wall)
            first_cell = rows[0][0].strip().lower() if rows[0] else ""
            is_wall_list = first_cell in ("type", "h", "v", "target", "wall", "hwall", "vwall")

            if is_wall_list:
                # รูปแบบ 1: Wall List Format
                for row in rows:
                    if not row or not row[0].strip():
                        continue
                    row_type = row[0].strip().lower()
                    if row_type in ("type", "header", "#"):
                        continue  # Skip header

                    if row_type in ("h", "hwall", "h_wall"):
                        x, y = int(row[1].strip()), int(row[2].strip())
                        h_walls.add((x, y))
                    elif row_type in ("v", "vwall", "v_wall"):
                        x, y = int(row[1].strip()), int(row[2].strip())
                        v_walls.add((x, y))
                    elif row_type == "target":
                        x, y = int(row[1].strip()), int(row[2].strip())
                        direction = row[3].strip().upper() if len(row) > 3 else "NORTH"
                        color = row[4].strip().lower() if len(row) > 4 else "red"
                        shape = row[5].strip().lower() if len(row) > 5 else "circle"
                        targets.append({"cell": (x, y), "direction": direction,
                                        "signs": [{"color": color, "shape": shape, "area": 500}]})
                    elif row_type in ("visited", "v_cell"):
                        x, y = int(row[1].strip()), int(row[2].strip())
                        visited.add((x, y))
                    elif row_type in ("grid", "size", "dimension"):
                        grid_w = int(row[1].strip())
                        grid_h = int(row[2].strip())

                # คำนวณขนาดตารางอัตโนมัติจากข้อมูลกำแพง
                if grid_w is None:
                    all_x = [w[0] for w in h_walls | v_walls] + [t["cell"][0] for t in targets]
                    all_y = [w[1] for w in h_walls | v_walls] + [t["cell"][1] for t in targets]
                    if all_x and all_y:
                        grid_w = max(all_x) + 1
                        grid_h = max(all_y) + 1
                    else:
                        grid_w = config.GRID_W
                        grid_h = config.GRID_H

            else:
                # รูปแบบ 2: Grid Matrix Format
                # แต่ละช่องเป็นตัวเลข: 0=ว่าง, 1=กำแพงขวา(v), 2=กำแพงบน(h), 3=ทั้งสอง
                data_rows = []
                for row in rows:
                    try:
                        numeric = [int(cell.strip()) for cell in row if cell.strip()]
                        if numeric:
                            data_rows.append(numeric)
                    except ValueError:
                        continue  # Skip non-numeric rows (headers)

                if not data_rows:
                    self.log("⚠️ ไม่พบข้อมูลตัวเลขในไฟล์ CSV!")
                    return

                grid_h = len(data_rows)
                grid_w = max(len(r) for r in data_rows)

                for row_idx, row in enumerate(data_rows):
                    y = grid_h - row_idx  # CSV row 0 = top = highest y
                    for col_idx, val in enumerate(row):
                        x = col_idx + 1
                        discovered.add((x, y))
                        if val & 1:  # bit 0 = กำแพงขวา (vertical wall)
                            v_walls.add((x, y))
                        if val & 2:  # bit 1 = กำแพงบน (horizontal wall)
                            h_walls.add((x, y))

            # อัปเดตขนาดตาราง
            if grid_w and grid_h:
                self.grid_w_var.set(grid_w)
                self.grid_h_var.set(grid_h)
                config.set_grid_dimensions(grid_w, grid_h)

            # เคลียร์ข้อมูลเก่าแล้วโหลดข้อมูลใหม่
            state.detected_h_walls.clear()
            state.detected_v_walls.clear()
            state.detected_signs.clear()
            state.discovered_cells.clear()
            state.visited_cells.clear()

            state.detected_h_walls.update(h_walls)
            state.detected_v_walls.update(v_walls)
            if discovered:
                state.discovered_cells.update(discovered)
            if visited:
                state.visited_cells.update(visited)

            for t in targets:
                key = (tuple(t["cell"]), t["direction"])
                state.detected_signs[key] = t["signs"]

            # อัปเดต canvas size
            max_side = 530
            self.cell_size = max(40, min(100, int((max_side - 2 * self.pad) / max(grid_w, grid_h))))
            new_cw = self.pad * 2 + self.cell_size * grid_w
            new_ch = self.pad * 2 + self.cell_size * grid_h
            self.canvas.config(width=new_cw, height=new_ch)

            self.redraw_preview()

            # อัปเดตข้อความสถานะ
            fname = os.path.basename(filepath)
            wall_count = len(h_walls) + len(v_walls)
            target_count = len(targets)
            status_msg = f"✅ โหลดแล้ว: {fname} ({grid_w}×{grid_h}, กำแพง {wall_count}, เป้า {target_count})"
            self.csv_status_label.config(text=status_msg, fg="#059669")
            self.log(f"📂 [CSV LOAD] โหลดไฟล์ CSV สำเร็จ: {fname}")
            self.log(f"   ▸ ขนาดตาราง: {grid_w} × {grid_h}")
            self.log(f"   ▸ กำแพงแนวนอน (h_walls): {len(h_walls)} แนว")
            self.log(f"   ▸ กำแพงแนวตั้ง (v_walls): {len(v_walls)} แนว")
            self.log(f"   ▸ เป้าหมาย: {target_count} จุด")
            if targets:
                for t in targets:
                    for s in t["signs"]:
                        self.log(f"   🎯 เป้า: {s['color'].upper()} {s['shape'].upper()} ที่ช่อง {t['cell']} ด้าน {t['direction']}")

            # บันทึกเป็น mission_map.json เพื่อให้ Run 2 ใช้ได้
            try:
                from slam import _save_mission_map
                start_cfg = self.get_start_config()
                _save_mission_map(start_cfg)
                self.log(f"💾 บันทึก mission_map.json จาก CSV เรียบร้อยแล้ว (พร้อมสำหรับ Run 2)")
            except Exception as save_err:
                self.log(f"⚠️ ไม่สามารถบันทึก mission_map.json: {save_err}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log(f"❌ ไม่สามารถโหลดไฟล์ CSV: {e}")
            self.csv_status_label.config(text=f"❌ โหลดไม่สำเร็จ: {e}", fg="#e11d48")

    def _on_zoom_adjust(self, value):
        """Apply the GUI zoom to both the live view and aim calibration."""
        vision.CAMERA_ZOOM = min(2.0, max(1.0, float(value)))

    def _change_zoom(self, delta):
        self.zoom_var.set(min(2.0, max(1.0, self.zoom_var.get() + delta)))
        self._on_zoom_adjust(self.zoom_var.get())

    def _on_grid_dimensions_changed(self):
        """Handle real-time changes to maze width, height, and cell size on dashboard."""
        if self.is_running:
            return
        try:
            w = self.grid_w_var.get()
            h = self.grid_h_var.get()
            pitch = self.grid_pitch_var.get()
            if w < 2 or h < 2 or pitch <= 0.05:
                return
            config.set_grid_dimensions(w, h, pitch)

            # Update maximum limits for Start X & Start Y spinboxes
            self.spin_x.config(to=w)
            self.spin_y.config(to=h)
            if self.start_x_var.get() > w:
                self.start_x_var.set(w)
            if self.start_y_var.get() > h:
                self.start_y_var.set(h)

            if hasattr(self, "size_info_label"):
                self.size_info_label.config(
                    text=f"ขนาดพื้นที่: {w} × {h} ช่อง ({w * pitch:.2f} × {h * pitch:.2f} m) · คลิกช่องในแผนที่เพื่อเลือกจุดเริ่มต้น"
                )

            # Dynamic cell sizing so canvas fits comfortably without being cut off
            max_side = 460
            self.cell_size = max(40, min(100, int((max_side - 2 * self.pad) / max(w, h))))
            new_w = self.pad * 2 + self.cell_size * w
            new_h = self.pad * 2 + self.cell_size * h
            self.canvas.config(width=new_w, height=new_h)

            self.redraw_preview()
        except Exception:
            pass

    def on_canvas_click(self, event):
        if self.is_running:
            return
        import config
        cur_gh = config.GRID_H
        gx = int((event.x - self.pad) // self.cell_size) + 1
        gy = cur_gh - int((event.y - self.pad) // self.cell_size)
        if config.in_bounds((gx, gy)):
            self.start_x_var.set(gx)
            self.start_y_var.set(gy)
            self.log(f"เลือกจุดเริ่มต้น: ({gx}, {gy})")

    def _format_time_str(self, seconds):
        if seconds is None:
            return "--:--.-"
        if seconds < 0:
            seconds = 0
        m = int(seconds // 60)
        s = int(seconds % 60)
        tenth = int((seconds * 10) % 10)
        return f"{m:02d}:{s:02d}.{tenth:01d}"

    def get_current_timer_str(self):
        if not self._timer_running or self._timer_start_time is None:
            active_t = self._round_times.get(self._active_round_key) if self._active_round_key else None
            return self._format_time_str(active_t)
        elapsed = time.monotonic() - self._timer_start_time
        return self._format_time_str(elapsed)

    def start_timer(self, round_key=None, round_label=None):
        """
        เริ่มจับเวลาทันทีเมื่อผู้ใช้กดปุ่มรัน หรือเมื่อเปลี่ยนรอบ
        """
        def _do_start():
            if self.closed:
                return
            now = time.monotonic()
            if self._mission_start_time is None or round_key == "round1":
                self._mission_start_time = now

            if round_key is not None:
                self._active_round_key = round_key
            elif self.run_mode in ("astar_only", "astar", "targets"):
                self._active_round_key = "round2"
            else:
                self._active_round_key = "round1"

            if round_label is not None:
                self._active_round_label = round_label
            elif self._active_round_key == "round2":
                self._active_round_label = "รอบ 2 (A*)"
            else:
                self._active_round_label = "รอบ 1 (SLAM)"

            self._timer_start_time = now
            self._timer_running = True
            self._warned_10min = False

            self.lbl_timer_main.config(fg="#38bdf8", bg="#0f172a", text="00:00.0")
            self.lbl_timer_status.config(
                text=f"🟢 กำลังจับเวลา {self._active_round_label}...",
                fg="#059669"
            )
            self._update_timer_labels()
            self._schedule_timer_tick()
            self.log(f"⏱️ [TIMER] เริ่มจับเวลาทันที: {self._active_round_label} (00:00.0)")

        if threading.current_thread() is threading.main_thread():
            _do_start()
        else:
            self._post_ui(_do_start)

    def _schedule_timer_tick(self):
        if self._timer_tick_id:
            try:
                self.root.after_cancel(self._timer_tick_id)
            except Exception:
                pass
            self._timer_tick_id = None
        if self._timer_running and not self.closed:
            self._timer_tick_id = self.root.after(250, self._on_timer_tick)

    def _on_timer_tick(self):
        self._timer_tick_id = None
        if not self._timer_running or self.closed:
            return
        now = time.monotonic()
        if self._timer_start_time is not None:
            round_elapsed = now - self._timer_start_time
            time_str = self._format_time_str(round_elapsed)

            # Warning if exceeding 10 minutes (600s), but KEEP TICKING and running
            if round_elapsed >= 600:
                self.lbl_timer_main.config(text=time_str, fg="#f59e0b")
                if not getattr(self, "_warned_10min", False):
                    self._warned_10min = True
                    self.lbl_timer_status.config(
                        text=f"⚠️ เวลาเกิน 10 นาที ({time_str}) - กำลังทำงานต่อเนื่องจนครบแมพ...",
                        fg="#d97706"
                    )
            else:
                self.lbl_timer_main.config(text=time_str, fg="#38bdf8")

            if self._active_round_key == "round1":
                self.lbl_round1_time.config(
                    text=f"🗺️ รอบ 1 (SLAM): {time_str} ⏱️",
                    bg="#dbeafe", fg="#1e40af"
                )
            elif self._active_round_key == "round2":
                self.lbl_round2_time.config(
                    text=f"⭐ รอบ 2 (A*): {time_str} ⏱️",
                    bg="#ede9fe", fg="#5b21b6"
                )

        if self._mission_start_time is not None:
            total_elapsed = now - self._mission_start_time
            self.lbl_total_time.config(
                text=f"⌛ เวลารวม: {self._format_time_str(total_elapsed)}",
                bg="#ecfdf5", fg="#047857"
            )

        self._schedule_timer_tick()

    def on_round_completed(self, round_key=None):
        """
        บันทึกเวลาเมื่อจบรอบหนึ่ง (เช่น จบรอบ 1 ในโหมดรันต่อเนื่อง auto_all)
        """
        def _do_complete():
            if self.closed:
                return
            now = time.monotonic()
            key = round_key or self._active_round_key or "round1"
            if self._timer_start_time is not None:
                duration = now - self._timer_start_time
                self._round_times[key] = duration
                t_str = self._format_time_str(duration)
                if key == "round1":
                    self.lbl_round1_time.config(
                        text=f"🗺️ รอบ 1 (SLAM): {t_str} ✅",
                        bg="#eff6ff", fg="#1d4ed8"
                    )
                    self.log(f"⏱️ [TIMER] จบรอบ 1 (SLAM)! บันทึกเวลา: {t_str}")
                elif key == "round2":
                    self.lbl_round2_time.config(
                        text=f"⭐ รอบ 2 (A*): {t_str} ✅",
                        bg="#f5f3ff", fg="#6d28d9"
                    )
                    self.log(f"⏱️ [TIMER] จบรอบ 2 (A*)! บันทึกเวลา: {t_str}")
            self._update_timer_labels()

        if threading.current_thread() is threading.main_thread():
            _do_complete()
        else:
            self._post_ui(_do_complete)

    def stop_timer(self, stopped_by_user=False):
        """
        หยุดจับเวลาและบันทึกเวลาสรุป
        """
        def _do_stop():
            if self.closed:
                return
            was_running = self._timer_running
            self._timer_running = False
            if self._timer_tick_id:
                try:
                    self.root.after_cancel(self._timer_tick_id)
                except Exception:
                    pass
                self._timer_tick_id = None

            now = time.monotonic()
            if was_running and self._timer_start_time is not None:
                round_duration = now - self._timer_start_time
                key = self._active_round_key or "round1"
                self._round_times[key] = round_duration
                if self._mission_start_time is not None:
                    self._round_times["total"] = now - self._mission_start_time

            self.lbl_timer_main.config(fg="#94a3b8", bg="#0f172a")
            if stopped_by_user:
                self.lbl_timer_status.config(
                    text="⏸️ ผู้ใช้สั่งหยุดเวลาฉุกเฉิน",
                    fg="#e11d48"
                )
                self.log(f"⏱️ [TIMER] หยุดจับเวลาฉุกเฉิน ที่เวลา {self.lbl_timer_main.cget('text')}")
            else:
                self.lbl_timer_status.config(
                    text=f"⏹️ บันทึกเวลาเสร็จสิ้น ({self._active_round_label or 'ภารกิจ'})",
                    fg="#0284c7"
                )
            self._update_timer_labels()

        if threading.current_thread() is threading.main_thread():
            _do_stop()
        else:
            self._post_ui(_do_stop)

    def reset_timer(self):
        """รีเซ็ตเวลาทั้งหมดเป็นค่าเริ่มต้น"""
        if self.is_running:
            self.log("⚠️ ไม่สามารถรีเซ็ตเวลาขณะภารกิจกำลังทำงาน")
            return
        self._timer_running = False
        self._timer_start_time = None
        self._mission_start_time = None
        self._active_round_key = None
        self._active_round_label = ""
        self._round_times = {"round1": None, "round2": None, "total": None}
        if self._timer_tick_id:
            try:
                self.root.after_cancel(self._timer_tick_id)
            except Exception:
                pass
            self._timer_tick_id = None
        self.lbl_timer_main.config(text="00:00.0", fg="#38bdf8", bg="#0f172a")
        self.lbl_timer_status.config(text="⚪ พร้อมจับเวลา (กดรันเพื่อเริ่มทันที)", fg="#64748b")
        self._update_timer_labels()
        self.log("⏱️ [TIMER] รีเซ็ตตัวจับเวลาเรียบร้อยแล้ว")

    def _update_timer_labels(self):
        r1 = self._round_times.get("round1")
        r2 = self._round_times.get("round2")
        total = self._round_times.get("total")

        if r1 is not None:
            self.lbl_round1_time.config(
                text=f"🗺️ รอบ 1 (SLAM): {self._format_time_str(r1)} ✅",
                bg="#eff6ff", fg="#1d4ed8"
            )
        elif self._active_round_key != "round1" or not self._timer_running:
            self.lbl_round1_time.config(
                text="🗺️ รอบ 1 (SLAM): --:--.-",
                bg="#f8fafc", fg="#64748b"
            )

        if r2 is not None:
            self.lbl_round2_time.config(
                text=f"⭐ รอบ 2 (A*): {self._format_time_str(r2)} ✅",
                bg="#f5f3ff", fg="#6d28d9"
            )
        elif self._active_round_key != "round2" or not self._timer_running:
            self.lbl_round2_time.config(
                text="⭐ รอบ 2 (A*): --:--.-",
                bg="#f8fafc", fg="#64748b"
            )

        if total is not None:
            self.lbl_total_time.config(
                text=f"⌛ เวลารวม: {self._format_time_str(total)}",
                bg="#ecfdf5", fg="#047857"
            )
        elif not self._timer_running:
            self.lbl_total_time.config(
                text="⌛ เวลารวม: --:--.-",
                bg="#f8fafc", fg="#64748b"
            )

    def handle_start_click(self, run_mode="mapping"):
        if self.is_running:
            return
        state.stop_requested = False
        import config

        if run_mode == "resume":
            try:
                from slam import _load_mission_map
                data = _load_mission_map()
            except Exception as e:
                self.log(f"❌ ไม่สามารถกู้คืนได้: {e}")
                self.log("💡 หากต้องการเริ่มใหม่ ให้กด 'รอบ 1: เดินแบบ SLAM' หรือ 'รันต่อเนื่อง'")
                return

            gw = int(data.get("grid_width", self.grid_w_var.get()))
            gh = int(data.get("grid_height", self.grid_h_var.get()))
            pitch = float(data.get("grid_size_m", self.grid_pitch_var.get()))

            last_pos = data.get("last_position")
            if last_pos and len(last_pos) == 2:
                sx, sy = int(last_pos[0]), int(last_pos[1])
            else:
                start_p = data.get("start_position", [self.start_x_var.get(), self.start_y_var.get()])
                sx, sy = int(start_p[0]), int(start_p[1])

            heading = data.get("last_heading") or data.get("start_heading") or self.start_heading_var.get()

            self.grid_w_var.set(gw)
            self.grid_h_var.set(gh)
            self.grid_pitch_var.set(pitch)
            self.start_x_var.set(sx)
            self.start_y_var.set(sy)
            self.start_heading_var.set(heading)
            config.set_grid_dimensions(gw, gh, pitch)

            state.detected_h_walls.clear()
            state.detected_v_walls.clear()
            state.detected_signs.clear()
            state.discovered_cells.clear()
            state.visited_cells.clear()
            state.fired_targets.clear()

            state.detected_h_walls.update([tuple(w) for w in data.get("h_walls", [])])
            state.detected_v_walls.update([tuple(w) for w in data.get("v_walls", [])])
            state.visited_cells.update([tuple(c) for c in data.get("visited_cells", [])])
            state.discovered_cells.update([tuple(c) for c in data.get("discovered_cells", [])])
            for t in data.get("targets", []):
                key = (tuple(t["cell"]), t["direction"])
                state.detected_signs[key] = t["signs"]
            for f in data.get("fired_targets", []):
                if isinstance(f, (list, tuple)) and len(f) >= 3:
                    c = tuple(f[0]) if isinstance(f[0], (list, tuple)) else f[0]
                    state.fired_targets.add((c, f[1], f[2]))
                else:
                    state.fired_targets.add(tuple(f) if isinstance(f, list) else f)

            max_side = 530
            self.cell_size = max(40, min(100, int((max_side - 2 * self.pad) / max(gw, gh))))
            new_cw = self.pad * 2 + self.cell_size * gw
            new_ch = self.pad * 2 + self.cell_size * gh
            self.canvas.config(width=new_cw, height=new_ch)
            self.redraw_preview()

            step = data.get("step", 0)
            self.log("=" * 60)
            self.log("🔄 [RECOVERY] โหลด Checkpoint สำเร็จ!")
            self.log(f"   ▸ พิกัดล่าสุด: ({sx}, {sy}) ทิศ {heading} (Step: {step})")
            self.log(f"   ▸ สำรวจแล้ว: {len(state.visited_cells)} ช่อง, ค้นพบกำแพง: {len(state.detected_h_walls)+len(state.detected_v_walls)} จุด")
            self.log(f"   ▸ เป้าหมายที่พบ: {len(data.get('targets', []))} จุด, ยิงแล้ว: {len(state.fired_targets)} จุด")
            self.log("=" * 60)
        elif run_mode in ("astar_only", "astar", "targets"):
            # รอบที่ 2 (A*): โหลดแผนที่และตรวจสอบเป้าหมายก่อนเริ่ม
            data = None
            try:
                from slam import _load_mission_map
                data = _load_mission_map()
            except Exception as e:
                data = None

            if data is None and not (state.detected_h_walls or state.detected_v_walls or state.detected_signs):
                self.log("=" * 60)
                self.log("⚠️ [รอบ 2 A*] ไม่สามารถเริ่มรอบ 2 ได้: ยังไม่มีข้อมูลแผนที่หรือเป้าหมายจากรอบ 1")
                self.log("💡 กรุณารัน [🗺️ รอบ 1: เดินแบบ SLAM] ให้เสร็จก่อน หรือกดปุ่ม [📂 โหลด CSV]")
                self.log("=" * 60)
                return

            if data:
                gw = int(data.get("grid_width", self.grid_w_var.get()))
                gh = int(data.get("grid_height", self.grid_h_var.get()))
                pitch = float(data.get("grid_size_m", self.grid_pitch_var.get()))

                sx = self.start_x_var.get()
                sy = self.start_y_var.get()
                heading = self.start_heading_var.get()

                self.grid_w_var.set(gw)
                self.grid_h_var.set(gh)
                self.grid_pitch_var.set(pitch)
                config.set_grid_dimensions(gw, gh, pitch)

                state.detected_h_walls.clear()
                state.detected_v_walls.clear()
                state.detected_signs.clear()
                state.discovered_cells.clear()
                state.visited_cells.clear()
                state.fired_targets.clear()

                state.detected_h_walls.update([tuple(w) for w in data.get("h_walls", [])])
                state.detected_v_walls.update([tuple(w) for w in data.get("v_walls", [])])
                state.visited_cells.update([tuple(c) for c in data.get("visited_cells", [])])
                state.discovered_cells.update([tuple(c) for c in data.get("discovered_cells", [])])
                for t in data.get("targets", []):
                    key = (tuple(t["cell"]), t["direction"])
                    state.detected_signs[key] = t["signs"]
                    for s in t.get("signs", []):
                        img_p = s.get("image_path")
                        if img_p and os.path.exists(img_p):
                            meta = {
                                "color": s.get("color", ""),
                                "shape": s.get("shape", ""),
                                "cell": tuple(t["cell"]),
                                "direction": t["direction"],
                                "fired": False,
                                "area": s.get("area", 0),
                                "time": time.strftime("%H:%M:%S")
                            }
                            self.add_target_result(img_p, meta)

                max_side = 530
                self.cell_size = max(40, min(100, int((max_side - 2 * self.pad) / max(gw, gh))))
                new_cw = self.pad * 2 + self.cell_size * gw
                new_ch = self.pad * 2 + self.cell_size * gh
                self.canvas.config(width=new_cw, height=new_ch)
                self.redraw_preview()

                target_count = sum(len(s) for s in state.detected_signs.values())
                self.log("=" * 60)
                self.log("⭐ [รอบ 2 A*] โหลดแผนที่พร้อมเป้าหมายสำเร็จ!")
                self.log(f"   ▸ จุดเริ่มต้น: ({sx}, {sy}) ทิศ {heading}")
                self.log(f"   ▸ แผนที่: {gw}×{gh} ช่อง, กำแพง {len(state.detected_h_walls)+len(state.detected_v_walls)} แนว")
                self.log(f"   ▸ เป้าหมายทั้งหมด: {target_count} จุด")
                self.log("=" * 60)
                if target_count == 0:
                    self.log("⚠️ คำเตือน: แผนที่นี้ไม่พบเป้าหมาย กรุณาตรวจสอบหรือรันรอบ 1 ใหม่")
                    return
            else:
                sx = self.start_x_var.get()
                sy = self.start_y_var.get()
                gw = self.grid_w_var.get()
                gh = self.grid_h_var.get()
                pitch = self.grid_pitch_var.get()
        else:
            gw = self.grid_w_var.get()
            gh = self.grid_h_var.get()
            pitch = self.grid_pitch_var.get()
            config.set_grid_dimensions(gw, gh, pitch)

            sx = self.start_x_var.get()
            sy = self.start_y_var.get()
            if not config.in_bounds((sx, sy)):
                self.log(f"พิกัด ({sx}, {sy}) ไม่อยู่ในตาราง {gw}x{gh}!")
                return

        self.is_running = True
        self.run_mode = run_mode
        self.btn_slam.config(state=tk.DISABLED, bg="#cbd5e1")
        self.btn_astar.config(state=tk.DISABLED, bg="#cbd5e1")
        self.btn_auto_all.config(state=tk.DISABLED, bg="#cbd5e1")
        self.btn_resume.config(state=tk.DISABLED, bg="#cbd5e1")
        self.btn_load_csv.config(state=tk.DISABLED, bg="#cbd5e1")
        self.btn_stop.config(state=tk.NORMAL, bg="#e11d48")
        self.spin_w.config(state=tk.DISABLED)
        self.spin_h.config(state=tk.DISABLED)
        self.spin_pitch.config(state=tk.DISABLED)
        self.spin_x.config(state=tk.DISABLED)
        self.spin_y.config(state=tk.DISABLED)
        self.heading_menu.config(state=tk.DISABLED)
        for control in self.vision_controls:
            control.config(state=tk.DISABLED)
        self.canvas.config(cursor="arrow")

        # กำหนดรอบและเริ่มจับเวลาทันทีที่กดปุ่มรัน! (Immediate Stopwatch Start)
        if run_mode in ("astar_only", "astar", "targets"):
            r_key = "round2"
            r_label = "รอบ 2 (A*)"
        elif run_mode == "auto_all":
            r_key = "round1"
            r_label = "รอบ 1 (SLAM)"
            self._round_times = {"round1": None, "round2": None, "total": None}
            self._mission_start_time = time.monotonic()
        elif run_mode == "resume":
            r_key = "round1"
            r_label = "กู้คืน / รันต่อ"
            if self._mission_start_time is None:
                self._mission_start_time = time.monotonic()
        else:
            r_key = "round1"
            r_label = "รอบ 1 (SLAM)"
            self._round_times["round1"] = None
        self.start_timer(round_key=r_key, round_label=r_label)

        start_config = (sx, sy, self.start_heading_var.get())
        mission_config = {
            "run_mode": run_mode,
            "is_resume": (run_mode == "resume"),
            "grid_w": gw,
            "grid_h": gh,
            "cell_size_m": pitch,
            "colors": {name for name, var in self.color_vars.items() if var.get()},
            "shapes": {name for name, var in self.shape_vars.items() if var.get()},
        }
        self.log(f"⚙️ เริ่มภารกิจด้วยขนาดตาราง: {gw} × {gh} ช่อง (ขนาดช่องละ {pitch:.2f} m)")
        if not mission_config["colors"] or not mission_config["shapes"]:
            self.log("No shots are enabled: select at least one target color and one target shape.")
        else:
            self.log(
                "Fire selection: " + ", ".join(sorted(mission_config["colors"])) +
                " / " + ", ".join(sorted(mission_config["shapes"])) +
                f"; camera zoom {vision.CAMERA_ZOOM:.1f}x."
            )
        if self.on_start:
            threading.Thread(target=self.on_start, args=(start_config, mission_config), daemon=True).start()

    def handle_stop_click(self):
        state.stop_requested = True
        self.stop_timer(stopped_by_user=True)
        self.log("⚠️ ผู้ใช้กดปุ่มหยุดภารกิจฉุกเฉิน")
        if self.on_stop:
            self.on_stop()

    def _post_ui(self, callback, key=None):
        """Queue UI work from robot/vision threads for the Tk thread to drain."""
        if self.closed:
            return
        if key is None:
            item = callback
        else:
            with self._ui_coalesce_lock:
                self._ui_coalesced[key] = callback
                if key in self._ui_coalesced_scheduled:
                    return
                self._ui_coalesced_scheduled.add(key)
            item = ("coalesced", key)
        try:
            self._ui_queue.put_nowait(item)
        except queue.Full:
            try:
                dropped = self._ui_queue.get_nowait()
                if isinstance(dropped, tuple) and dropped[:1] == ("coalesced",):
                    with self._ui_coalesce_lock:
                        self._ui_coalesced_scheduled.discard(dropped[1])
                self._ui_queue.put_nowait(item)
            except (queue.Empty, queue.Full):
                if key is not None:
                    with self._ui_coalesce_lock:
                        self._ui_coalesced_scheduled.discard(key)

    def get_start_config(self):
        """Return the current start coordinates and heading from GUI controls."""
        return (self.start_x_var.get(), self.start_y_var.get(), self.start_heading_var.get())

    def reset_session_ui(self):
        """Reset dashboard UI elements (target gallery, previews, logs) for a fresh mapping session."""
        def _reset():
            if self.closed:
                return
            self.target_results.clear()
            self._latest_target_photo = None
            self.btn_gallery.config(text="🖼️ ภาพเป้าหมาย (0)")
            self.target_thumb_label.config(
                image="", text="ยังไม่พบเป้าหมาย\n(จะแสดงภาพเมื่อตรวจพบ/ยิง)", width=26, height=4
            )
            self.target_info_label.config(
                text="เป้าหมาย: (รอตรวจจับ)\nพิกัด: -  |  ทิศทาง: -\nสถานะ: ยังไม่มีการยิง\nไฟล์ภาพ: -"
            )
            self.logs.delete("1.0", tk.END)
            self.reset_timer()
            self.redraw_preview()
        self._post_ui(_reset)

    def close(self):
        """Handle window close event (clicking [X]). Guaranteed to save mission map & logs."""
        if self.is_running:
            import tkinter.messagebox as mb
            try:
                if not mb.askyesno("ยืนยันการปิดหน้าต่าง", "หุ่นยนต์กำลังทำงานอยู่ ต้องการหยุดภารกิจและปิดโปรแกรมใช่หรือไม่?"):
                    return
            except Exception:
                pass
        state.stop_requested = True
        if self._timer_tick_id:
            try:
                self.root.after_cancel(self._timer_tick_id)
            except Exception:
                pass
            self._timer_tick_id = None

        if state.visited_cells:
            try:
                from slam import _save_mission_map
                from evaluation import save_outputs
                start_cfg = self.get_start_config()
                _save_mission_map(start_cfg)
                save_outputs(round_name="window_closed")
                print(f"💾 [WINDOW CLOSE] บันทึกแผนที่ ({len(state.visited_cells)} ช่อง) และ Log การเดินลงไฟล์เรียบร้อยแล้ว!")
            except Exception as e:
                print(f"[SAVE ON CLOSE ERROR] {e}")

        self.closed = True
        try:
            self.root.destroy()
        except Exception:
            pass

    def run_finished(self):
        """Restore run controls on the Tk thread after the mission worker exits."""
        def finish():
            if self.closed:
                return
            self.stop_timer(stopped_by_user=False)
            self.is_running = False
            self.btn_slam.config(state=tk.NORMAL, bg="#2563eb")
            self.btn_astar.config(state=tk.NORMAL, bg="#7c3aed")
            self.btn_auto_all.config(state=tk.NORMAL, bg="#059669")
            self.btn_resume.config(state=tk.NORMAL, bg="#0284c7")
            self.btn_load_csv.config(state=tk.NORMAL, bg="#ea580c")
            self.btn_stop.config(state=tk.DISABLED, bg="#cbd5e1")
            self.spin_w.config(state=tk.NORMAL)
            self.spin_h.config(state=tk.NORMAL)
            self.spin_pitch.config(state=tk.NORMAL)
            self.spin_x.config(state=tk.NORMAL)
            self.spin_y.config(state=tk.NORMAL)
            self.heading_menu.config(state=tk.NORMAL)
            for control in self.vision_controls:
                control.config(state=tk.NORMAL)
            self.canvas.config(cursor="hand2")
        self._post_ui(finish)

    def set_vision_source(self, reader, analyzer):
        self.vision_reader = reader
        self.vision_analyzer = analyzer

    def _show_camera_frame(self, frame):
        if self.closed or frame is None:
            return
        if getattr(self, "_camera_busy", False):
            return
        self._camera_busy = True
        try:
            if self._aim_reticle_active:
                frame = frame.copy()
                height, width = frame.shape[:2]
                center = (width // 2, height // 2)
                cv2.drawMarker(frame, center, (0, 255, 255), cv2.MARKER_CROSS, 28, 2)
                cv2.circle(frame, center, 16, (0, 255, 255), 1, cv2.LINE_AA)

            # Fast fixed-size resize (400 x 225 pixels: 16:9, light on RAM and GDI memory)
            resized_bgr = cv2.resize(frame, (400, 225), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)

            photo = ImageTk.PhotoImage(image)
            old_photo = self._camera_photo
            self._camera_photo = photo
            self.camera_label.configure(image=photo, text="")
            if old_photo is not None:
                try:
                    self.root.call("image", "delete", str(old_photo))
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            self._camera_busy = False

    def set_aim_reticle(self, active):
        """Show a camera-center reticle while the gimbal is aiming and firing."""
        if self.closed:
            return
        def update_reticle():
            self._aim_reticle_active = bool(active)
        self._post_ui(update_reticle, key="aim_reticle")

    def log(self, message):
        now_str = time.strftime("%H:%M:%S")
        formatted = f"[{now_str}] {message}"
        try:
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, "mission_log.txt"), "a", encoding="utf-8") as f:
                f.write(formatted + "\n")
        except Exception:
            pass

        if not self.closed:
            def _append():
                if not self.closed:
                    try:
                        self.logs.insert(tk.END, formatted + "\n")
                        line_count = int(self.logs.index("end-1c").split(".")[0])
                        if line_count > 150:
                            self.logs.delete("1.0", f"{line_count - 100}.0")
                        self.logs.see(tk.END)
                    except Exception:
                        pass
            self._post_ui(_append)

    def clear_data_and_logs(self):
        """Clear memory cache, prune logs, and run GC to keep GUI extremely stable during long runs."""
        try:
            self.logs.delete("1.0", tk.END)
            with self._ui_coalesce_lock:
                self._ui_coalesced.clear()
                self._ui_coalesced_scheduled.clear()
            self._last_camera_frame = None
            import gc
            gc.collect()
            self.log("🧹 [CLEANUP] เคลียร์ข้อมูลและ Log ในหน่วยความจำเรียบร้อยแล้ว (GUI พร้อมทำงานต่อเนื่อง)")
        except Exception as e:
            print(f"[CLEANUP ERROR] {e}")

    def _periodic_cleanup(self):
        """Automatically prunes logs and runs GC every 30 seconds to prevent memory bloat and keep GUI alive."""
        if self.closed:
            return
        try:
            try:
                line_count = int(self.logs.index("end-1c").split(".")[0])
                if line_count > 150:
                    self.logs.delete("1.0", f"{line_count - 100}.0")
            except Exception:
                pass

            self._last_camera_frame = None
            import gc
            gc.collect()
        except Exception:
            pass
        finally:
            if not self.closed:
                try:
                    self.root.after(30000, self._periodic_cleanup)
                except (tk.TclError, Exception):
                    pass

    def add_target_result(self, filepath, metadata):
        """Register a new target snapshot into the dashboard gallery and update the preview card."""
        if not filepath or not metadata:
            return
        def _apply():
            if self.closed:
                return
            self.target_results.append(metadata)
            # 1. Update Gallery Button Count
            self.btn_gallery.config(text=f"🖼️ ภาพเป้าหมาย ({len(self.target_results)})")
            # 2. Update Latest Target Card
            try:
                img = Image.open(filepath)
                img.thumbnail((160, 90), Image.Resampling.BILINEAR)
                self._latest_target_photo = ImageTk.PhotoImage(img)
                self.target_thumb_label.config(image=self._latest_target_photo, text="", width=160, height=90)
            except Exception:
                pass

            color = metadata.get("color", "").upper()
            shape = metadata.get("shape", "").upper()
            cell = metadata.get("cell", (0, 0))
            direction = metadata.get("direction", "")
            fired = metadata.get("fired", False)
            area = metadata.get("area", 0)
            time_str = metadata.get("time", "")
            is_hostage = bool(metadata.get("is_hostage") or shape == "HOSTAGE")
            if is_hostage:
                status_desc = "⚠️ ตัวประกัน (ลูกไก่) - ห้ามยิงเด็ดขาด!"
                info_text = (
                    f"🚨 วัตถุ: ตัวประกัน (ลูกไก่ / HOSTAGE)\n"
                    f"พิกัด: ({cell[0]},{cell[1]}) ด้าน {direction}\n"
                    f"สถานะ: {status_desc}\n"
                    f"เวลา: {time_str}  |  ไฟล์: {os.path.basename(filepath)}"
                )
                self.target_info_label.config(text=info_text, fg="#b45309")
                self.target_card.config(text=" ⚠️ ตรวจพบตัวประกัน (Hostage Detected - DO NOT SHOOT) ", fg="#b45309")
            else:
                status_desc = "💥 ยิง 2x เจล เข้าเป้าแล้ว" if fired else "🔍 ตรวจพบ (ยังไม่ยิง)"
                info_text = (
                    f"เป้าหมาย: {color} {shape} ({area} px)\n"
                    f"พิกัด: ({cell[0]},{cell[1]}) ด้าน {direction}\n"
                    f"สถานะ: {status_desc}\n"
                    f"เวลา: {time_str}  |  ไฟล์: {os.path.basename(filepath)}"
                )
                self.target_info_label.config(text=info_text, fg="#1e293b")
                self.target_card.config(text=" 🎯 ผลลัพธ์ภาพเป้าหมายล่าสุด (Latest Target Snapshot) ", fg="#0f172a")
        self._post_ui(_apply)

    def open_results_folder(self):
        """Open the results directory in Windows File Explorer."""
        import os
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        try:
            os.startfile(results_dir)
        except Exception as e:
            self.log(f"ไม่สามารถเปิดโฟลเดอร์ results ได้: {e}")

    def show_final_map_window(self, initial_tab=None):
        """Display the generated SLAM trajectory map in a high-res pop-up window with Round 1 & Round 2 switching."""
        import os
        code_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(code_dir, "results")

        map_files = {
            "round1": [
                os.path.join(results_dir, "robot_trajectory_round1.png"),
                os.path.join(code_dir, "robot_trajectory_round1.png"),
            ],
            "round2": [
                os.path.join(results_dir, "robot_trajectory_round2.png"),
                os.path.join(code_dir, "robot_trajectory_round2.png"),
            ],
            "overview": [
                os.path.join(results_dir, "robot_trajectory.png"),
                os.path.join(code_dir, "robot_trajectory.png"),
                os.path.join(results_dir, "final_slam_map.png"),
                os.path.join(code_dir, "final_slam_map.png"),
            ],
        }

        def _find_path(key):
            for p in map_files.get(key, []):
                if os.path.exists(p):
                    return p
            return None

        has_r1 = bool(_find_path("round1"))
        has_r2 = bool(_find_path("round2"))
        has_any = has_r1 or has_r2 or bool(_find_path("overview"))

        if not has_any:
            self.log("⚠️ ยังไม่มีภาพแผนที่ผลลัพธ์ (จะถูกสร้างเมื่อสำรวจเสร็จสิ้น)")
            return

        map_win = tk.Toplevel(self.root)
        map_win.title("RoboMaster SLAM · Trajectory Map (Round 1 & Round 2)")
        map_win.geometry("860x980")
        map_win.configure(bg="#f8fafc")

        top_bar = tk.Frame(map_win, bg="#ffffff", padx=12, pady=8, highlightbackground="#cbd5e1", highlightthickness=1)
        top_bar.pack(fill=tk.X)

        lbl_title = tk.Label(top_bar, text="🗺️ แผนที่เส้นทางการเดิน:", bg="#ffffff", fg="#0f172a", font=("Segoe UI", 10, "bold"))
        lbl_title.pack(side=tk.LEFT, padx=(0, 10))

        content_frame = tk.Frame(map_win, bg="#f8fafc")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        lbl_img = tk.Label(content_frame, bg="#f8fafc")
        lbl_img.pack(fill=tk.BOTH, expand=True)

        map_win._photo = None

        def _load_and_display(key):
            target_path = _find_path(key)
            if not target_path:
                lbl_img.config(image="", text=f"ยังไม่มีข้อมูลภาพแผนที่สำหรับ {key.upper()}\n(ไฟล์จะถูกบันทึกอัตโนมัติเมื่อสิ้นสุดรอบ {key})", fg="#64748b", font=("Segoe UI", 11))
                return
            try:
                img = Image.open(target_path)
                img.thumbnail((820, 880), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                map_win._photo = photo
                lbl_img.config(image=photo, text="")
                self.log(f"🗺️ แสดงแผนที่: {os.path.basename(target_path)}")
            except Exception as err:
                lbl_img.config(text=f"Error loading map: {err}", fg="red")

        btn_r1 = tk.Button(top_bar, text="🗺️ รอบ 1: สำรวจ SLAM", command=lambda: _load_and_display("round1"),
                           bg="#059669" if has_r1 else "#94a3b8", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, padx=8, pady=3, cursor="hand2")
        btn_r1.pack(side=tk.LEFT, padx=4)

        btn_r2 = tk.Button(top_bar, text="⭐ รอบ 2: ยิงเป้าหมาย A*", command=lambda: _load_and_display("round2"),
                           bg="#2563eb" if has_r2 else "#94a3b8", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, padx=8, pady=3, cursor="hand2")
        btn_r2.pack(side=tk.LEFT, padx=4)

        btn_all = tk.Button(top_bar, text="🌐 แผนที่รวมล่าสุด", command=lambda: _load_and_display("overview"),
                            bg="#475569", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, padx=8, pady=3, cursor="hand2")
        btn_all.pack(side=tk.LEFT, padx=4)

        tk.Button(top_bar, text="📁 Results Folder", command=self.open_results_folder,
                  bg="#64748b", fg="white", font=("Segoe UI", 8, "bold"), relief=tk.FLAT, padx=8, pady=3, cursor="hand2").pack(side=tk.RIGHT)

        # Select initial view
        if initial_tab and _find_path(initial_tab):
            _load_and_display(initial_tab)
        elif has_r1:
            _load_and_display("round1")
        elif has_r2:
            _load_and_display("round2")
        else:
            _load_and_display("overview")

    def show_target_gallery(self):
        """Open a dedicated pop-up gallery showing all detected and fired target snapshots."""
        import os, glob
        gallery_win = tk.Toplevel(self.root)
        gallery_win.title("RoboMaster SLAM · Target Snapshots Gallery")
        gallery_win.geometry("920x680")
        gallery_win.configure(bg="#f8fafc")

        header_frame = tk.Frame(gallery_win, bg="#ffffff", padx=16, pady=10, highlightbackground="#cbd5e1", highlightthickness=1)
        header_frame.pack(fill=tk.X)

        tk.Label(
            header_frame,
            text=f"🎯 แกลเลอรีภาพเป้าหมายที่ตรวจพบ & ยิง (พบ {len(self.target_results)} ภาพ)",
            bg="#ffffff", fg="#0f172a", font=("Segoe UI", 12, "bold")
        ).pack(side=tk.LEFT)

        tk.Button(
            header_frame, text="📁 เปิดโฟลเดอร์ภาพ", command=self.open_results_folder,
            bg="#2563eb", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT,
            padx=10, pady=4, cursor="hand2"
        ).pack(side=tk.RIGHT)

        container = tk.Frame(gallery_win, bg="#f8fafc")
        container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        canvas = tk.Canvas(container, bg="#f8fafc", highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#f8fafc")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        targets_to_show = list(self.target_results)
        if not targets_to_show:
            code_dir = os.path.dirname(os.path.abspath(__file__))
            disk_files = glob.glob(os.path.join(code_dir, "results", "targets", "*.jpg"))
            for df in disk_files:
                targets_to_show.append({"filepath": df, "filename": os.path.basename(df)})

        if not targets_to_show:
            tk.Label(
                scrollable_frame,
                text="ยังไม่มีภาพเป้าหมายที่บันทึกไว้ในรอบนี้\nเมื่อหุ่นยนต์ตรวจพบหรือยิงเป้าหมาย ภาพสแนปช็อตจะปรากฏที่นี่โดยอัตโนมัติ",
                bg="#f8fafc", fg="#64748b", font=("Segoe UI", 11),
                pady=40
            ).pack()
            return

        gallery_win._photos = []
        for idx, item in enumerate(reversed(targets_to_show)):
            fp = item.get("filepath", "")
            if not os.path.exists(fp):
                continue
            card = tk.Frame(scrollable_frame, bg="#ffffff", padx=10, pady=8,
                            highlightbackground="#cbd5e1", highlightthickness=1)
            card.pack(fill=tk.X, pady=6, padx=4)

            try:
                img = Image.open(fp)
                img.thumbnail((240, 135), Image.Resampling.BILINEAR)
                photo = ImageTk.PhotoImage(img)
                gallery_win._photos.append(photo)
                lbl_img = tk.Label(card, image=photo, bg="#ffffff", cursor="hand2")
                lbl_img.pack(side=tk.LEFT, padx=(0, 12))
                lbl_img.bind("<Button-1>", lambda e, p=fp: self._open_single_image(p))
            except Exception:
                pass

            color = item.get("color", "").upper()
            shape = item.get("shape", "").upper()
            cell = item.get("cell", "")
            dir_str = item.get("direction", "")
            fired = item.get("fired", True)
            t_str = item.get("time", "")

            is_hostage = bool(item.get("is_hostage") or shape == "HOSTAGE" or "hostage" in os.path.basename(fp).lower())
            if is_hostage:
                badge_text = "⚠️ ตัวประกัน (ลูกไก่) - ห้ามยิงเด็ดขาด (HOSTAGE SAFE)"
                badge_fg = "#d97706"
                target_title = f"ตัวประกัน #{len(targets_to_show) - idx}: ลูกไก่ (HOSTAGE)"
            elif fired:
                badge_text = "💥 ยิงเข้าเป้า (FIRED HIT)"
                badge_fg = "#059669"
                target_title = f"เป้าหมาย #{len(targets_to_show) - idx}: {color} {shape}"
            else:
                badge_text = "🔍 ตรวจพบ (DETECTED)"
                badge_fg = "#d97706"
                target_title = f"เป้าหมาย #{len(targets_to_show) - idx}: {color} {shape}"

            tk.Label(card, text=target_title,
                     bg="#ffffff", fg="#0f172a", font=("Segoe UI", 11, "bold")).pack(anchor="w")
            tk.Label(card, text=f"สถานะ: {badge_text}", bg="#ffffff", fg=badge_fg,
                     font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(2, 0))
            if cell:
                tk.Label(card, text=f"พิกัด: ช่อง {cell}  |  ทิศทาง: {dir_str}  |  เวลา: {t_str}",
                         bg="#ffffff", fg="#475569", font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 0))
            tk.Label(card, text=f"ไฟล์: {os.path.basename(fp)}", bg="#ffffff", fg="#94a3b8",
                     font=("Consolas", 8)).pack(anchor="w", pady=(2, 0))

    def _open_single_image(self, filepath):
        """Open a single snapshot image in full resolution."""
        import os
        if not os.path.exists(filepath):
            return
        top = tk.Toplevel(self.root)
        top.title(f"Target Snapshot · {os.path.basename(filepath)}")
        top.geometry("960x640")
        top.configure(bg="#0f172a")
        try:
            img = Image.open(filepath)
            img.thumbnail((940, 620), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            lbl = tk.Label(top, image=photo, bg="#0f172a")
            lbl.image = photo
            lbl.pack(fill=tk.BOTH, expand=True)
        except Exception:
            pass

    def redraw_preview(self):
        if self.is_running:
            return
        try:
            sx = self.start_x_var.get()
            sy = self.start_y_var.get()
            heading = self.start_heading_var.get()
            if in_bounds((sx, sy)):
                self.update((sx, sy), heading, 0, "รอเริ่มสำรวจ (กด 'เริ่มสำรวจ')", extra_readings=None)
        except Exception:
            pass

    def update(self, position, heading, step, status, extra_readings=None, gimbal_dir=None):
        if self.closed:
            return

        def _do_update():
            if self.closed:
                return

            with state.state_lock:
                snap_visited = set(state.visited_cells)
                snap_discovered = set(state.discovered_cells)
                snap_h_walls = list(state.detected_h_walls)
                snap_v_walls = list(state.detected_v_walls)
                snap_signs = [(k, list(v)) for k, v in state.detected_signs.items()]
                snap_trajectory = list(state.trajectory)
                snap_distance = float(state.current_distance)

            self.canvas.delete("all")

            cur_gw = config.GRID_W
            cur_gh = config.GRID_H

            # 1. Draw cell tiles
            for y in range(1, cur_gh + 1):
                for x in range(1, cur_gw + 1):
                    cx = self.pad + (x - 1) * self.cell_size
                    cy = self.pad + (cur_gh - y) * self.cell_size
                    cell_coord = (x, y)

                    if cell_coord in snap_visited:
                        fill_color = "#f0fdf4"   # Luxury Mint Cream
                        border_color = "#10b981" # Emerald border
                        label = "VISITED"
                        label_color = "#047857"
                        coord_color = "#065f46"
                    elif cell_coord in snap_discovered:
                        fill_color = "#f0f9ff"   # Azure Light
                        border_color = "#38bdf8"
                        label = "OPEN"
                        label_color = "#0284c7"
                        coord_color = "#0369a1"
                    else:
                        fill_color = "#ffffff"   # Clean Pearl White
                        border_color = "#cbd5e1"
                        label = "?"
                        label_color = "#94a3b8"
                        coord_color = "#64748b"

                    self.canvas.create_rectangle(
                        cx, cy, cx + self.cell_size, cy + self.cell_size,
                        fill=fill_color, outline=border_color, width=1.2,
                        dash=(3, 3) if cell_coord not in snap_visited else None
                    )

                    coord_font_size = max(7, min(10, int(self.cell_size * 0.11)))
                    label_font_size = max(8, min(11, int(self.cell_size * 0.12)))
                    self.canvas.create_text(
                        cx + self.cell_size / 2, cy + int(self.cell_size * 0.20), text=f"({x},{y})",
                        fill=coord_color, font=("Segoe UI", coord_font_size, "bold")
                    )
                    self.canvas.create_text(
                        cx + self.cell_size / 2, cy + int(self.cell_size * 0.44), text=label,
                        fill=label_color, font=("Segoe UI", label_font_size, "bold")
                    )

                    cell_signs = {}
                    for (sign_cell, _direction), signs in snap_signs:
                        if sign_cell == cell_coord:
                            for sign in signs:
                                cell_signs[(sign["color"], sign["shape"])] = sign

                    icon_spacing = max(9, min(14, int(self.cell_size * 0.14)))
                    for sign_index, sign in enumerate(cell_signs.values()):
                        icon_col, icon_row = sign_index % 6, sign_index // 6
                        sx = cx + int(self.cell_size * 0.16) + icon_col * icon_spacing
                        sy = cy + int(self.cell_size * 0.70) + icon_row * icon_spacing
                        color_bgr = COLORS[sign["color"]]["bgr"]
                        color_hex = "#%02x%02x%02x" % tuple(reversed(color_bgr))
                        irad = max(3, int(self.cell_size * 0.04))
                        if sign["shape"] == "circle":
                            self.canvas.create_oval(sx - irad, sy - irad, sx + irad, sy + irad,
                                                    fill=color_hex, outline="#0f172a", width=1)
                        elif sign["shape"] == "square":
                            self.canvas.create_rectangle(sx - irad, sy - irad, sx + irad, sy + irad,
                                                         fill=color_hex, outline="#0f172a", width=1)
                        elif sign["shape"] == "horizontal":
                            self.canvas.create_rectangle(sx - irad - 1, sy - irad + 1, sx + irad + 1, sy + irad - 1,
                                                         fill=color_hex, outline="#0f172a", width=1)
                        elif sign["shape"] == "vertical":
                            self.canvas.create_rectangle(sx - irad + 1, sy - irad - 1, sx + irad - 1, sy + irad + 1,
                                                         fill=color_hex, outline="#0f172a", width=1)

            # 2. Draw foam wall lines
            wall_color = "#e11d48" # Laser Crimson Red
            wall_glow = "#fecdd3"  # Soft Rose Red aura

            # Horizontal foam walls: between (x, y) and (x, y+1)
            for (wx, wy) in snap_h_walls:
                lx1 = self.pad + (wx - 1) * self.cell_size
                lx2 = lx1 + self.cell_size
                ly = self.pad + (cur_gh - wy) * self.cell_size
                self.canvas.create_line(lx1, ly, lx2, ly, fill=wall_glow, width=7, capstyle=tk.ROUND)
                self.canvas.create_line(lx1, ly, lx2, ly, fill=wall_color, width=4, capstyle=tk.ROUND)

            # Vertical foam walls: between (x, y) and (x+1, y)
            for (wx, wy) in snap_v_walls:
                lx = self.pad + wx * self.cell_size
                ly1 = self.pad + (cur_gh - wy) * self.cell_size
                ly2 = ly1 + self.cell_size
                self.canvas.create_line(lx, ly1, lx, ly2, fill=wall_glow, width=7, capstyle=tk.ROUND)
                self.canvas.create_line(lx, ly1, lx, ly2, fill=wall_color, width=4, capstyle=tk.ROUND)

            # 3. Draw trajectory path
            for item in snap_trajectory:
                if item["previous"] is None:
                    continue
                previous = item["previous"]
                current = item["cell"]
                x1 = self.pad + (previous[0] - 0.5) * self.cell_size
                y1 = self.pad + (cur_gh - previous[1] + 0.5) * self.cell_size
                x2 = self.pad + (current[0] - 0.5) * self.cell_size
                y2 = self.pad + (cur_gh - current[1] + 0.5) * self.cell_size
                self.canvas.create_line(x1, y1, x2, y2, fill="#2563eb", width=3)
                self.canvas.create_oval(x2 - 3, y2 - 3, x2 + 3, y2 + 3, fill="#3b82f6", outline="#ffffff", width=1)

            # 4. Draw robot chassis marker
            cx = self.pad + (position[0] - 0.5) * self.cell_size
            cy = self.pad + (cur_gh - position[1] + 0.5) * self.cell_size
            radius = max(14, int(self.cell_size * 0.22))

            self.canvas.create_oval(
                cx - radius, cy - radius, cx + radius, cy + radius,
                fill="#1d4ed8", outline="#ffffff", width=2.5
            )
            self.canvas.create_text(
                cx, cy, text=SYMBOL.get(heading, "^"), fill="white",
                font=("Segoe UI", max(10, int(radius * 0.9)), "bold")
            )

            # 5. Draw active gimbal scanning beam if scanning
            if gimbal_dir and gimbal_dir in DELTA:
                gdx, gdy = DELTA[gimbal_dir]
                beam_len = max(24, int(self.cell_size * 0.40))
                beam_x = cx + int(gdx * beam_len)
                beam_y = cy - int(gdy * beam_len)
                self.canvas.create_line(cx, cy, beam_x, beam_y, fill="#d97706", width=3, arrow=tk.LAST)

            visited_cnt = len(snap_visited)
            total_cells = cur_gw * cur_gh
            wall_cnt = len(snap_h_walls) + len(snap_v_walls)
            target_cnt = sum(len(s) for _, s in snap_signs)

            readings_str = ""
            if extra_readings:
                readings_str = "  |  ToF: " + " ".join(f"{k[0]}:{v:.0f}mm" for k, v in extra_readings.items())

            timer_str = self.get_current_timer_str()
            timer_badge = f"   [⏱️ {self._active_round_label or 'เวลา'}: {timer_str}]" if self._timer_running else f"   [⏱️ เวลา: {timer_str}]"

            self.status.config(text=(
                f"สถานะปัจจุบัน: {status}{timer_badge}\n"
                f"รอบที่: {step}   |   พิกัดปัจจุบัน: {position}   |   ทิศทางหุ่น: {heading}\n"
                f"สำรวจสำเร็จ: {visited_cnt}/{total_cells} ช่อง ({visited_cnt / total_cells * 100:.1f}%)   |   กำแพงโฟม: {wall_cnt} แนว   |   เป้าหมายที่พบ: {target_cnt} เป้า\n"
                f"ระยะ ToF ด้านหน้า: {snap_distance:.0f} mm{readings_str}"
            ))

        # Robot telemetry can request redraws faster than Tk can paint them.
        # Keep just the newest map pose/state rather than queueing stale full
        # canvas redraws that make the GUI lag behind and eventually freeze.
        self._post_ui(_do_update, key="map_update")

    def update_camera_frame(self, frame):
        """Display a cropped, annotated BGR frame on the Tk main thread."""
        if self.closed or frame is None:
            return

        with self._camera_lock:
            self._camera_pending_frame = frame
            if self._camera_callback_pending:
                return
            self._camera_callback_pending = True
        self._post_ui(self._drain_camera_frame)

    def _drain_ui_queue(self):
        if self.closed:
            return
        # Process pending UI updates up to 15ms to keep UI snappy without starving Windows event pump
        start_t = time.monotonic()
        while time.monotonic() - start_t < 0.015:
            try:
                callback = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                if isinstance(callback, tuple) and callback[:1] == ("coalesced",):
                    key = callback[1]
                    with self._ui_coalesce_lock:
                        callback = self._ui_coalesced.pop(key, None)
                        self._ui_coalesced_scheduled.discard(key)
                if callback is not None:
                    callback()
            except Exception:
                traceback.print_exc()
        try:
            self.root.after(30, self._drain_ui_queue)
        except (tk.TclError, Exception):
            pass

    def _report_tk_callback_exception(self, exc_type, exc_value, exc_tb):
        """Report Tk callback errors without allowing them to close the GUI."""
        traceback.print_exception(exc_type, exc_value, exc_tb)
        try:
            self.log(f"GUI callback error: {exc_value}")
        except Exception:
            pass

    def _drain_camera_frame(self):
        with self._camera_lock:
            frame = self._camera_pending_frame
            self._camera_pending_frame = None
            self._camera_callback_pending = False
        if self.closed or frame is None:
            return
        now = time.monotonic()
        if now - self._last_camera_display < 0.20:
            return
        self._last_camera_display = now
        try:
            self._show_camera_frame(frame)
        except Exception:
            pass

    def _refresh_camera(self):
        if self.closed:
            return
        try:
            if self.vision_reader is not None:
                now = time.monotonic()
                if now - self._last_camera_display >= 0.20:  # 5 FPS (smooth & lightweight)
                    from vision import crop_view
                    _frame_id, frame = self.vision_reader.latest()
                    if frame is not None and _frame_id != self._last_camera_source_frame_id:
                        self._last_camera_display = now
                        self._last_camera_source_frame_id = _frame_id
                        view = crop_view(frame)
                        self._show_camera_frame(view)
        except Exception:
            pass
        finally:
            if not self.closed:
                try:
                    self.root.after(200, self._refresh_camera)
                except (tk.TclError, Exception):
                    pass
