# -*- coding: utf-8 -*-
"""Standalone RoboMaster live camera and color/shape detection preview.

Run with: python vision_preview.py
This preview does not drive the chassis or pan the gimbal. It only tilts pitch
down to the configured limit, then displays the live cropped camera feed.
"""

import threading
import tkinter as tk

import cv2
from PIL import Image, ImageTk
import vision as vision_config

from vision import (
    COLOR_SCAN_PITCH,
    COLOR_SCAN_SPEED,
    COLORS,
    MIN_AREA,
    SHAPES,
    CameraFrameReader,
    VisionFrameAnalyzer,
)

try:
    from robomaster import robot as robomaster
except ImportError:
    robomaster = None


class VisionPreview:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RoboMaster — Live Color & Shape Check")
        self.root.geometry("1220x780")
        self.root.minsize(1000, 640)
        self.root.configure(bg="#07111f")
        self.closing = threading.Event()
        self.ep_robot = None
        self.camera = None
        self.reader = None
        self.analyzer = None
        self.camera_started = False
        self.last_frame_id = -1
        self.photo = None
        self.analysis_stop = threading.Event()
        self.analysis_lock = threading.Lock()
        self.analysis_thread = None
        self.analysis_frame_id = -1
        self.analysis_view = None
        self.analysis_detections = []

        header = tk.Label(
            self.root,
            text="LIVE CAMERA · COLOR / SHAPE DETECTOR",
            bg="#07111f", fg="#7df9ff", font=("Segoe UI", 17, "bold"),
        )
        header.pack(anchor="w", padx=18, pady=(14, 8))

        body = tk.Frame(self.root, bg="#07111f")
        body.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 14))

        self.image_label = tk.Label(
            body, text="Connect to RoboMaster to start the live preview",
            bg="#020617", fg="#8eb9d6", font=("Segoe UI", 12),
            highlightbackground="#1d4e73", highlightthickness=1,
        )
        self.image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        panel = tk.Frame(body, bg="#0d1b2a", width=285, padx=16, pady=16)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(14, 0))
        panel.pack_propagate(False)

        self.status_var = tk.StringVar(value="Ready — chassis will remain still")
        tk.Label(
            panel, textvariable=self.status_var, justify=tk.LEFT, wraplength=245,
            bg="#0d1b2a", fg="#d9f3ff", font=("Segoe UI", 10),
        ).pack(fill=tk.X, anchor="w", pady=(0, 12))

        self.connect_button = tk.Button(
            panel, text="เชื่อมต่อกล้องและก้ม Gimbal",
            bg="#0284c7", fg="white", activebackground="#0369a1",
            font=("Segoe UI", 10, "bold"), relief=tk.FLAT,
            command=self.connect,
        )
        self.connect_button.pack(fill=tk.X, pady=(0, 18))

        tk.Label(
            panel, text="ตรวจพบในภาพล่าสุด",
            bg="#0d1b2a", fg="#7df9ff", font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", pady=(0, 8))
        self.detections_text = tk.Text(
            panel, height=12, wrap=tk.WORD, bg="#07111f", fg="#d9f3ff",
            font=("Segoe UI", 10), padx=10, pady=10, bd=0,
        )
        self.detections_text.pack(fill=tk.BOTH, expand=True)
        for color_name, config in COLORS.items():
            color_hex = "#%02x%02x%02x" % tuple(reversed(config["bgr"]))
            self.detections_text.tag_configure(
                color_name, foreground=color_hex, font=("Segoe UI", 10, "bold")
            )
        self._set_detection_rows([])

        self.min_area_var = tk.IntVar(value=vision_config.MIN_AREA)
        self.max_area_var = tk.DoubleVar(value=vision_config.MAX_OBJECT_AREA_RATIO)
        self.max_dimension_var = tk.DoubleVar(value=vision_config.MAX_OBJECT_DIMENSION_RATIO)
        settings = tk.LabelFrame(panel, text="Object size filter", bg="#0d1b2a", fg="#00e5ff",
                                 padx=8, pady=5)
        settings.pack(fill=tk.X, pady=(8, 0))
        self._add_size_scale(settings, "Minimum area (px²)", self.min_area_var,
                             15000, 50000, 1000, self._apply_size_settings)
        self._add_size_scale(settings, "Maximum area (% of ROI)", self.max_area_var,
                             0.05, 0.15, 0.01, self._apply_size_settings, digits=2)
        self._add_size_scale(settings, "Maximum width/height (% ROI)", self.max_dimension_var,
                             0.25, 0.90, 0.01, self._apply_size_settings, digits=2)

        tk.Label(
            panel,
            text=f"Size shown in pixels · physical cm needs camera calibration\nPitch = {COLOR_SCAN_PITCH}° (ก้มสุด)",
            justify=tk.LEFT, bg="#0d1b2a", fg="#8eb9d6", font=("Consolas", 9),
        ).pack(anchor="w", pady=(12, 0))

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self.refresh_visualization)

    def _add_size_scale(self, parent, label, variable, minimum, maximum, resolution, callback, digits=0):
        tk.Label(parent, text=label, bg="#0d1b2a", fg="#d9f3ff",
                 font=("Segoe UI", 8)).pack(anchor="w")
        scale = tk.Scale(parent, from_=minimum, to=maximum, resolution=resolution,
                         orient=tk.HORIZONTAL, variable=variable, showvalue=True,
                         digits=digits, command=lambda _value: callback(), length=225,
                         bg="#0d1b2a", fg="#d9f3ff", highlightthickness=0)
        scale.pack(fill=tk.X)

    def _apply_size_settings(self):
        vision_config.MIN_AREA = self.min_area_var.get()
        vision_config.MAX_OBJECT_AREA_RATIO = self.max_area_var.get()
        vision_config.MAX_OBJECT_DIMENSION_RATIO = self.max_dimension_var.get()

    def connect(self):
        if self.ep_robot is not None or self.connect_button["state"] == tk.DISABLED:
            return
        if robomaster is None:
            self.status_var.set("RoboMaster SDK is not installed in this Python environment.")
            return
        self.connect_button.config(state=tk.DISABLED)
        self.status_var.set("Connecting… the chassis will not move.")
        threading.Thread(target=self._connect_worker, name="robomaster-preview-connect", daemon=True).start()

    def _connect_worker(self):
        ep_robot = None
        camera = None
        try:
            ep_robot = robomaster.Robot()
            ep_robot.initialize(conn_type="ap")
            if self.closing.is_set():
                ep_robot.close()
                return

            gimbal = ep_robot.gimbal
            angle_ready = threading.Event()
            angle_state = {}

            def save_angle(info):
                if isinstance(info, (list, tuple)) and len(info) >= 2:
                    angle_state["pitch"] = float(info[0])
                    angle_ready.set()

            gimbal.sub_angle(freq=10, callback=save_angle)
            if not angle_ready.wait(timeout=3.0):
                raise RuntimeError("Gimbal angle was not received; cannot set the down angle safely.")

            # Relative pitch-only move preserves the current yaw exactly.
            pitch_delta = COLOR_SCAN_PITCH - angle_state["pitch"]
            if abs(pitch_delta) >= 0.5:
                gimbal.move(
                    pitch=pitch_delta, yaw=0,
                    pitch_speed=COLOR_SCAN_SPEED, yaw_speed=0,
                ).wait_for_completed()
            gimbal.unsub_angle()

            camera = ep_robot.camera
            if not camera.start_video_stream(display=False, resolution="720p"):
                raise RuntimeError("RoboMaster video stream did not start.")

            self.ep_robot = ep_robot
            self.camera = camera
            self.camera_started = True
            self.reader = CameraFrameReader(camera)
            self.reader.start()
            self.analyzer = VisionFrameAnalyzer(self.reader)
            # The analyzer is disabled by default for the main robot flow;
            # this standalone preview needs it to publish frames for display.
            self.analyzer.set_enabled(True)
            self.analyzer.start()
            self.root.after(0, lambda: self.status_var.set(
                "Connected · gimbal pitched down · chassis and yaw unchanged"
            ))
        except Exception as exc:
            if camera is not None:
                try:
                    camera.stop_video_stream()
                except Exception:
                    pass
            if ep_robot is not None:
                try:
                    ep_robot.close()
                except Exception:
                    pass
            if not self.closing.is_set():
                self.root.after(0, lambda message=str(exc): self._connect_failed(message))

    def _connect_failed(self, message):
        if self.closing.is_set():
            return
        self.status_var.set(f"Camera connection failed: {message}")
        self.connect_button.config(state=tk.NORMAL)

    def _set_detection_rows(self, detections):
        self.detections_text.configure(state=tk.NORMAL)
        self.detections_text.delete("1.0", tk.END)
        if not detections:
            self.detections_text.insert(tk.END, "No color/shape found inside the detection area.")
        else:
            unique = {}
            for item in detections:
                unique[(item["color"], item["shape"])] = item
            for (color, shape), item in sorted(unique.items()):
                width, height = item.get("size_px", (0, 0))
                row = (f"● {COLORS[color]['label']} · {SHAPES[shape]} · "
                       f"{width}×{height}px · area {item['area']:.0f}px²\n")
                self.detections_text.insert(tk.END, row, color)
        self.detections_text.configure(state=tk.DISABLED)

    def refresh_visualization(self):
        if self.closing.is_set():
            return
        if self.analyzer is not None:
            frame_id, view, detections = self.analyzer.latest()
            if view is not None and frame_id != self.last_frame_id:
                self.last_frame_id = frame_id
                rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                max_width = max(320, self.image_label.winfo_width() - 20)
                max_height = max(240, self.image_label.winfo_height() - 20)
                scale = min(max_width / image.width, max_height / image.height)
                size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
                image = image.resize(size, Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=self.photo, text="")
                self._set_detection_rows(detections)
        self.root.after(80, self.refresh_visualization)

    def close(self):
        if self.closing.is_set():
            return
        self.closing.set()
        if self.analyzer is not None:
            self.analyzer.stop()
        if self.reader is not None:
            self.reader.stop()
        if self.camera_started and self.camera is not None:
            try:
                self.camera.stop_video_stream()
            except Exception:
                pass
        if self.ep_robot is not None:
            try:
                self.ep_robot.close()
            except Exception:
                pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    VisionPreview().run()
