# -*- coding: utf-8 -*-
"""Color and shape recognition for signs seen during a stationary gimbal scan."""

from collections import Counter, deque
import threading
import time

import cv2
import numpy as np

# Keep image analysis from taking every CPU core away from the Tk interface.
cv2.setNumThreads(2)

MIN_AREA = 2500
# Reject weakly colored regions such as floor glare and reflections even when
# their shape happens to resemble a sign.
MIN_COLOR_PURITY = 0.72
# Large flat patches (often tape or colored clutter) are not useful targets.
# Cap both total area and a single dimension relative to the detection window.
MAX_OBJECT_AREA_RATIO = 0.40
MAX_OBJECT_DIMENSION_RATIO = 0.75
MIN_OBJECT_DIMENSION_PX = 35
SQUARE_RATIO = 0.75
COLORS = {
    "red": {
        "label": "แดง",
        "bgr": (40, 40, 240),
        "ranges": [((0, 110, 65), (12, 255, 255)), ((168, 110, 65), (179, 255, 255))],
    },
    "green": {
        "label": "เขียว",
        "bgr": (50, 200, 50),
        # Reject low-saturation grey wall pixels that merely have a green hue.
        # Green signs can be dark; retain their high chroma requirement while
        # allowing lower V. Neutral grey shadows still fail the S threshold.
        "ranges": [((36, 70, 25), (86, 255, 255))],
    },
    "yellow": {
        "label": "เหลือง",
        "bgr": (60, 220, 240),
        # Saturation floor leaves headroom for uneven illumination and camera AWB.
        "ranges": [((20, 100, 60), (50, 255, 255))],
    },
    "blue": {
        "label": "น้ำเงิน",
        "bgr": (240, 100, 30),
        "ranges": [((90, 75, 30), (140, 255, 255))],
    },
}
SHAPES = {
    "circle": "วงกลม",
    "square": "จัตุรัส",
    "horizontal": "ผืนผ้าแนวนอน",
    "vertical": "ผืนผ้าแนวตั้ง",
}

# Camera framing controls: these are fractions of the original frame.
# Tighten by increasing the first number or decreasing the second number.
# The right and bottom edges are trimmed more to remove the red blaster/handle.
CROP_X = (0, 1)
CROP_Y = (0, 1)
# Digital zoom applied after the outer crop; 1.0 disables zoom.
CAMERA_ZOOM = 1.00
CAMERA_HFOV_DEG = 96.0
CAMERA_VFOV_DEG = 54.0
# Color/shape detection ignores the outer edge of the displayed crop to reduce
# detections on walls at the sides. Widen toward 0.0/1.0 to include more view.
DETECTION_ROI = (0.03, 0.97)
# Camera-mounted blaster/turret occlusion in normalized (x, y) coordinates.
# Mask the visible blaster/barrel near the lower center. Keep this narrow so
# it does not cut off real targets that extend toward the lower-right of view.
ROBOT_OCCLUSION_POLYGON = ((0.465, 0.84), (0.55, 0.84), (0.63, 1.0), (0.39, 1.0))
COLOR_SCAN_DURATION_S = 2.0
COLOR_VOTE_FRAME_COUNT = 6
COLOR_VOTE_MIN_COUNT = 4
COLOR_VOTE_FRAME_INTERVAL_S = 0.08
COLOR_VOTE_TIMEOUT_S = 2.2
# RoboMaster controllable pitch limit is -20..35 degrees; -20 is the
# lowest commanded angle and points the camera down for wall color scans.
COLOR_SCAN_PITCH = -20
COLOR_SCAN_SPEED = 90
# Keep raw brightness for the color mask. CLAHE boosts dark shadows and can make
# them pass HSV thresholds as if they were strongly colored objects.
LIGHT_NORMALIZATION = False
STABLE_WINDOW = 15
STABLE_HITS = 3
STABLE_MISSES = 10
MORPH_KERNEL_SIZES = (3, 5, 7)
KERNEL_VOTES_REQUIRED = 2
CONTOUR_MATCH_IOU = 0.25


def crop_view(frame):
    height, width = frame.shape[:2]
    x0, x1 = int(width * CROP_X[0]), int(width * CROP_X[1])
    y0, y1 = int(height * CROP_Y[0]), int(height * CROP_Y[1])
    view = frame[y0:y1, x0:x1].copy()
    if CAMERA_ZOOM > 1.0 and view.size:
        view_height, view_width = view.shape[:2]
        zoom_width = max(1, int(view_width / CAMERA_ZOOM))
        zoom_height = max(1, int(view_height / CAMERA_ZOOM))
        left = (view_width - zoom_width) // 2
        top = (view_height - zoom_height) // 2
        view = cv2.resize(
            view[top:top + zoom_height, left:left + zoom_width],
            (view_width, view_height), interpolation=cv2.INTER_CUBIC,
        )
    return view


def _shape_for_contour(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0 or area / hull_area < 0.68:
        return None
    polygon = cv2.approxPolyDP(hull, 0.018 * perimeter, True)
    x, y, width, height = cv2.boundingRect(contour)
    if 3 <= len(polygon) <= 5:
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        if not rw or not rh:
            return None
        ratio = min(rw, rh) / max(rw, rh)
        # Tilt/low viewing angle turns a square into a perspective trapezoid.
        # Treat a convex four-corner contour as square when it is not an extreme
        # elongated rectangle; axis-aligned bounding boxes exaggerate tilt.
        if ratio >= SQUARE_RATIO:
            return "square"
        # OpenCV can swap minAreaRect width/height when its angle wraps at 90°.
        # Normalize the long side and use its angle to keep orientation stable.
        angle = rect[2]
        if rw < rh:
            rw, rh = rh, rw
            angle += 90.0
        angle = ((angle + 90.0) % 180.0) - 90.0
        return "horizontal" if abs(angle) < 45.0 else "vertical"

    circularity = 4 * np.pi * area / (perimeter * perimeter)
    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    ratio = min(rw, rh) / max(rw, rh) if min(rw, rh) else 0
    # A printed circle viewed from below/at an angle appears as an ellipse.
    # A circle viewed obliquely becomes an ellipse. Fit-ellipse eccentricity
    # provides a scale/rotation-independent check, unlike the axis-aligned box.
    eccentricity = 1.0
    if len(contour) >= 5:
        (_, _), axes, _ = cv2.fitEllipse(contour)
        minor, major = sorted(axes)
        eccentricity = minor / major if major else 0.0
    # Loose thresholds make crumpled tape and irregular colored scraps look
    # like circles. Require a noticeably round, compact contour.
    if circularity >= 0.62 and ratio >= 0.50 and eccentricity >= 0.52:
        return "circle"
    return None


def _box_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix, iy = max(ax, bx), max(ay, by)
    iw, ih = max(0, min(ax + aw, bx + bw) - ix), max(0, min(ay + ah, by + bh) - iy)
    intersection = iw * ih
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def _merge_kernel_candidates(candidates):
    """Merge overlapping detections from different morphology kernel sizes."""
    clusters = []
    for candidate in sorted(candidates, key=lambda c: c["area"], reverse=True):
        match = None
        for cluster in clusters:
            if _box_iou(candidate["bbox"], cluster["bbox"]) >= CONTOUR_MATCH_IOU:
                match = cluster
                break
        if match is None:
            clusters.append({"items": [candidate], "bbox": candidate["bbox"]})
        else:
            match["items"].append(candidate)
            boxes = [item["bbox"] for item in match["items"]]
            x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
            x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
            match["bbox"] = (x0, y0, x1 - x0, y1 - y0)
    merged = []
    for cluster in clusters:
        items = cluster["items"]
        # Several sizes detecting the same region raises confidence. A large,
        # clean object may survive only one kernel, so retain it if its area is
        # substantial; small isolated specks are discarded.
        votes = len({item["kernel"] for item in items})
        if votes < KERNEL_VOTES_REQUIRED and max(item["area"] for item in items) < MIN_AREA * 2:
            continue
        shape_votes = Counter(item["shape"] for item in items)
        shape = shape_votes.most_common(1)[0][0]
        merged.append({"shape": shape, "bbox": cluster["bbox"],
                       "area": max(item["area"] for item in items), "kernel_votes": votes})
    return merged


def _detect_signs_detailed(frame, color_filter=None, shape_filter=None):
    """Return annotated view, detections and a color-coded filtered-mask preview."""
    view = crop_view(frame)
    # Apply Gaussian blur to suppress noise/grain and smooth out background clutter
    hsv = cv2.cvtColor(cv2.GaussianBlur(view, (5, 5), 0), cv2.COLOR_BGR2HSV)
    if LIGHT_NORMALIZATION:
        hsv[:, :, 2] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(hsv[:, :, 2])
    detections = []
    filtered = np.zeros((height := view.shape[0], width := view.shape[1]), dtype=np.uint8)
    # Different structuring-element geometries preserve different contours:
    # ellipse for circles, rectangle for sign edges, and cross for noisy borders.
    kernels = [(cv2.MORPH_ELLIPSE, size,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
               for size in MORPH_KERNEL_SIZES]
    height, width = hsv.shape[:2]
    roi_start = int(width * DETECTION_ROI[0])
    roi_end = int(width * DETECTION_ROI[1])
    # Cut off top 12% background looking above walls into the room
    roi_top = int(height * max(0.12, DETECTION_ROI[0]))
    roi_bottom = int(height * DETECTION_ROI[1])
    cv2.rectangle(view, (roi_start, roi_top), (roi_end - 1, roi_bottom - 1), (255, 255, 255), 2)
    hsv_roi = hsv[roi_top:roi_bottom, roi_start:roi_end]
    ignore_polygon = np.array([[(int(x * width) - roi_start, int(y * height) - roi_top)
                                for x, y in ROBOT_OCCLUSION_POLYGON]], dtype=np.int32)
    ignore_mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
    cv2.fillPoly(ignore_mask, ignore_polygon, 255)

    selected_colors = set(COLORS) if color_filter is None else set(color_filter)
    for color_name, config in COLORS.items():
        if color_name not in selected_colors:
            continue
        mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
        for lower, upper in config["ranges"]:
            mask |= cv2.inRange(hsv_roi, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        mask[ignore_mask > 0] = 0
        kernel_votes = np.zeros(mask.shape, dtype=np.uint8)
        for _, _, kernel in kernels:
            pass_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            pass_mask = cv2.morphologyEx(pass_mask, cv2.MORPH_CLOSE, kernel)
            kernel_votes += (pass_mask > 0).astype(np.uint8)
        # Segment by pixelwise consensus first, then find contours once. This
        # prevents boxes from different kernels being unioned into oversized boxes.
        consensus_mask = np.where(kernel_votes >= KERNEL_VOTES_REQUIRED, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(consensus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA or area / max(1, mask.shape[0] * mask.shape[1]) > MAX_OBJECT_AREA_RATIO:
                continue
            shape = _shape_for_contour(contour)
            if shape is None or (shape_filter is not None and shape not in shape_filter):
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if (box_width < MIN_OBJECT_DIMENSION_PX or box_height < MIN_OBJECT_DIMENSION_PX or
                box_width / max(1, mask.shape[1]) > MAX_OBJECT_DIMENSION_RATIO or
                box_height / max(1, mask.shape[0]) > MAX_OBJECT_DIMENSION_RATIO):
                continue
            accepted_component = np.zeros_like(consensus_mask)
            cv2.drawContours(accepted_component, [contour], -1, 255, thickness=cv2.FILLED)
            component_pixels = max(1, cv2.countNonZero(accepted_component))
            color_pixels = cv2.countNonZero(cv2.bitwise_and(mask, accepted_component))
            color_purity = color_pixels / component_pixels
            if color_purity < MIN_COLOR_PURITY:
                continue
            detections.append({"color": color_name, "shape": shape, "area": float(area),
                "bbox": (x + roi_start, y + roi_top, box_width, box_height),
                "size_px": (box_width, box_height),
                "_color_purity": color_purity,
                "_component": accepted_component})

    # Resolve competing colors/shapes globally. Keep the most color-pure
    # candidate first, and reject every later box that shares even one pixel
    # with an accepted box so the GUI/map never receives overlapping objects.
    detections.sort(key=lambda item: (item["_color_purity"], item["area"]), reverse=True)
    non_overlapping = []
    for candidate in detections:
        x, y, w, h = candidate["bbox"]
        overlaps = False
        for kept in non_overlapping:
            kx, ky, kw, kh = kept["bbox"]
            if x < kx + kw and kx < x + w and y < ky + kh and ky < y + h:
                overlaps = True
                break
        if not overlaps:
            non_overlapping.append(candidate)

    filtered.fill(0)
    for item in non_overlapping:
        filtered[roi_top:roi_bottom, roi_start:roi_end] |= item.pop("_component")
        item.pop("_color_purity", None)
    detections = non_overlapping
    # Keep the exact detection area visible in the binary filter preview.
    cv2.rectangle(filtered, (roi_start, roi_top), (roi_end - 1, roi_bottom - 1), 180, 2)
    return view, detections, filtered


def detect_signs(frame):
    """Backward-compatible detector API used by the robot controller."""
    view, detections, _ = _detect_signs_detailed(frame)
    return view, detections


class CameraFrameReader:
    """Continuously read the RoboMaster stream and share the newest frame."""

    def __init__(self, camera, dashboard=None):
        self.camera = camera
        self.dashboard = dashboard
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._read_loop, name="robomaster-camera", daemon=True)
        self._thread.start()

    def _read_loop(self):
        while not self._stop.is_set():
            try:
                frame = self.camera.read_cv2_image(timeout=1, strategy="newest")
            except Exception:
                frame = None
            if frame is None:
                self._stop.wait(0.05)
                continue
            with self._lock:
                self._frame = frame.copy()
                self._frame_id += 1

    def latest(self):
        with self._lock:
            # The reader replaces each frame rather than mutating it, so callers
            # can safely inspect the shared read-only image without another copy.
            return self._frame_id, self._frame

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.2)


class VisionFrameAnalyzer:
    """Run color/shape detection off the Tk thread and retain the latest result."""

    def __init__(self, frame_reader, interval_s=0.12):
        self.frame_reader = frame_reader
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame_id = -1
        self._view = None
        self._detections = []
        self._raw_detections = []
        self._filter_view = None
        self._filter_history = deque(maxlen=5)
        # The robot is stationary during color scans, so a short temporal
        # median removes single-frame glare/exposure changes without smearing
        # moving camera content across a long interval.
        self._vision_frames = deque(maxlen=3)
        self._tracks = {}
        self.color_filter = None
        self.shape_filter = None
        self._thread = None
        self._enabled = False
        self._reset_analysis = False

    def set_enabled(self, enabled):
        """Run color/shape analysis only during an explicitly requested scan."""
        enabled = bool(enabled)
        with self._lock:
            if self._enabled == enabled:
                return
            self._enabled = enabled
            self._reset_analysis = True
            self._detections = []
            self._raw_detections = []
            self._filter_view = None

    def set_filters(self, color_filter=None, shape_filter=None):
        with self._lock:
            self.color_filter = None if color_filter is None else set(color_filter)
            self.shape_filter = None if shape_filter is None else set(shape_filter)

    def start(self):
        self._thread = threading.Thread(target=self._analyze_loop, name="robomaster-vision", daemon=True)
        self._thread.start()

    def _analyze_loop(self):
        last_frame_id = -1
        last_analysis = 0.0
        while not self._stop.is_set():
            frame_id, frame = self.frame_reader.latest()
            now = time.monotonic()
            with self._lock:
                enabled = self._enabled
                reset_analysis = self._reset_analysis if enabled else False
                if reset_analysis:
                    self._reset_analysis = False
                    self._tracks.clear()
                    self._filter_history.clear()
                    self._vision_frames.clear()
            if not enabled:
                last_frame_id = frame_id
                self._stop.wait(0.03)
                continue
            if frame is None or frame_id == last_frame_id or now - last_analysis < self.interval_s:
                self._stop.wait(0.03)
                continue
            last_frame_id = frame_id
            last_analysis = now
            try:
                with self._lock:
                    color_filter = None if self.color_filter is None else set(self.color_filter)
                    shape_filter = None if self.shape_filter is None else set(self.shape_filter)
                with self._lock:
                    self._vision_frames.append(frame.copy())
                    frame_history = list(self._vision_frames)
                stable_input = np.median(np.stack(frame_history, axis=0), axis=0).astype(np.uint8)
                view, detections, filter_view = _detect_signs_detailed(stable_input, color_filter, shape_filter)
            except Exception:
                continue
            with self._lock:
                if not self._enabled:
                    continue
            stable = self._stabilize(detections)
            with self._lock:
                self._filter_history.append(filter_view.copy())
                history = list(self._filter_history)
            white_votes = np.sum([item > 200 for item in history], axis=0)
            filter_view = np.where(white_votes >= 3, 255, 0).astype(np.uint8)
            height, width = filter_view.shape[:2]
            margin, far_edge = DETECTION_ROI
            cv2.rectangle(filter_view, (int(width * margin), int(height * margin)),
                          (int(width * far_edge) - 1, int(height * far_edge) - 1), 180, 2)
            for item in stable:
                x, y, w, h = item["bbox"]
                cfg = COLORS[item["color"]]
                cv2.rectangle(view, (x, y), (x + w, y + h), cfg["bgr"], 2)
                cv2.putText(view, f'{item["color"]} {item["shape"]} {w}x{h}px', (x, max(18, y - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, cfg["bgr"], 2, cv2.LINE_AA)
            with self._lock:
                if not self._enabled:
                    continue
                self._frame_id = frame_id
                self._view = view
                self._detections = stable
                self._raw_detections = detections
                self._filter_view = filter_view

    def _stabilize(self, detections):
        # Match same color/shape by IoU; a candidate must persist in most recent frames.
        assigned = set()
        for track_id, track in list(self._tracks.items()):
            track["history"].append(False)
            track["misses"] += 1
        for item in detections:
            x, y, w, h = item["bbox"]
            best_id, best_iou = None, 0.0
            for track_id, track in self._tracks.items():
                if track_id in assigned or track["color"] != item["color"] or track["shape"] != item["shape"]:
                    continue
                a, b, aw, ah = track["bbox"]
                ix, iy = max(x, a), max(y, b)
                iw, ih = max(0, min(x+w, a+aw)-ix), max(0, min(y+h, b+ah)-iy)
                intersection = iw * ih
                union = w*h + aw*ah - intersection
                iou = intersection / union if union else 0
                if iou > best_iou:
                    best_id, best_iou = track_id, iou
            if best_id is None or best_iou < 0.18:
                best_id = self._next_track_id = getattr(self, "_next_track_id", 0) + 1
                self._tracks[best_id] = {"color": item["color"], "shape": item["shape"],
                    "bbox": item["bbox"], "history": deque(maxlen=STABLE_WINDOW), "misses": 0}
            track = self._tracks[best_id]
            track["history"].append(True)
            track["misses"] = 0
            track["bbox"] = tuple(int(0.55 * old + 0.45 * new) for old, new in zip(track["bbox"], item["bbox"]))
            track["item"] = {**item, "bbox": track["bbox"],
                              "size_px": (track["bbox"][2], track["bbox"][3])}
            assigned.add(best_id)
        stable = []
        for track_id, track in list(self._tracks.items()):
            if track["misses"] > STABLE_MISSES:
                del self._tracks[track_id]
            elif len(track["history"]) >= STABLE_HITS and sum(track["history"]) >= STABLE_HITS and "item" in track:
                stable.append(track["item"])
        return stable

    def latest(self):
        with self._lock:
            return self._frame_id, self._view, list(self._detections)

    def latest_filter(self):
        with self._lock:
            return self._frame_id, self._filter_view, list(self._raw_detections)

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)


def detect_stable_signs(frame_reader, dashboard=None, color_filter=None, shape_filter=None):
    """Require the same color, shape, and image location across fresh frames."""
    tracks = []
    frame_observations = []
    last_view = None
    processed_frames = 0
    last_frame_id = -1
    last_analysis_time = 0.0
    deadline = time.monotonic() + COLOR_VOTE_TIMEOUT_S
    while processed_frames < COLOR_VOTE_FRAME_COUNT and time.monotonic() < deadline:
        now = time.monotonic()
        frame_id, frame = frame_reader.latest()
        if (frame is not None and frame_id != last_frame_id and
                now - last_analysis_time >= COLOR_VOTE_FRAME_INTERVAL_S):
            last_frame_id = frame_id
            last_analysis_time = now
            processed_frames += 1
            last_view, detections, _ = _detect_signs_detailed(frame, color_filter, shape_filter)
            for item in detections:
                frame_observations.append({
                    "frame": processed_frames,
                    "color": item["color"], "shape": item["shape"],
                    "area": float(item["area"]),
                    "size_px": list(item["size_px"]),
                    "bbox": list(item["bbox"]),
                })
                best_track, best_iou = None, 0.0
                for track in tracks:
                    if (track["color"] != item["color"] or track["shape"] != item["shape"]
                            or track["last_frame"] < processed_frames - 2):
                        continue
                    overlap = _box_iou(track["observations"][-1]["bbox"], item["bbox"])
                    if overlap > best_iou:
                        best_track, best_iou = track, overlap
                if best_track is None or best_iou < 0.25:
                    best_track = {"color": item["color"], "shape": item["shape"],
                                  "observations": [], "last_frame": processed_frames}
                    tracks.append(best_track)
                best_track["observations"].append({**item, "frame_id": processed_frames})
                best_track["last_frame"] = processed_frames
        else:
            time.sleep(0.01)

    # Require consistent votes across fresh frames. Select a
    # representative close to the median observed area to avoid one bad frame.
    stable = []
    vote_rows = []
    for track in tracks:
        observations = track["observations"]
        count = len({item["frame_id"] for item in observations})
        vote_rows.append({"color": track["color"], "shape": track["shape"],
                          "votes": int(count), "confidence": count / COLOR_VOTE_FRAME_COUNT})
        min_required_votes = max(3, min(COLOR_VOTE_MIN_COUNT, processed_frames))
        if count < min_required_votes:
            continue
        median_area = float(np.median([item["area"] for item in observations]))
        representative = min(observations, key=lambda item: abs(float(item["area"]) - median_area))
        stable.append({**representative, "votes": int(count),
                       "frames_used": processed_frames,
                       "confidence": count / max(1, processed_frames)})
    if processed_frames < 3:
        stable = []
    summary = {
        "frames_expected": COLOR_VOTE_FRAME_COUNT,
        "frames_used": processed_frames,
        "minimum_votes": COLOR_VOTE_MIN_COUNT,
        "votes": sorted(vote_rows, key=lambda row: (-row["votes"], row["color"], row["shape"])),
        "frame_detections": frame_observations,
    }
    if dashboard and last_view is not None:
        for item in stable:
            x, y, w, h = item["bbox"]
            color = COLORS[item["color"]]["bgr"]
            cv2.rectangle(last_view, (x, y), (x + w, y + h), color, 2)
            cv2.putText(last_view, f'{item["color"]} {item["shape"]} {item["votes"]}/{COLOR_VOTE_FRAME_COUNT}',
                        (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        color, 2, cv2.LINE_AA)
        dashboard.update_camera_frame(last_view)
    return stable, last_view, summary


def save_target_snapshot(frame, cell, direction, item, fired=False, pitch=None, yaw=None, dashboard=None):
    """
    Save an annotated snapshot of a detected or fired target sign to results/targets/.
    Also registers the snapshot with the dashboard gallery.
    """
    import os
    if frame is None or not item:
        return None, None

    code_dir = os.path.dirname(os.path.abspath(__file__))
    targets_dir = os.path.join(code_dir, "results", "targets")
    os.makedirs(targets_dir, exist_ok=True)

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    time_display = time.strftime("%H:%M:%S")

    color_name = str(item.get("color", "unknown")).lower()
    shape_name = str(item.get("shape", "unknown")).lower()
    x_cell, y_cell = cell

    status_tag = "hit" if fired else "det"
    filename = f"target_{x_cell}_{y_cell}_{direction}_{color_name}_{shape_name}_{status_tag}_{timestamp_str}.jpg"
    filepath = os.path.join(targets_dir, filename)

    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # 1. Draw target bounding box
    bbox = item.get("bbox")
    bgr = COLORS.get(color_name, {}).get("bgr", (0, 255, 0))
    if bbox:
        bx, by, bw, bh = bbox
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), bgr, 3)
        # Center crosshair reticle
        cx, cy = int(bx + bw / 2), int(by + bh / 2)
        cv2.drawMarker(annotated, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.circle(annotated, (cx, cy), 14, (0, 255, 255), 1, cv2.LINE_AA)

    # 2. Draw Top Status Header Banner
    cv2.rectangle(annotated, (0, 0), (w, 44), (15, 23, 42), -1)
    badge_title = "🎯 TARGET HIT · FIRED 2x GEL BEADS" if fired else "🔍 TARGET DETECTED"
    badge_color = (16, 185, 129) if fired else (245, 158, 11)  # Emerald vs Amber
    cv2.circle(annotated, (20, 22), 7, badge_color, -1)
    cv2.putText(annotated, f"ROBOMASTER SLAM · {badge_title}", (35, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # 3. Draw Bottom Telemetry Banner
    cv2.rectangle(annotated, (0, h - 42), (w, h), (15, 23, 42), -1)
    area_px = int(item.get("area", 0))
    conf_pct = int(item.get("confidence", 1.0) * 100) if "confidence" in item else 100
    gimbal_info = f" | Pitch:{pitch:+.1f}deg Yaw:{yaw:+.1f}deg" if (pitch is not None and yaw is not None) else ""
    info_text = (f"Pos: ({x_cell},{y_cell}) {direction} | {color_name.upper()} {shape_name.upper()} | "
                 f"Area: {area_px}px | Conf: {conf_pct}%{gimbal_info} | Time: {time_display}")
    cv2.putText(annotated, info_text, (12, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (226, 232, 240), 1, cv2.LINE_AA)

    cv2.imwrite(filepath, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

    metadata = {
        "filepath": filepath,
        "filename": filename,
        "cell": cell,
        "direction": direction,
        "color": color_name,
        "shape": shape_name,
        "area": area_px,
        "confidence": conf_pct,
        "fired": fired,
        "time": time_display,
        "pitch": pitch,
        "yaw": yaw,
    }

    if dashboard and hasattr(dashboard, "add_target_result"):
        dashboard.add_target_result(filepath, metadata)

    return filepath, metadata


class VisionPreview:
    """Standalone live camera and color-filter preview; no chassis commands."""

    def __init__(self):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.root = tk.Tk()
        self.root.title("RoboMaster vision test")
        self.root.geometry("1100x760")
        self.root.configure(bg="#171a20")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.robot = self.reader = self.analyzer = None
        self.connected = False
        self.last_frame_id = -1
        self._photos = {}
        self.status = tk.StringVar(value="Disconnected")
        top = tk.Frame(self.root, bg="#171a20")
        top.pack(fill="x", padx=12, pady=8)
        tk.Label(top, text="Live camera · stable detections", fg="white", bg="#171a20",
                 font=("Segoe UI", 14, "bold")).pack(side="left")
        self.connect_btn = ttk.Button(top, text="Connect camera", command=self.connect)
        self.connect_btn.pack(side="right")
        tk.Label(self.root, textvariable=self.status, fg="#c7ced8", bg="#171a20").pack(anchor="w", padx=14)
        self.camera_label = tk.Label(self.root, bg="#08090b")
        self.camera_label.pack(fill="both", expand=True, padx=12, pady=8)
        self.results = tk.Frame(self.root, bg="#222730")
        self.results.pack(fill="x", padx=12, pady=(0, 12))
        tk.Label(self.results, text="Accepted color + shape", fg="white", bg="#222730",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=10, pady=(7, 2))
        self.result_line = tk.Label(self.results, text="—", fg="#dce3eb", bg="#222730",
                                    font=("Segoe UI", 12))
        self.result_line.pack(anchor="w", padx=10, pady=(2, 8))
        self.filter_win = tk.Toplevel(self.root)
        self.filter_win.title("Color filter pass · mask")
        self.filter_win.geometry("520x420")
        self.filter_win.configure(bg="#171a20")
        tk.Label(self.filter_win, text="Pixels passing color ranges + morphology",
                 fg="white", bg="#171a20", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=10, pady=8)
        self.filter_label = tk.Label(self.filter_win, bg="#08090b")
        self.filter_label.pack(fill="both", expand=True, padx=10, pady=6)
        self.filter_status = tk.Label(self.filter_win, text="Waiting for camera", fg="#c7ced8", bg="#171a20")
        self.filter_status.pack(anchor="w", padx=10, pady=6)
        self.root.after(40, self.refresh)

    def connect(self):
        if self.connected:
            return
        self.connect_btn.configure(state="disabled")
        self.status.set("Connecting to RoboMaster…")
        threading.Thread(target=self._connect_worker, daemon=True).start()

    def _connect_worker(self):
        try:
            import robomaster
            robot = robomaster.robot.Robot()
            robot.initialize(conn_type="ap")
            # Only pitch is commanded; chassis and gimbal yaw remain still.
            pitch_state = {"value": None}
            event = threading.Event()
            def angle_callback(msg):
                try:
                    pitch_state["value"] = msg[1]
                    event.set()
                except (IndexError, TypeError):
                    pass
            robot.gimbal.sub_angle(freq=10, callback=angle_callback)
            event.wait(1.5)
            if pitch_state["value"] is not None:
                robot.gimbal.move(pitch=COLOR_SCAN_PITCH - pitch_state["value"], yaw=0,
                                  pitch_speed=COLOR_SCAN_SPEED, yaw_speed=0).wait_for_completed()
            robot.camera.start_video_stream(display=False, resolution="720p")
            reader = CameraFrameReader(robot.camera)
            reader.start()
            analyzer = VisionFrameAnalyzer(reader)
            analyzer.start()
            self.robot, self.reader, self.analyzer = robot, reader, analyzer
            self.connected = True
            self.root.after(0, lambda: self.status.set("Connected · gimbal pitched down · chassis stationary"))
        except Exception as exc:
            self.root.after(0, lambda error=str(exc): self.status.set(f"Connection failed: {error}"))
            self.root.after(0, lambda: self.connect_btn.configure(state="normal"))

    def _show_image(self, widget, frame, key, max_size):
        from PIL import Image, ImageTk
        h, w = frame.shape[:2]
        scale = min(max_size[0] / w, max_size[1] / h)
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        if resized.ndim == 2:
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        widget.configure(image=photo)
        self._photos[key] = photo

    def refresh(self):
        if self.analyzer:
            frame_id, view, detections = self.analyzer.latest()
            filter_id, filtered, raw = self.analyzer.latest_filter()
            if view is not None and frame_id != self.last_frame_id:
                self.last_frame_id = frame_id
                self._show_image(self.camera_label, view, "camera", (1060, 550))
                labels = sorted({(d["color"], d["shape"], d.get("size_px", (0, 0))) for d in detections})
                self.result_line.configure(
                    text="    ".join(f'{c} {s} {size[0]}×{size[1]} px' for c, s, size in labels)
                    if labels else "No stable detections"
                )
                if filtered is not None:
                    self._show_image(self.filter_label, filtered, "filter", (490, 330))
                    self.filter_status.configure(text=f"Raw contours: {len(raw)}  ·  accepted after temporal filter: {len(detections)}")
        self.root.after(40, self.refresh)

    def close(self):
        if self.analyzer:
            self.analyzer.stop()
        if self.reader:
            self.reader.stop()
        if self.robot:
            try:
                self.robot.camera.stop_video_stream()
                self.robot.close()
            except Exception:
                pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    VisionPreview().run()
