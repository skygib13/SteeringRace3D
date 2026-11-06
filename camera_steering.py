"""Webcam control.  All OpenCV/MediaPipe work runs in a worker thread.

The Ursina main thread never calls cv2.imshow / reads VideoCapture.
"""
from __future__ import annotations

import math
import platform
import threading
import time

from steering_math import angle_between_hands, steering_from_angle


class CameraSteering:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker = None
        self._steering = 0.0
        self._angle = 0.0
        self._zero = 0.0
        self._hands = 0
        self._preview_rgb = None
        self._preview_serial = 0
        self._status = "STARTING..."
        self._last_detected = 0.0
        self._last_calibration = "NOT CALIBRATED"
        self._active = False

    def start(self):
        if self._worker is not None:
            return
        self._worker = threading.Thread(target=self._loop, name="camera-pose", daemon=True)
        self._worker.start()

    def snapshot(self):
        with self._lock:
            return {
                "steer": self._steering,
                "angle": self._angle,
                "hands": self._hands,
                "preview": self._preview_rgb,
                "serial": self._preview_serial,
                "status": self._status,
                "active": self._active,
                "calibration": self._last_calibration,
                "detected_recently": time.monotonic() - self._last_detected < 0.45,
            }

    def calibrate(self):
        with self._lock:
            if self._hands != 2:
                self._last_calibration = "SHOW BOTH HANDS, THEN PRESS C"
                return False
            self._zero = self._angle
            self._last_calibration = "CENTER CALIBRATED"
            self._steering = 0.0
            return True

    def close(self):
        self._stop.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=1.8)

    def _loop(self):
        # Delayed imports mean the game can still open in keyboard-only mode
        # if a camera-related package is missing or fails to initialize.
        try:
            import cv2
            import mediapipe as mp
        except Exception as exc:
            with self._lock:
                self._status = "CAMERA PACKAGES NOT READY: " + str(exc)[:60]
            return

        cap = None
        try:
            if platform.system() == "Windows":
                cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                with self._lock:
                    self._status = "NO CAMERA (KEYBOARD AVAILABLE)"
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not hasattr(mp, "solutions"):
                raise RuntimeError("MediaPipe Solutions unavailable. Install mediapipe==0.10.21")
            hands_module = mp.solutions.hands
            with hands_module.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.58,
                min_tracking_confidence=0.50,
            ) as hands:
                with self._lock:
                    self._active = True
                    self._status = "SHOW BOTH HANDS"

                failed_frames = 0
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        failed_frames += 1
                        if failed_frames > 35:
                            with self._lock:
                                self._status = "CAMERA LOST - USE A/D"
                            break
                        time.sleep(0.025)
                        continue
                    failed_frames = 0
                    frame = cv2.flip(frame, 1)
                    h, w = frame.shape[:2]
                    input_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    input_rgb.flags.writeable = False
                    result = hands.process(input_rgb)
                    hand_points = []
                    if result.multi_hand_landmarks:
                        for item in result.multi_hand_landmarks[:2]:
                            # Palm center is steadier than just the wrist point.
                            wrist = item.landmark[0]
                            middle = item.landmark[9]
                            p = ((wrist.x + middle.x) / 2, (wrist.y + middle.y) / 2)
                            hand_points.append(p)

                    detected = False
                    angle = 0.0
                    if len(hand_points) == 2:
                        hand_points.sort(key=lambda p: p[0])
                        left, right = hand_points
                        # Hands should be apart like at 9 and 3 o'clock.
                        if right[0] - left[0] >= 0.13:
                            angle = angle_between_hands(left, right)
                            detected = True

                    now = time.monotonic()
                    with self._lock:
                        self._hands = 2 if detected else (1 if hand_points else 0)
                        if detected:
                            self._angle = angle
                            raw = steering_from_angle(angle, self._zero)
                            self._steering = 0.27 * raw + 0.73 * self._steering
                            self._last_detected = now
                            self._status = "CAMERA STEERING ACTIVE"
                        else:
                            self._steering *= 0.80
                            self._status = "SPREAD OUT TWO HANDS" if len(hand_points) == 2 else "SHOW TWO HANDS"

                    # Tiny in-game camera preview, no extra OpenCV window.
                    preview = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
                    for idx, (px, py) in enumerate(hand_points):
                        cv2.circle(preview, (int(px * 320), int(py * 180)), 7,
                                   (80, 230, 255), 2, cv2.LINE_AA)
                    if detected:
                        pa, pb = hand_points
                        cv2.line(preview, (int(pa[0] * 320), int(pa[1] * 180)),
                                 (int(pb[0] * 320), int(pb[1] * 180)),
                                 (0, 220, 140), 3, cv2.LINE_AA)
                        cv2.putText(preview, f"{math.degrees(angle):+.0f} deg",
                                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                    (255, 255, 255), 2, cv2.LINE_AA)
                    preview_rgb = cv2.flip(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), 0)
                    # Panda3D texture upload occurs only from the main thread.
                    with self._lock:
                        self._preview_rgb = preview_rgb.tobytes()
                        self._preview_serial += 1
        except Exception as exc:
            with self._lock:
                self._status = "CAMERA ERROR: " + str(exc)[:70]
        finally:
            if cap is not None:
                cap.release()
            with self._lock:
                self._active = False
