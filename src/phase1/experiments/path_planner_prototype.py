from ultralytics import YOLO
import argparse
import cv2
import numpy as np
import torch
import time  # Import time module for FPS calculation
from pathlib import Path

from config import CONE_MODEL_PATH, OUTPUTS_DIR, SAMPLES_DIR

# Configuration
DEBUG = True
FRAMEDROP = 2               # Process every Nth frame
PIXELS_PER_METER = 100.0
REAL_WORLD_WIDTH_M = 3.5     # Lane width in meters
REAL_WORLD_LENGTH_M = 10.0   # Visible stretch ahead in meters

# Normalized perspective source points (percent of width/height)
NORM_PERSPECTIVE_SRC = np.float32([
    (0.15, 0.25),  # top-left
    (0.65, 0.25),  # top-right
    (0.9, 1.0),    # bottom-right
    (0.1, 1.0)     # bottom-left
])

class ConeTopView:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.names = self.model.names
        if DEBUG:
            print(f"[DEBUG] Using Device: {self.device}")
            print(f"[DEBUG] Class names: {self.names}")
        self.M = None
        self.M_inv = None
        self.top_w = None
        self.top_h = None
        self.orig_w = None
        self.orig_h = None

    def compute_perspective(self, w, h):
        self.orig_w, self.orig_h = w, h
        
        # Calculate source points based on original resolution
        src = np.float32([(x*w, y*h) for (x, y) in NORM_PERSPECTIVE_SRC])
        
        # Calculate destination points for top-down view
        self.top_w = int(REAL_WORLD_WIDTH_M * PIXELS_PER_METER)
        self.top_h = int(REAL_WORLD_LENGTH_M * PIXELS_PER_METER)
        dst = np.float32([[0,0], [self.top_w-1,0], [self.top_w-1,self.top_h-1], [0,self.top_h-1]])
        
        # Compute perspective transformation matrices
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)
        
        if DEBUG:
            print(f"[DEBUG] Original resolution: {w}x{h}")
            print(f"[DEBUG] Perspective src: {src}")
            print(f"[DEBUG] Perspective dst: {dst}")

    def to_top(self, pts):
        arr = np.array(pts, dtype=np.float32).reshape(-1,1,2)
        trans = cv2.perspectiveTransform(arr, self.M)
        return [tuple(pt) for pt in trans.reshape(-1,2).astype(int)]

    def to_orig(self, pts):
        arr = np.array(pts, dtype=np.float32).reshape(-1,1,2)
        trans = cv2.perspectiveTransform(arr, self.M_inv)
        return [tuple(pt) for pt in trans.reshape(-1,2).astype(int)]

    def detect_cones(self, frame):
        # Resize frame to 416x416 for processing
        processing_frame = cv2.resize(frame, (416, 416))
        
        # Run YOLO detection
        res = self.model(processing_frame)
        boxes = res[0].boxes.xyxy.cpu().numpy()
        classes = res[0].boxes.cls.cpu().numpy().astype(int)

        left, right = [], []
        for (x1, y1, x2, y2), c in zip(boxes, classes):
            # Scale coordinates back to original resolution
            scale_x = self.orig_w / 416
            scale_y = self.orig_h / 416
            midx, boty = ((x1 + x2) / 2) * scale_x, y2 * scale_y
            
            try:
                pt = self.to_top([(midx, boty)])[0]
            except Exception:
                continue

            # Assign cone to left/right based on X position
            if pt[0] < self.top_w // 2:
                left.append(pt)
            else:
                right.append(pt)

        # Create annotation on original frame
        annotated_frame = res[0].plot()
        annotated_frame = cv2.resize(annotated_frame, (self.orig_w, self.orig_h))
        
        return left, right, annotated_frame

    def sort_filter(self, pts):
        if not pts:
            return []
        pts = sorted(pts, key=lambda p: p[1])
        if len(pts) > 2:
            ys = np.array([p[1] for p in pts])
            med = np.median(ys)
            mad = np.median(np.abs(ys - med))
            pts = [p for p in pts if abs(p[1] - med) < 3 * mad]
        return pts

    def process_video(self, path, output_dir=OUTPUTS_DIR):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error opening video: {path}")
            return
        ret, frame = cap.read()
        if not ret:
            print("Failed to read video.")
            return
        h, w = frame.shape[:2]
        self.compute_perspective(w, h)

        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_orig = cv2.VideoWriter(str(output_dir / 'prototype_orig_with_lane.mp4'), fourcc, fps, (w, h))
        out_top = cv2.VideoWriter(str(output_dir / 'prototype_top_view.mp4'), fourcc, fps, (self.top_w, self.top_h))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        prev_time = time.time()  # Initialize time for FPS calculation
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            if idx % FRAMEDROP != 0:
                continue

            # Calculate FPS
            current_time = time.time()
            processing_fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time

            left, right, ann = self.detect_cones(frame)
            left = self.sort_filter(left)
            right = self.sort_filter(right)

            # Top-down view
            top = np.zeros((self.top_h, self.top_w, 3), dtype=np.uint8)
            if len(left) >= 2:
                cv2.polylines(top, [np.array(left, dtype=int)], False, (0,165,255), 2)  # orange-ish for left
            if len(right) >= 2:
                cv2.polylines(top, [np.array(right, dtype=int)], False, (255,0,0), 2)    # blue for right

            # --- Compute center path between left and right cones ---
            center_path = []
            if left and right:
                for lx, ly in left:
                    # Find closest right cone with similar y
                    closest = min(right, key=lambda p: abs(p[1] - ly))
                    cx = int((lx + closest[0]) / 2)
                    cy = int((ly + closest[1]) / 2)
                    center_path.append((cx, cy))

                if len(center_path) >= 2:
                    cv2.polylines(top, [np.array(center_path, dtype=int)], False, (0,255,0), 2)  # green path

            # Fill polygon if both sides detected
            if len(left) >= 2 and len(right) >= 2:
                poly = np.array(left + right[::-1], dtype=np.int32)
                ov = top.copy()
                cv2.fillPoly(ov, [poly], (100,100,255))
                cv2.addWeighted(ov, 0.3, top, 0.7, 0, top)

            # Original view with overlay
            orig = cv2.cvtColor(ann, cv2.COLOR_RGB2BGR)

            if len(center_path) >= 2:
                orig_center = self.to_orig(center_path)
                cv2.polylines(orig, [np.array(orig_center, dtype=int)], False, (0,255,0), 2)  # green path

            if len(left) >= 2 and len(right) >= 2:
                orig_poly = self.to_orig(poly.tolist())
                cv2.polylines(orig, [np.array(orig_poly[:len(left)], dtype=int)], False, (0,165,255), 2)
                cv2.polylines(orig, [np.array(orig_poly[len(left):], dtype=int)], False, (255,0,0), 2)
                cv2.fillPoly(orig, [np.array(orig_poly, dtype=int)], (100,100,255))

            # Display FPS on both views
            fps_text = f"FPS: {processing_fps:.2f}"
            cv2.putText(orig, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(top, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            out_top.write(top)
            out_orig.write(orig)

            if DEBUG:
                cv2.imshow('Top-View', top)
                cv2.imshow('Original+Lane', orig)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        out_orig.release()
        out_top.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the retained Phase 1 path-planning prototype.")
    parser.add_argument("--video", type=Path, default=SAMPLES_DIR / "cone_course_01.mp4")
    parser.add_argument("--model", type=Path, default=CONE_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    args = parser.parse_args()
    ConeTopView(str(args.model)).process_video(str(args.video), args.output_dir)
