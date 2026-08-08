from ultralytics import YOLO
import argparse
import cv2
import numpy as np
import torch
import time
import math
from pathlib import Path

from config import CONE_MODEL_PATH, OUTPUTS_DIR, SAMPLES_DIR, ensure_runtime_directories

# Configuration
DEBUG = False
FRAMEDROP = 2
PIXELS_PER_METER = 100
REAL_WORLD_WIDTH_M = 3.5
REAL_WORLD_LENGTH_M = 10.0

# Car and safety parameters
CAR_WIDTH = 0.4  # meters
CAR_LENGTH = 0.6  # meters
SAFETY_MARGIN = 0.3  # meters
MIN_PATH_WIDTH = CAR_WIDTH + 2 * SAFETY_MARGIN  # Minimum required path width

# Convert to pixels
CAR_WIDTH_PX = int(CAR_WIDTH * PIXELS_PER_METER)
CAR_LENGTH_PX = int(CAR_LENGTH * PIXELS_PER_METER)
MIN_PATH_WIDTH_PX = int(MIN_PATH_WIDTH * PIXELS_PER_METER)

# Pure Pursuit parameters
LOOKAHEAD_DISTANCE = 50  # pixels
CURVATURE_THRESHOLD = math.radians(40)  # Sharp turn threshold

NORM_PERSPECTIVE_SRC = np.float32([
    (0.15, 0.25),  # top-left
    (0.65, 0.25),  # top-right
    (0.9, 1.0),    # bottom-right
    (0.1, 1.0)     # bottom-left
])

class PurePursuit:
    def __init__(self, lookahead_distance):
        self.lookahead_distance = lookahead_distance
    
    def smooth_path_with_pure_pursuit(self, path, current_pos):
        """Apply Pure Pursuit smoothing to sharp turns in the path"""
        if len(path) < 3:
            return path
        
        smoothed_path = [path[0]]  # Start with first point
        
        for i in range(1, len(path) - 1):
            prev_point = path[i - 1]
            curr_point = path[i]
            next_point = path[i + 1]
            
            # Calculate curvature at this segment
            curvature = self.calculate_curvature(prev_point, curr_point, next_point)
            
            # If sharp turn detected, apply Pure Pursuit smoothing
            if curvature > CURVATURE_THRESHOLD:
                # Use lookahead point to smooth the turn
                lookahead_point = self.find_lookahead_point(path, curr_point)
                if lookahead_point:
                    # Create a smoother curve towards the lookahead point
                    smoothed_point = self.interpolate_points(curr_point, lookahead_point, 0.5)
                    smoothed_path.append(smoothed_point)
                else:
                    smoothed_path.append(curr_point)
            else:
                smoothed_path.append(curr_point)
        
        smoothed_path.append(path[-1])  # End with last point
        return smoothed_path
    
    def calculate_curvature(self, p1, p2, p3):
        """Calculate curvature between three points"""
        # Vectors between points
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        # Calculate angle between vectors
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag_v1 * mag_v2 == 0:
            return 0
            
        cos_angle = dot_product / (mag_v1 * mag_v2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to avoid numerical issues
        angle = math.acos(cos_angle)
        
        return angle
    
    def find_lookahead_point(self, path, current_point):
        """Find point ahead on the path for Pure Pursuit"""
        for point in path:
            if point[1] > current_point[1]:  # Look for points ahead (higher Y)
                distance = math.sqrt((point[0] - current_point[0])**2 + 
                                   (point[1] - current_point[1])**2)
                if distance >= self.lookahead_distance:
                    return point
        return path[-1] if path else None
    
    def interpolate_points(self, p1, p2, factor=0.5):
        """Interpolate between two points"""
        return (
            int(p1[0] + (p2[0] - p1[0]) * factor),
            int(p1[1] + (p2[1] - p1[1]) * factor)
        )

class ObstacleAvoidance:
    def __init__(self, car_width_px, min_path_width_px, safety_margin_px):
        self.car_width_px = car_width_px
        self.min_path_width_px = min_path_width_px
        self.safety_margin_px = safety_margin_px
    
    def detect_obstacles_in_path(self, left_cones, right_cones, midpoint_path):
        """Detect obstacles (cones) that are between left and right boundaries"""
        obstacles = []
        
        if not left_cones or not right_cones or not midpoint_path:
            return obstacles
        
        # Create a list of all cones
        all_cones = left_cones + right_cones
        
        for path_point in midpoint_path:
            # Find closest left and right cones at similar Y position
            left_cones_near = [cone for cone in left_cones if abs(cone[1] - path_point[1]) < 50]
            right_cones_near = [cone for cone in right_cones if abs(cone[1] - path_point[1]) < 50]
            
            if not left_cones_near or not right_cones_near:
                continue
            
            left_bound = max(cone[0] for cone in left_cones_near)
            right_bound = min(cone[0] for cone in right_cones_near)
            
            # Check if path width is sufficient
            path_width = right_bound - left_bound
            if path_width < self.min_path_width_px:
                # Find cones that are causing the narrow path
                for cone in all_cones:
                    if (abs(cone[1] - path_point[1]) < 30 and 
                        left_bound < cone[0] < right_bound):
                        obstacles.append({
                            'position': cone,
                            'path_point': path_point,
                            'path_width': path_width,
                            'left_bound': left_bound,
                            'right_bound': right_bound
                        })
        
        return obstacles
    
    def calculate_safe_path_around_obstacle(self, obstacle, left_cones, right_cones):
        """Calculate a safe path around an obstacle"""
        obstacle_pos = obstacle['position']
        left_bound = obstacle['left_bound']
        right_bound = obstacle['right_bound']
        
        # Calculate available space on left and right sides
        left_space = obstacle_pos[0] - left_bound
        right_space = right_bound - obstacle_pos[0]
        
        # Choose the side with more space
        if left_space >= right_space and left_space >= self.min_path_width_px:
            # Go left of the obstacle
            safe_x = obstacle_pos[0] - self.safety_margin_px - self.car_width_px // 2
            return safe_x
        elif right_space >= self.min_path_width_px:
            # Go right of the obstacle
            safe_x = obstacle_pos[0] + self.safety_margin_px + self.car_width_px // 2
            return safe_x
        else:
            # Not enough space on either side, can't safely pass
            return None
    
    def adjust_path_for_obstacles(self, midpoint_path, left_cones, right_cones):
        """Adjust the path to avoid obstacles"""
        if not midpoint_path:
            return midpoint_path
        
        obstacles = self.detect_obstacles_in_path(left_cones, right_cones, midpoint_path)
        
        if not obstacles:
            return midpoint_path
        
        adjusted_path = midpoint_path.copy()
        
        for obstacle in obstacles:
            obstacle_y = obstacle['position'][1]
            safe_x = self.calculate_safe_path_around_obstacle(obstacle, left_cones, right_cones)
            
            if safe_x is not None:
                # Find the closest path point to the obstacle
                closest_idx = min(range(len(adjusted_path)), 
                                key=lambda i: abs(adjusted_path[i][1] - obstacle_y))
                
                # Adjust points around the obstacle
                for i in range(max(0, closest_idx-2), min(len(adjusted_path), closest_idx+3)):
                    # Smooth transition to avoid the obstacle
                    distance_to_obstacle = abs(adjusted_path[i][1] - obstacle_y)
                    if distance_to_obstacle < 80:  # Adjust points within this range
                        blend_factor = 1 - (distance_to_obstacle / 80)
                        new_x = int(adjusted_path[i][0] * (1 - blend_factor) + safe_x * blend_factor)
                        adjusted_path[i] = (new_x, adjusted_path[i][1])
        
        return adjusted_path

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
        self.pure_pursuit = PurePursuit(LOOKAHEAD_DISTANCE)
        self.obstacle_avoidance = ObstacleAvoidance(CAR_WIDTH_PX, MIN_PATH_WIDTH_PX, 
                                                   int(SAFETY_MARGIN * PIXELS_PER_METER))

    def compute_perspective(self, w, h):
        src = np.float32([(x*w, y*h) for (x, y) in NORM_PERSPECTIVE_SRC])
        self.top_w = int(REAL_WORLD_WIDTH_M * PIXELS_PER_METER)
        self.top_h = int(REAL_WORLD_LENGTH_M * PIXELS_PER_METER)
        dst = np.float32([[0,0], [self.top_w-1,0], [self.top_w-1,self.top_h-1], [0,self.top_h-1]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)
        if DEBUG:
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
        res = self.model(frame)
        boxes = res[0].boxes.xyxy.cpu().numpy()
        classes = res[0].boxes.cls.cpu().numpy().astype(int)

        left, right, middle = [], [], []
        for (x1, y1, x2, y2), c in zip(boxes, classes):
            midx, boty = (x1 + x2) / 2, y2
            try:
                pt = self.to_top([(midx, boty)])[0]
            except Exception:
                continue

            if pt[0] < self.top_w // 3:
                left.append(pt)
            elif pt[0] > 2 * self.top_w // 3:
                right.append(pt)
            else:
                middle.append(pt)  # Cones in the middle

        return left, right, middle, res[0].plot()

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

    def get_start_end_points(self, left, right):
        if not left or not right:
            return None, None
        
        start_left = max(left, key=lambda p: p[1]) if left else None
        start_right = max(right, key=lambda p: p[1]) if right else None
        end_left = min(left, key=lambda p: p[1]) if left else None
        end_right = min(right, key=lambda p: p[1]) if right else None
        
        if start_left and start_right:
            start_point = (
                int((start_left[0] + start_right[0]) / 2),
                int((start_left[1] + start_right[1]) / 2)
            )
        else:
            start_point = None
        
        if end_left and end_right:
            end_point = (
                int((end_left[0] + end_right[0]) / 2),
                int((end_left[1] + end_right[1]) / 2)
            )
        else:
            end_point = None
        
        return start_point, end_point

    def generate_midpoint_path(self, left, right):
        """Generate path using midpoint algorithm between left and right cones"""
        path = []
        
        if not left or not right:
            return path
        
        # Sort cones by Y coordinate
        left_sorted = sorted(left, key=lambda p: p[1])
        right_sorted = sorted(right, key=lambda p: p[1])
        
        # Create midpoint path by pairing cones at similar Y positions
        for left_cone in left_sorted:
            # Find closest right cone at similar Y position
            closest_right = min(right_sorted, key=lambda p: abs(p[1] - left_cone[1]))
            
            # Calculate midpoint
            mid_x = int((left_cone[0] + closest_right[0]) / 2)
            mid_y = int((left_cone[1] + closest_right[1]) / 2)
            
            path.append((mid_x, mid_y))
        
        # Sort path by Y coordinate
        path = sorted(path, key=lambda p: p[1])
        
        return path

    def detect_sharp_turns(self, path):
        """Detect sharp turns in the path and return their positions"""
        sharp_turns = []
        
        if len(path) < 3:
            return sharp_turns
        
        for i in range(1, len(path) - 1):
            p1, p2, p3 = path[i-1], path[i], path[i+1]
            
            # Calculate curvature
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag_v1 * mag_v2 == 0:
                continue
                
            cos_angle = dot_product / (mag_v1 * mag_v2)
            cos_angle = max(-1, min(1, cos_angle))
            angle = math.acos(cos_angle)
            
            if angle > CURVATURE_THRESHOLD:
                sharp_turns.append((i, p2, math.degrees(angle)))  # index, point, angle in degrees
        
        return sharp_turns

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
        out_orig = cv2.VideoWriter(str(output_dir / 'orig_with_lane.mp4'), fourcc, fps, (w, h))
        out_top = cv2.VideoWriter(str(output_dir / 'top_view.mp4'), fourcc, fps, (self.top_w, self.top_h))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        prev_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            if idx % FRAMEDROP != 0:
                continue

            current_time = time.time()
            processing_fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time

            left, right, middle, ann = self.detect_cones(frame)
            left = self.sort_filter(left)
            right = self.sort_filter(right)
            middle = self.sort_filter(middle)

            start_point, end_point = self.get_start_end_points(left, right)

            # Generate midpoint path
            midpoint_path = self.generate_midpoint_path(left, right)
            
            # Check for obstacles and adjust path
            obstacles = self.obstacle_avoidance.detect_obstacles_in_path(left, right, midpoint_path)
            safe_path = self.obstacle_avoidance.adjust_path_for_obstacles(midpoint_path, left, right)
            
            # Apply Pure Pursuit smoothing for sharp turns
            smoothed_path = safe_path
            sharp_turns = []
            
            if safe_path and start_point:
                smoothed_path = self.pure_pursuit.smooth_path_with_pure_pursuit(safe_path, start_point)
                sharp_turns = self.detect_sharp_turns(safe_path)

            # Top-down view
            top = np.zeros((self.top_h, self.top_w, 3), dtype=np.uint8)
            
            # Draw cones with different colors
            for cone in left:
                cv2.circle(top, cone, 5, (0, 165, 255), -1)  # Orange for left
            for cone in right:
                cv2.circle(top, cone, 5, (255, 0, 0), -1)    # Blue for right
            for cone in middle:
                cv2.circle(top, cone, 7, (0, 0, 255), -1)    # Red for middle obstacles

            # Draw start and end points
            if start_point:
                cv2.circle(top, start_point, 8, (0, 255, 0), -1)
                cv2.putText(top, "Start", (start_point[0] + 10, start_point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if end_point:
                cv2.circle(top, end_point, 8, (0, 0, 255), -1)
                cv2.putText(top, "End", (end_point[0] + 10, end_point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Draw paths
            if midpoint_path and len(midpoint_path) >= 2:
                # Original midpoint path (blue)
                cv2.polylines(top, [np.array(midpoint_path, dtype=int)], False, (255, 0, 255), 2)
                
            if safe_path and len(safe_path) >= 2 and safe_path != midpoint_path:
                # Safe path avoiding obstacles (yellow)
                cv2.polylines(top, [np.array(safe_path, dtype=int)], False, (0, 255, 255), 3)
                
            if smoothed_path and len(smoothed_path) >= 2:
                # Final smoothed path with Pure Pursuit (green)
                cv2.polylines(top, [np.array(smoothed_path, dtype=int)], False, (0, 255, 0), 3)

            # Highlight obstacles and narrow sections
            for obstacle in obstacles:
                obstacle_pos = obstacle['position']
                cv2.circle(top, obstacle_pos, 8, (0, 0, 255), -1)
                cv2.putText(top, f"Width: {obstacle['path_width']}px", 
                           (obstacle_pos[0] + 10, obstacle_pos[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Highlight sharp turns
            for turn_idx, turn_point, angle in sharp_turns:
                cv2.circle(top, turn_point, 10, (0, 255, 255), -1)
                cv2.putText(top, f"{angle:.1f}°", (turn_point[0] + 15, turn_point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Fill polygon
            if len(left) >= 2 and len(right) >= 2:
                poly = np.array(left + right[::-1], dtype=np.int32)
                ov = top.copy()
                cv2.fillPoly(ov, [poly], (100,100,255))
                cv2.addWeighted(ov, 0.3, top, 0.7, 0, top)

            # Draw car dimensions for reference
            if start_point:
                car_rect = np.array([
                    [start_point[0] - CAR_WIDTH_PX//2, start_point[1] - CAR_LENGTH_PX//2],
                    [start_point[0] + CAR_WIDTH_PX//2, start_point[1] - CAR_LENGTH_PX//2],
                    [start_point[0] + CAR_WIDTH_PX//2, start_point[1] + CAR_LENGTH_PX//2],
                    [start_point[0] - CAR_WIDTH_PX//2, start_point[1] + CAR_LENGTH_PX//2]
                ], dtype=np.int32)
                cv2.polylines(top, [car_rect], True, (255, 255, 255), 2)

            # Original view with overlay
            orig = cv2.cvtColor(ann, cv2.COLOR_RGB2BGR)

            if smoothed_path and len(smoothed_path) >= 2:
                orig_path = self.to_orig(smoothed_path)
                cv2.polylines(orig, [np.array(orig_path, dtype=int)], False, (0, 255, 0), 3)

            if len(left) >= 2 and len(right) >= 2:
                orig_poly = self.to_orig(poly.tolist())
                cv2.polylines(orig, [np.array(orig_poly[:len(left)], dtype=int)], False, (0,165,255), 2)
                cv2.polylines(orig, [np.array(orig_poly[len(left):], dtype=int)], False, (255,0,0), 2)
                cv2.fillPoly(orig, [np.array(orig_poly, dtype=int)], (100,100,255))

            # Draw start and end points on original view
            if start_point:
                orig_start = self.to_orig([start_point])[0]
                cv2.circle(orig, tuple(map(int, orig_start)), 8, (0, 255, 0), -1)
            
            if end_point:
                orig_end = self.to_orig([end_point])[0]
                cv2.circle(orig, tuple(map(int, orig_end)), 8, (0, 0, 255), -1)

            # Display info
            fps_text = f"FPS: {processing_fps:.2f}"
            cv2.putText(orig, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(top, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display obstacle and turn info
            info_y = 60
            if obstacles:
                obstacle_text = f"Obstacles: {len(obstacles)}"
                cv2.putText(top, obstacle_text, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                info_y += 30
            
            if sharp_turns:
                turn_info = f"Sharp turns: {len(sharp_turns)}"
                cv2.putText(top, turn_info, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                info_y += 30
                
            if safe_path != midpoint_path:
                cv2.putText(top, "Obstacle Avoidance Active", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                info_y += 30
                
            if smoothed_path != safe_path:
                cv2.putText(top, "Pure Pursuit Active", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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

def main():
    parser = argparse.ArgumentParser(description="Plan a cone-course path from a recorded video.")
    parser.add_argument("--video", type=Path, default=SAMPLES_DIR / "cone_course_01.mp4")
    parser.add_argument("--model", type=Path, default=CONE_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--debug", action="store_true", help="Show live OpenCV windows.")
    args = parser.parse_args()
    if not args.video.exists():
        parser.error(f"Video not found: {args.video}")
    if not args.model.exists():
        parser.error(f"Model not found: {args.model}")
    global DEBUG
    DEBUG = args.debug
    ensure_runtime_directories()
    ConeTopView(str(args.model)).process_video(str(args.video), args.output_dir)


if __name__ == '__main__':
    main()
