from ultralytics import YOLO
import argparse
import cv2
import numpy as np
import torch
import time
from pathlib import Path

from config import CONE_MODEL_PATH, OUTPUTS_DIR, SAMPLES_DIR, ensure_runtime_directories

# Configuration
DEBUG = False
TARGET_FPS = 43
FRAME_SKIP = 0  # Start with no frame skipping
RESIZE_FACTOR = 0.3  # Reduce resolution to 50% (0.5 = half, 0.75 = 75%, etc.)

class ConeDetector:
    def __init__(self, model_path):
        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        half = self.device == "cuda"  # Use half precision for GPU
        
        # Load model with optimizations
        self.model = YOLO(model_path)
        self.model.overrides['conf'] = 0.3    # Confidence threshold
        self.model.overrides['iou'] = 0.4     # IOU threshold
        if half:
            self.model.overrides['half'] = True  # Half precision on CUDA only
        self.model.overrides['verbose'] = False
        self.model.overrides['agnostic_nms'] = True
        
        self.names = self.model.names
        print(f"[INFO] Using Device: {self.device}")
        print(f"[INFO] Target FPS: {TARGET_FPS}")
        print(f"[INFO] Resize Factor: {RESIZE_FACTOR}")
        if self.device == "cuda":
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
            
        # Initialize target FPS as instance variable
        self.target_fps = TARGET_FPS
        self.resize_factor = RESIZE_FACTOR

    def detect_cones(self, frame):
        # Run YOLO inference
        results = self.model(frame)
        
        # Extract bounding boxes and classes
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        
        # Separate left and right cones based on x-position
        left_cones = []
        right_cones = []
        h, w = frame.shape[:2]
        
        for (x1, y1, x2, y2), cls_id, conf in zip(boxes, classes, confidences):
            # Calculate center bottom point
            center_x = (x1 + x2) / 2
            bottom_y = y2
            
            # Classify as left or right based on position
            if center_x < w / 2:
                left_cones.append((int(center_x), int(bottom_y), int(cls_id), float(conf)))
            else:
                right_cones.append((int(center_x), int(bottom_y), int(cls_id), float(conf)))
                
        return left_cones, right_cones, results[0].plot()

    def process_video(self, path, output_dir=OUTPUTS_DIR):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error opening video: {path}")
            return
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate new resolution
        new_w = int(original_w * self.resize_factor)
        new_h = int(original_h * self.resize_factor)
        
        print(f"Original Resolution: {original_w}x{original_h}")
        print(f"Reduced Resolution: {new_w}x{new_h} ({self.resize_factor*100:.0f}%)")
        print(f"Original FPS: {original_fps:.2f}, Total frames: {total_frames}")
        
        # Define output video writer with target FPS (using original resolution for output)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(str(output_dir / 'cone_detection_output.mp4'), fourcc, self.target_fps, (original_w, original_h))
        
        frame_count = 0
        processed_count = 0
        inference_times = []
        frame_times = []
        
        # Warm-up the model
        print("Warming up model...")
        for _ in range(5):
            ret, warmup_frame = cap.read()
            if ret:
                # Resize warmup frame
                warmup_frame_resized = cv2.resize(warmup_frame, (new_w, new_h))
                self.detect_cones(warmup_frame_resized)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        print("Starting processing...")
        print("Press 'Q' to quit, '+' to increase FPS, '-' to decrease FPS")
        print("Press 'R' to decrease resolution, 'F' to increase resolution")
        
        # Adaptive frame skipping to maintain target FPS
        skip_frames = FRAME_SKIP
        last_adjustment = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Skip frames if needed to maintain target FPS
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                continue
                
            processed_count += 1
            
            # Start timing
            frame_start_time = time.time()
            
            # Resize frame for processing (FASTER INFERENCE)
            frame_resized = cv2.resize(frame, (new_w, new_h))
            
            # Detect cones on resized frame
            inference_start = time.time()
            left_cones, right_cones, annotated_frame_resized = self.detect_cones(frame_resized)
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)
            
            # Resize annotated frame back to original size for output
            annotated_frame = cv2.resize(annotated_frame_resized, (original_w, original_h))
            
            # Calculate current FPS
            if len(inference_times) > 10:
                current_fps = 1 / np.mean(inference_times[-10:])
            else:
                current_fps = 1 / inference_time if inference_time > 0 else 0
            
            # Adaptive frame skipping
            if frame_count - last_adjustment > 30:  # Adjust every 30 frames
                if current_fps < self.target_fps * 0.9:  # Below target
                    skip_frames = min(skip_frames + 1, 5)
                    last_adjustment = frame_count
                    print(f"FPS {current_fps:.1f} below target, increasing skip to {skip_frames}")
                elif current_fps > self.target_fps * 1.1 and skip_frames > 0:  # Above target
                    skip_frames = max(skip_frames - 1, 0)
                    last_adjustment = frame_count
                    print(f"FPS {current_fps:.1f} above target, decreasing skip to {skip_frames}")
            
            # Draw performance info
            cv2.putText(annotated_frame, f"Target FPS: {self.target_fps}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Current FPS: {current_fps:.1f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if current_fps >= self.target_fps else (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Inference: {inference_time*1000:.1f}ms", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Frame skip: {skip_frames}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Resolution: {new_w}x{new_h}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Left: {len(left_cones)}", (10, 180), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.putText(annotated_frame, f"Right: {len(right_cones)}", (10, 210), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Draw center line
            cv2.line(annotated_frame, (original_w//2, 0), (original_w//2, original_h), (255, 255, 255), 1)
            
            # Write to output video with target FPS timing
            out.write(annotated_frame)
            
            frame_time = time.time() - frame_start_time
            frame_times.append(frame_time)
            
            # Display progress
            if processed_count % 30 == 0:
                avg_fps = 1 / np.mean(inference_times[-30:]) if inference_times else 0
                print(f"Frame {frame_count}/{total_frames} - FPS: {avg_fps:.1f} - Res: {new_w}x{new_h} - Skip: {skip_frames}")
            
            if DEBUG:
                cv2.imshow(f'Cone Detection - Target: {self.target_fps} FPS', annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('+'):  # Increase target FPS
                    self.target_fps = min(self.target_fps + 5, 60)
                    print(f"Target FPS increased to: {self.target_fps}")
                elif key == ord('-'):  # Decrease target FPS
                    self.target_fps = max(self.target_fps - 5, 10)
                    print(f"Target FPS decreased to: {self.target_fps}")
                elif key == ord('r'):  # Decrease resolution
                    self.resize_factor = max(self.resize_factor - 0.1, 0.3)
                    new_w = int(original_w * self.resize_factor)
                    new_h = int(original_h * self.resize_factor)
                    print(f"Resolution decreased to: {new_w}x{new_h} ({self.resize_factor*100:.0f}%)")
                elif key == ord('f'):  # Increase resolution
                    self.resize_factor = min(self.resize_factor + 0.1, 1.0)
                    new_w = int(original_w * self.resize_factor)
                    new_h = int(original_h * self.resize_factor)
                    print(f"Resolution increased to: {new_w}x{new_h} ({self.resize_factor*100:.0f}%)")
        
        # Calculate final performance
        if inference_times:
            avg_inference_time = np.mean(inference_times)
            avg_fps = 1 / avg_inference_time
            achieved_fps = processed_count / np.sum(frame_times) if frame_times else 0
            
            print(f"\n{'='*50}")
            print(f"PERFORMANCE SUMMARY")
            print(f"{'='*50}")
            print(f"Target FPS: {self.target_fps}")
            print(f"Achieved FPS: {achieved_fps:.1f}")
            print(f"Average Inference FPS: {avg_fps:.1f}")
            print(f"Final Resolution: {new_w}x{new_h} ({self.resize_factor*100:.0f}%)")
            print(f"Frames processed: {processed_count}/{frame_count}")
            print(f"Frame skip used: {skip_frames}")
            print(f"Average inference time: {avg_inference_time*1000:.1f}ms")
            print(f"Device: {self.device}")
            print(f"{'='*50}")
            
            if abs(achieved_fps - self.target_fps) <= 2:
                print("🎯 TARGET FPS ACHIEVED!")
            else:
                print("⚠️  Target FPS not reached. Try:")
                print("   - Reducing resolution further (press 'R')")
                print("   - Using a lighter model")
                print("   - Increasing frame skipping")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Run YOLO cone detection on a recorded video.")
    parser.add_argument("--video", type=Path, default=SAMPLES_DIR / "cone_course_01.mp4")
    parser.add_argument("--model", type=Path, default=CONE_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--debug", action="store_true", help="Show a live OpenCV window.")
    args = parser.parse_args()
    if not args.video.exists():
        parser.error(f"Video not found: {args.video}")
    if not args.model.exists():
        parser.error(f"Model not found: {args.model}")
    global DEBUG
    DEBUG = args.debug
    ensure_runtime_directories()
    ConeDetector(str(args.model)).process_video(str(args.video), args.output_dir)


if __name__ == '__main__':
    main()
