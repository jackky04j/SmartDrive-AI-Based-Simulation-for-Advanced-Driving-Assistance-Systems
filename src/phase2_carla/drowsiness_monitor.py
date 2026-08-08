import time
import cv2
import dlib
import numpy as np
from imutils.video import VideoStream
from imutils import face_utils
from pathlib import Path

from .config import (
    SHAPE_PREDICTOR_PATH,
    EYE_AR_THRESH,
    MOUTH_AR_THRESH,
    EYE_AR_CONSEC_FRAMES,
    HEAD_TILT_THRESH,
)
from .drowsiness_state import DrowsinessState
from .drowsiness.EAR import eye_aspect_ratio
from .drowsiness.MAR import mouth_aspect_ratio
from .drowsiness.HeadPose import getHeadTiltAndCoords


def drowsiness_thread(state: DrowsinessState, camera_index: int = 0):
    predictor_path = Path(SHAPE_PREDICTOR_PATH)

    if not predictor_path.exists():
        print(f"[Drowsiness] ERROR: Shape predictor not found at:\n  {predictor_path}")
        state.running = False
        return

    print(f"[Drowsiness] Loading facial landmark predictor from:\n  {predictor_path}")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(predictor_path))

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = (49, 68)

    image_points = np.array([
        (359, 391), (399, 561), (337, 297),
        (513, 301), (345, 465), (453, 469)
    ], dtype="double")

    vs = VideoStream(src=camera_index).start()
    time.sleep(2.0)
    print("[Drowsiness] Camera ready.")

    eye_counter = 0

    try:
        while state.running:
            frame = vs.read()
            if frame is None:
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (640, 360))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = gray.shape
            rects = detector(gray, 0)

            face_detected = len(rects) > 0
            ear = 1.0
            mar = 0.0
            tilt = 0.0
            eyes_closed = False
            yawning = False
            head_tilted = False

            for rect in rects:
                shape = predictor(gray, rect)
                shape_np = face_utils.shape_to_np(shape)

                leftEye = shape_np[lStart:lEnd]
                rightEye = shape_np[rStart:rEnd]
                ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

                mouth = shape_np[mStart:mEnd]
                mar = mouth_aspect_ratio(mouth)

                idx_map = {33: 0, 8: 1, 36: 2, 45: 3, 48: 4, 54: 5}
                for i, (x, y) in enumerate(shape_np):
                    if i in idx_map:
                        image_points[idx_map[i]] = np.array([x, y], dtype='double')

                try:
                    (head_tilt_degree, start_pt, end_pt, end_pt_alt) = \
                        getHeadTiltAndCoords(size, image_points, 360)
                    tilt = float(head_tilt_degree[0]) if head_tilt_degree is not None else 0.0
                except Exception:
                    tilt = 0.0

                if ear < EYE_AR_THRESH:
                    eye_counter += 1
                else:
                    eye_counter = 0

                eyes_closed = eye_counter >= EYE_AR_CONSEC_FRAMES
                yawning = mar > MOUTH_AR_THRESH
                head_tilted = tilt > HEAD_TILT_THRESH

                color_eye = (0, 0, 255) if eyes_closed else (0, 255, 0)
                color_mouth = (0, 0, 255) if yawning else (0, 255, 0)
                color_head = (0, 0, 255) if head_tilted else (0, 255, 0)

                cv2.putText(frame, f"EAR:{ear:.2f}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_eye, 2)
                cv2.putText(frame, f"MAR:{mar:.2f}", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_mouth, 2)
                cv2.putText(frame, f"Tilt:{tilt:.1f}deg", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_head, 2)

                if eyes_closed:
                    cv2.putText(frame, "!! DROWSY - EYES CLOSED !!", (100, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if yawning:
                    cv2.putText(frame, "YAWNING", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                if head_tilted:
                    cv2.putText(frame, "HEAD TILT!", (10, 125),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                for hull in [cv2.convexHull(leftEye), cv2.convexHull(rightEye),
                              cv2.convexHull(mouth)]:
                    cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)

                break

            state.update(eyes_closed, yawning, head_tilted, ear, mar, tilt, face_detected)

            cv2.imshow("Driver Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyWindow("Driver Monitor")
        vs.stop()
        print("[Drowsiness] Monitor stopped.")
