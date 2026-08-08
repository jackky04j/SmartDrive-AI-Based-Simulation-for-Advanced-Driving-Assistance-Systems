import math
import pygame

import carla

from .config import (
    ADAS_OBJECTS,
    CONF_THRESHOLD,
    MIN_BOX_AREA,
    MIN_VALID_DISTANCE,
    DIST_SLOW,
    DIST_BRAKE,
    DIST_EMERGENCY,
    TTC_BRAKE,
    TTC_EMERGENCY,
)
from .pid_controller import estimate_distance


def run_yolo_detection(yolo, rgb, surface, font, vehicle, WIDTH, HEIGHT):
    stop_detected = False
    min_distance = float("inf")

    results = yolo(rgb, verbose=False, conf=CONF_THRESHOLD)
    for r in results:
        for box in r.boxes:
            cls = yolo.names[int(box.cls[0])]
            if cls not in ADAS_OBJECTS:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if y2 > int(HEIGHT * 0.85):
                continue
            area = (x2 - x1) * (y2 - y1)
            if area < MIN_BOX_AREA:
                continue
            dist = estimate_distance(y2 - y1)
            if dist < MIN_VALID_DISTANCE:
                continue
            if cls == "traffic light":
                if vehicle.get_traffic_light_state() == carla.TrafficLightState.Green:
                    continue
            min_distance = min(min_distance, dist)
            if cls == "stop sign":
                stop_detected = True
            pygame.draw.rect(surface, (0, 255, 0), (x1, y1, x2 - x1, y2 - y1), 2)
            surface.blit(font.render(f"{cls} {dist:.1f}m", True, (0, 255, 0)), (x1, y1 - 18))

    return stop_detected, min_distance


def compute_risk_level(min_distance, speed_ms):
    ttc = min_distance / speed_ms if speed_ms > 0.3 else float("inf")

    if min_distance < DIST_EMERGENCY or ttc < TTC_EMERGENCY:
        return "EMERGENCY"
    elif min_distance < DIST_BRAKE or ttc < TTC_BRAKE:
        return "BRAKE"
    elif min_distance < DIST_SLOW:
        return "SLOW"
    return "SAFE"
