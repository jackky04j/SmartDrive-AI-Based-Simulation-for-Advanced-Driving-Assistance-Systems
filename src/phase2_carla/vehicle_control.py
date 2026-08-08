import math
import time
import random

import carla
import pygame

from .config import (
    DT,
    TARGET_SPEED,
    LOOKAHEAD_STRAIGHT,
    LOOKAHEAD_CURVE,
    CURVE_YAW_THRESHOLD,
)
from .pid_controller import PIDController, get_image_point


def handle_manual_control(keys, drowsy_critical):
    control = carla.VehicleControl()
    if drowsy_critical:
        control.brake = 1.0
        control.throttle = 0.0
    else:
        control.throttle = 0.8 if keys[pygame.K_w] else 0.0
        control.brake = 1.0 if keys[pygame.K_s] else 0.0
        control.steer = 0.5 * (keys[pygame.K_d] - keys[pygame.K_a])
        control.reverse = keys[pygame.K_r]
    return control


def handle_auto_control(
    vehicle, wp, pid, pid_throttle,
    stop_detected, risk_level,
    drowsy_critical, drowsy_warn, drowsy_brake_until,
    prev_steer, stuck_timer, recovery_mode,
    in_junction, lane_lock_id,
    camera, K_mat, surface, WIDTH, HEIGHT,
):
    control = carla.VehicleControl()

    veh_yaw = vehicle.get_transform().rotation.yaw
    road_yaw = wp.transform.rotation.yaw

    yaw_preview = abs(math.atan2(
        math.sin(math.radians(road_yaw - veh_yaw)),
        math.cos(math.radians(road_yaw - veh_yaw))
    ))

    dynamic_lookahead = (LOOKAHEAD_CURVE if yaw_preview > CURVE_YAW_THRESHOLD
                         else LOOKAHEAD_STRAIGHT)
    next_wps = wp.next(dynamic_lookahead)

    if wp.is_junction:
        if not in_junction:
            best_wp = random.choice(next_wps)
            target_wp = best_wp
            lane_lock_id = best_wp.lane_id
            in_junction = True
        else:
            target_wp = next_wps[0]
            for p in next_wps:
                if p.lane_id == lane_lock_id:
                    target_wp = p
                    break
    else:
        in_junction = False
        lane_lock_id = None
        target_wp = next_wps[0]
        for p in next_wps:
            if p.lane_id == wp.lane_id:
                target_wp = p
                break

    veh_tf = vehicle.get_transform()
    yaw_err = math.atan2(
        math.sin(math.radians(target_wp.transform.rotation.yaw - veh_tf.rotation.yaw)),
        math.cos(math.radians(target_wp.transform.rotation.yaw - veh_tf.rotation.yaw))
    )
    dx = target_wp.transform.location.x - veh_tf.location.x
    dy = target_wp.transform.location.y - veh_tf.location.y
    cte = -(math.sin(math.radians(veh_tf.rotation.yaw)) * dx -
            math.cos(math.radians(veh_tf.rotation.yaw)) * dy)

    raw_steer = max(-0.8, min(0.8, pid.step(cte + yaw_err, DT)))
    control.steer = 0.3 * raw_steer + 0.7 * prev_steer

    v = vehicle.get_velocity()
    speed_ms = math.sqrt(v.x**2 + v.y**2 + v.z**2)
    speed_kmh = speed_ms * 3.6

    if drowsy_critical or time.time() < drowsy_brake_until:
        control.brake = 1.0
        control.throttle = 0.0
        pid_throttle.integral = 0.0
    elif stop_detected or risk_level == "EMERGENCY":
        control.brake = 1.0
        pid_throttle.integral = 0.0
    elif risk_level == "BRAKE":
        control.brake = 0.7
        pid_throttle.integral = 0.0
    elif risk_level == "SLOW" or drowsy_warn:
        control.throttle = 0.15
    elif recovery_mode:
        control.reverse = True
        control.throttle = 0.4
        control.steer = -0.3
        if time.time() - stuck_timer > 4.0:
            recovery_mode = False
            stuck_timer = None
    else:
        speed_error = (TARGET_SPEED / 3.6) - speed_ms
        control.throttle = max(0.0, min(0.8, pid_throttle.step(speed_error, DT)))
        if speed_kmh < 1.0:
            if stuck_timer is None:
                stuck_timer = time.time()
            elif time.time() - stuck_timer > 3.0:
                recovery_mode = True
        else:
            stuck_timer = None

    w2c = import_numpy().array(camera.get_transform().get_inverse_matrix())
    _draw_lane_projection(surface, camera, target_wp, wp, w2c, K_mat, WIDTH, HEIGHT)

    return control, in_junction, lane_lock_id, stuck_timer, recovery_mode


def _draw_lane_projection(surface, camera, target_wp, wp, w2c, K_mat, WIDTH, HEIGHT):
    import numpy as np
    import pygame

    w2c = np.array(camera.get_transform().get_inverse_matrix())

    future_wps = [target_wp]
    for _ in range(15):
        next_opt = future_wps[-1].next(2.0)
        if not next_opt:
            break
        future_wps.append(next_opt[0])

    center_pts = []
    left_pts = []
    right_pts = []

    for fw in future_wps:
        img_pt = get_image_point(fw.transform.location, w2c, K_mat)
        if img_pt and 0 <= img_pt[0] < WIDTH and 0 <= img_pt[1] < HEIGHT:
            center_pts.append(img_pt)
            pygame.draw.circle(surface, (255, 0, 0), img_pt, 3)
        hw = fw.lane_width / 2.0
        right_vec = fw.transform.get_right_vector()
        loc_l = fw.transform.location - right_vec * hw
        loc_r = fw.transform.location + right_vec * hw
        pt_l = get_image_point(loc_l, w2c, K_mat)
        pt_r = get_image_point(loc_r, w2c, K_mat)
        if pt_l and 0 <= pt_l[0] < WIDTH and 0 <= pt_l[1] < HEIGHT:
            left_pts.append(pt_l)
        if pt_r and 0 <= pt_r[0] < WIDTH and 0 <= pt_r[1] < HEIGHT:
            right_pts.append(pt_r)

    if len(center_pts) > 1:
        pygame.draw.lines(surface, (0, 255, 255), False, center_pts, 3)
    if len(left_pts) > 1:
        pygame.draw.lines(surface, (255, 255, 0), False, left_pts, 2)
    if len(right_pts) > 1:
        pygame.draw.lines(surface, (255, 255, 0), False, right_pts, 2)


def import_numpy():
    import numpy as np
    return np
