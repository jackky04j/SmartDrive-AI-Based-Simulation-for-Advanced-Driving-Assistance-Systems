import numpy as np
from .config import FOCAL_SCALE


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.prev_error = 0.0
        self.integral = 0.0

    def step(self, error, dt):
        self.integral = max(min(self.integral + error * dt, 2.0), -2.0)
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


def estimate_distance(h):
    return FOCAL_SCALE / max(h, 1)


def get_image_point(loc, w2c, K):
    point = np.array([loc.x, loc.y, loc.z, 1.0])
    point_cam = np.dot(w2c, point)
    point_cam = np.array([point_cam[1], -point_cam[2], point_cam[0]])
    if point_cam[2] > 0:
        point_img = np.dot(K, point_cam)
        u, v = point_img[0] / point_img[2], point_img[1] / point_img[2]
        return int(max(-1000, min(u, 2000))), int(max(-1000, min(v, 2000)))
    return None
