import math
import queue
import random
import time
import threading

import carla
import numpy as np
import pygame
from ultralytics import YOLO

from .config import DT, K_P, K_I, K_D, YOLO_MODEL_PATH
from .pid_controller import PIDController
from .drowsiness_state import DrowsinessState
from .drowsiness_monitor import drowsiness_thread
from .adas_detection import run_yolo_detection, compute_risk_level
from .vehicle_control import handle_manual_control, handle_auto_control
from .hud import draw_hud, draw_mode_indicator, draw_alerts


def main():
    pygame.init()
    WIDTH, HEIGHT = 900, 600
    is_fullscreen = False
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("CARLA ADAS + Drowsiness Monitor  [F / F11 = Fullscreen]")
    font = pygame.font.SysFont("Arial", 18)
    alert_font = pygame.font.SysFont("Arial", 42, bold=True)
    clock = pygame.time.Clock()

    dstate = DrowsinessState()
    d_thread = threading.Thread(target=drowsiness_thread, args=(dstate, 0), daemon=True)
    d_thread.start()

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = DT
    world.apply_settings(settings)

    actors = []
    try:
        bp_lib = world.get_blueprint_library()
        vehicle = world.spawn_actor(
            bp_lib.filter("vehicle.tesla.model3")[0],
            world.get_map().get_spawn_points()[0]
        )
        actors.append(vehicle)

        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(WIDTH))
        cam_bp.set_attribute("image_size_y", str(HEIGHT))

        cam_views = {
            1: carla.Transform(carla.Location(x=-6, z=4), carla.Rotation(pitch=-15)),
            2: carla.Transform(carla.Location(x=0.0, z=20.0), carla.Rotation(pitch=-90.0)),
            3: carla.Transform(carla.Location(x=2.0, z=1.2))
        }
        camera = world.spawn_actor(cam_bp, cam_views[1], attach_to=vehicle)
        actors.append(camera)

        image_queue = queue.Queue(maxsize=1)

        def _enqueue(img):
            if not image_queue.empty():
                try:
                    image_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                image_queue.put_nowait(img)
            except queue.Full:
                pass

        camera.listen(_enqueue)

        yolo = YOLO(YOLO_MODEL_PATH)
        pid = PIDController(K_P, K_I, K_D)
        pid_throttle = PIDController(0.3, 0.01, 0.1)

        fov = 90.0
        cx = WIDTH / 2.0
        cy = HEIGHT / 2.0
        f = cx / math.tan(math.radians(fov / 2.0))
        K_mat = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

        AUTO_MODE = False
        stuck_timer = None
        recovery_mode = False
        prev_steer = 0.0
        lane_lock_id = None
        in_junction = False
        drowsy_brake_until = 0.0

        while True:
            world.tick()
            clock.tick(20)

            try:
                image = image_queue.get(timeout=2.0)
            except Exception:
                continue

            raw = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((HEIGHT, WIDTH, 4))
            rgb = raw[:, :, :3][:, :, ::-1]
            surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

            (alert_level, eyes_closed, yawning, head_tilted,
             ear, mar, head_tilt_deg, face_detected) = dstate.snapshot()

            drowsy_critical = alert_level == "ALERT_CRITICAL"
            drowsy_warn = alert_level == "ALERT_WARN"

            if drowsy_critical and AUTO_MODE:
                drowsy_brake_until = time.time() + 3.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m:
                        AUTO_MODE = not AUTO_MODE
                    if event.key in (pygame.K_f, pygame.K_F11):
                        is_fullscreen = not is_fullscreen
                        if is_fullscreen:
                            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        else:
                            screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                    if event.key == pygame.K_1:
                        camera.set_transform(cam_views[1])
                    if event.key == pygame.K_2:
                        camera.set_transform(cam_views[2])
                    if event.key == pygame.K_3:
                        camera.set_transform(cam_views[3])
                    if event.key == pygame.K_t:
                        dummy_bp = random.choice(bp_lib.filter("vehicle.*"))
                        spawn_tf = vehicle.get_transform()
                        fwd = spawn_tf.get_forward_vector()
                        spawn_tf.location.x += fwd.x * 20.0
                        spawn_tf.location.y += fwd.y * 20.0
                        spawn_tf.location.z += 0.5
                        dummy = world.try_spawn_actor(dummy_bp, spawn_tf)
                        if dummy:
                            actors.append(dummy)

            wp = world.get_map().get_waypoint(vehicle.get_location(), project_to_road=True)

            stop_detected, min_distance = run_yolo_detection(
                yolo, rgb, surface, font, vehicle, WIDTH, HEIGHT
            )

            v = vehicle.get_velocity()
            speed_ms = math.sqrt(v.x**2 + v.y**2 + v.z**2)
            speed_kmh = speed_ms * 3.6
            risk_level = compute_risk_level(min_distance, speed_ms)

            if not AUTO_MODE:
                keys = pygame.key.get_pressed()
                control = handle_manual_control(keys, drowsy_critical)
            else:
                control, in_junction, lane_lock_id, stuck_timer, recovery_mode = handle_auto_control(
                    vehicle, wp, pid, pid_throttle,
                    stop_detected, risk_level,
                    drowsy_critical, drowsy_warn, drowsy_brake_until,
                    prev_steer, stuck_timer, recovery_mode,
                    in_junction, lane_lock_id,
                    camera, K_mat, surface, WIDTH, HEIGHT,
                )

            vehicle.apply_control(control)
            prev_steer = control.steer

            sw, sh = screen.get_size()
            if (sw, sh) != (WIDTH, HEIGHT):
                surface = pygame.transform.scale(surface, (sw, sh))
            screen.blit(surface, (0, 0))

            draw_mode_indicator(screen, font, AUTO_MODE)
            draw_hud(screen, font, wp, speed_kmh, risk_level,
                     alert_level, face_detected, ear, mar, head_tilt_deg)
            draw_alerts(screen, alert_font, alert_level, risk_level, WIDTH, HEIGHT)

            pygame.display.flip()

    finally:
        dstate.running = False
        world.apply_settings(original_settings)
        for a in actors:
            a.destroy()
        pygame.quit()


if __name__ == "__main__":
    main()
