# carla_frame_server.py  (Python 3.7)
import socket
import struct
import time
import numpy as np
import cv2
import carla
from threading import Thread, Lock

HOST = '127.0.0.1'
PORT = 6000

lock = Lock()
last_frame = None

def carla_camera_callback(image):
    global last_frame
    # convert CARLA raw_data -> BGRA -> BGR
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    bgr = array[:, :, :3][:, :, ::-1]  # CARLA raw is BGRA? ensure BGR
    with lock:
        last_frame = bgr.copy()

def frame_sender(conn):
    global last_frame
    try:
        while True:
            with lock:
                if last_frame is None:
                    frame = None
                else:
                    frame = last_frame.copy()
            if frame is None:
                time.sleep(0.01)
                continue

            # JPEG encode
            ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue
            data = buf.tobytes()
            # send length prefix then data
            conn.sendall(struct.pack('>I', len(data)))
            conn.sendall(data)
            # regulate send rate (e.g., ~15-25 FPS)
            time.sleep(0.05)
    except Exception as e:
        print('Sender error:', e)
    finally:
        try:
            conn.close()
        except:
            pass

def main():
    # Connect to running CARLA server
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    # spawn vehicle (or find an existing one)
    vehicle_bp = bp_lib.filter('vehicle')[0]
    spawn_points = world.get_map().get_spawn_points()
    transform = spawn_points[0] if spawn_points else carla.Transform(carla.Location(x=0, y=0, z=2))
    vehicle = world.try_spawn_actor(vehicle_bp, transform)
    if vehicle is None:
        # fallback: attach camera to spectator
        print('Failed to spawn vehicle — attaching to spectator')
        spectator = world.get_spectator()
        cam_parent = spectator
    else:
        cam_parent = vehicle

    # create camera
    cam_bp = bp_lib.find('sensor.camera.rgb')
    width, height, fov = 1280, 720, 90
    cam_bp.set_attribute('image_size_x', str(width))
    cam_bp.set_attribute('image_size_y', str(height))
    cam_bp.set_attribute('fov', str(fov))
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=cam_parent)
    camera.listen(carla_camera_callback)
    print('Camera listening. Waiting for client connection...')

    # TCP server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    conn, addr = server.accept()
    print('Client connected from', addr)

    sender_thread = Thread(target=frame_sender, args=(conn,))
    sender_thread.daemon = True
    sender_thread.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print('Shutting down...')
    finally:
        camera.stop()
        try:
            camera.destroy()
        except:
            pass
        try:
            vehicle.destroy()
        except:
            pass
        server.close()

if __name__ == '__main__':
    main()
