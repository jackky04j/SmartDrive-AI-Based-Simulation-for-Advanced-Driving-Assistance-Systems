# yolo_frame_client.py  (Python 3.10)
import socket
import struct
import numpy as np
import cv2
from ultralytics import YOLO
from config import CONE_MODEL_PATH

HOST = '127.0.0.1'
PORT = 6000

def recv_all(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

model = YOLO(str(CONE_MODEL_PATH))  # YOLOv11, run under Python 3.10

def process_frame(frame_bgr):
    # frame_bgr is an OpenCV BGR image
    # run detection
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = model(rgb)   # adapt if your API differs
    ann = res[0].plot() if hasattr(res[0], 'plot') else rgb
    ann_bgr = cv2.cvtColor(ann, cv2.COLOR_RGB2BGR)
    return ann_bgr

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    print('Connected to frame server')

    try:
        while True:
            # read 4-byte length
            raw_len = recv_all(sock, 4)
            if not raw_len:
                print('Server closed connection')
                break
            (length,) = struct.unpack('>I', raw_len)
            data = recv_all(sock, length)
            if data is None:
                print('Incomplete frame')
                break
            # decode jpeg
            buf = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR
            if img is None:
                continue

            out = process_frame(img)
            cv2.imshow('YOLO Output', out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print('Interrupted')
    finally:
        sock.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
