import threading


class DrowsinessState:
    def __init__(self):
        self._lock = threading.Lock()
        self.eyes_closed = False
        self.yawning = False
        self.head_tilted = False
        self.alert_level = "ALERT_NONE"
        self.ear = 1.0
        self.mar = 0.0
        self.head_tilt_deg = 0.0
        self.face_detected = False
        self.running = True

    def update(self, eyes_closed, yawning, head_tilted, ear, mar, tilt, face_det):
        with self._lock:
            self.eyes_closed = eyes_closed
            self.yawning = yawning
            self.head_tilted = head_tilted
            self.ear = ear
            self.mar = mar
            self.head_tilt_deg = tilt
            self.face_detected = face_det

            if eyes_closed:
                self.alert_level = "ALERT_CRITICAL"
            elif yawning or head_tilted:
                self.alert_level = "ALERT_WARN"
            else:
                self.alert_level = "ALERT_NONE"

    def snapshot(self):
        with self._lock:
            return (
                self.alert_level,
                self.eyes_closed,
                self.yawning,
                self.head_tilted,
                self.ear,
                self.mar,
                self.head_tilt_deg,
                self.face_detected,
            )
