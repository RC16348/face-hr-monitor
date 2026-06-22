import numpy as np
import threading

FACE_TARGET = 1600
FACE_SIZE = 36

class FaceBuffer:
    def __init__(self, capacity=FACE_TARGET, face_size=FACE_SIZE):
        self.capacity = capacity
        self.face_size = face_size
        self.faces = np.zeros((capacity, face_size, face_size, 3), dtype=np.float16)
        self.count = 0
        self.lock = threading.Lock()
        self._complete = False

    def add(self, face_crop):
        if self.count >= self.capacity:
            return False
        with self.lock:
            self.faces[self.count] = face_crop
            self.count += 1
            if self.count >= self.capacity:
                self._complete = True
        return True

    def is_full(self):
        return self._complete

    def reset(self):
        with self.lock:
            self.faces.fill(0)
            self.count = 0
            self._complete = False

    def get_data(self):
        with self.lock:
            return self.faces[:self.count].copy()

    def save(self, path):
        np.save(path, self.get_data())
        return path

    @property
    def progress(self):
        with self.lock:
            return self.count

def crop_face(frame, box, margin=0.2, face_size=FACE_SIZE):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    fw, fh = x2 - x1, y2 - y1
    mx, my = fw * margin, fh * margin
    x1 = max(0, int(x1 - mx))
    y1 = max(0, int(y1 - my))
    x2 = min(w - 1, int(x2 + mx))
    y2 = min(h - 1, int(y2 + my))
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (face_size, face_size))
    return face.astype(np.float16) / 255.0

import cv2
