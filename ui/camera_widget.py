import cv2
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QImage, QPixmap
from PySide6.QtCore import Qt


def put_text_bg(img, text, pos, scale=0.6, text_color=(255, 255, 255),
                bg_color=(0, 0, 0), thickness=2, bg_alpha=0.55, padding=6):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = pos
    x1 = max(0, x - padding)
    y1 = max(0, y - th - padding)
    x2 = x + tw + padding
    y2 = y + padding
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                text_color, thickness, cv2.LINE_AA)


def draw_circular_progress(img, center, radius, progress, total):
    overlay = img.copy()
    angle = int(360 * progress / total) if total > 0 else 0
    color = (0, 255, 0) if progress < total else (0, 200, 255)
    cv2.ellipse(overlay, center, (radius, radius), 0, -90, angle - 90,
                color, 8, cv2.LINE_AA)
    cv2.putText(overlay, f'{progress}/{total}',
                (center[0] - 60, center[1] + 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)


class CameraWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.frame = None
        self.boxes = []
        self.scores = []
        self.kpts = []
        self.face_count = 0
        self.face_target = 1600
        self.fps = 0
        self.no_face = False
        self.collecting = False
        self.detecting = False

    def update_frame(self, frame, boxes, scores, kpts, face_count, fps,
                     no_face=False, collecting=False, detecting=False):
        self.frame = frame
        self.boxes = boxes
        self.scores = scores
        self.kpts = kpts
        self.face_count = face_count
        self.fps = fps
        self.no_face = no_face
        self.collecting = collecting
        self.detecting = detecting
        self.face_target = 1600
        self.update()

    def paintEvent(self, event):
        if self.frame is None:
            return

        h, w = self.frame.shape[:2]
        scaled = self.frame.copy()

        put_text_bg(scaled, f'{w}x{h}  |  {int(self.fps)} FPS',
                    (12, 28), scale=0.48,
                    text_color=(180, 200, 220), bg_color=(10, 15, 30), bg_alpha=0.4,
                    padding=5)

        if not self.detecting:
            put_text_bg(scaled, '检测待启动', (12, 56), scale=0.48,
                        text_color=(120, 140, 160), bg_color=(10, 15, 30), bg_alpha=0.4,
                        padding=5)
        else:
            status_color = (15, 188, 156) if not self.no_face else (231, 76, 60)
            status_text = 'Face Detected' if not self.no_face else 'No Face'
            if self.collecting:
                status_text += '  |  Collecting...'
            put_text_bg(scaled, status_text, (12, 56), scale=0.55,
                        text_color=status_color, bg_color=(10, 15, 30), bg_alpha=0.5)

            for box, score, kp in zip(self.boxes, self.scores, self.kpts):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(scaled, (x1, y1), (x2, y2), (15, 188, 156), 2)
                put_text_bg(scaled, f'{score:.2f}', (x1, max(y1 - 16, 0)),
                            scale=0.45, text_color=(15, 188, 156),
                            bg_color=(10, 15, 30), bg_alpha=0.5, padding=4)
                for kx, ky, conf in kp:
                    if conf > 0.5:
                        cv2.circle(scaled, (int(kx), int(ky)), 2, (231, 76, 60), -1)

            draw_circular_progress(scaled, (w - 90, 80), 48,
                                   self.face_count, self.face_target)

            put_text_bg(scaled, f'采集: {self.face_count}/{self.face_target}',
                        (12, h - 14), scale=0.5,
                        text_color=(15, 188, 156) if self.collecting else (120, 140, 160),
                        bg_color=(10, 15, 30), bg_alpha=0.45, padding=6)

        rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
        h2, w2 = rgb.shape[:2]
        qimg = QImage(rgb.data, w2, h2, w2 * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled_pix = pix.scaled(self.size(), Qt.KeepAspectRatio,
                                Qt.SmoothTransformation)
        painter = QPainter(self)
        painter.drawPixmap(0, 0, scaled_pix)
        painter.end()
