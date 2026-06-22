import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

CONF_THRESH = 0.5
IOU_THRESH = 0.5

class FaceDetector:
    def __init__(self, model_path):
        self.sess = ort.InferenceSession(str(model_path))
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess(self, frame, target_size=640):
        h, w = frame.shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_size, target_size))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None, :, :, :]
        return img, w, h

    def postprocess(self, output, orig_w, orig_h):
        data = output[0][0]
        scores = data[4, :]
        mask = scores > CONF_THRESH
        if not mask.any():
            return [], [], []

        data = data[:, mask]
        scores = data[4, :]
        cx, cy, w, h = data[:4]

        x1 = ((cx - w / 2) / 640 * orig_w).clip(0, orig_w - 1)
        y1 = ((cy - h / 2) / 640 * orig_h).clip(0, orig_h - 1)
        x2 = ((cx + w / 2) / 640 * orig_w).clip(0, orig_w - 1)
        y2 = ((cy + h / 2) / 640 * orig_h).clip(0, orig_h - 1)
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        keep = self._nms(boxes, scores)
        boxes = boxes[keep]
        scores = scores[keep]

        kpts = []
        for i in keep:
            kp = data[5:, i].reshape(5, 3)
            kp[:, 0] = (kp[:, 0] / 640 * orig_w).clip(0, orig_w - 1)
            kp[:, 1] = (kp[:, 1] / 640 * orig_h).clip(0, orig_h - 1)
            kpts.append(kp)
        return boxes, scores, kpts

    def _nms(self, boxes, scores):
        idxs = scores.argsort()[::-1]
        keep = []
        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)
            if len(idxs) == 1:
                break
            xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_j = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
            union = area_i + area_j - inter
            iou = inter / (union + 1e-6)
            idxs = idxs[1:][iou <= IOU_THRESH]
        return keep

    def detect(self, frame):
        input_arr, ow, oh = self.preprocess(frame)
        out = self.sess.run(None, {self.input_name: input_arr})
        return self.postprocess(out, ow, oh)

