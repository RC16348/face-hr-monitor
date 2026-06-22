import numpy as np
import onnxruntime as ort
import time

class FacePhysInferencer:
    def __init__(self, model_path, state_path=None):
        self.sess = ort.InferenceSession(str(model_path))
        self.state = None
        if state_path:
            self.state = self._load_state(state_path)

    def _load_state(self, path):
        import gzip, json
        with gzip.open(path, 'r') as f:
            return {k: np.array(v, dtype='float32')
                    for k, v in json.loads(f.read().decode()).items()}

    def inference(self, input_faces, fps=30):
        if self.state is None:
            self.state = self.init_state
        input_arr = np.array(input_faces, 'float32')
        y, t, dt = [], time.time(), np.array(1 / fps, 'float32')
        state = self.state

        for i in range(input_arr.shape[0]):
            r, state = self._step(input_arr[i], state, dt=dt)
            y.append(r)

        y = np.array(y)
        infer_fps = input_arr.shape[0] / (time.time() - t)
        return y, state, infer_fps

    def _step(self, img, state, dt=1 / 30):
        result = self.sess.run(None, {
            "input": img[None, None],
            "dt": [dt],
            **state
        })
        bvp, new_state = result[0][0, 0], result[1:]
        return bvp, dict(zip(state, new_state))

    @property
    def init_state(self):
        inputs = self.sess.get_inputs()[2:]
        return {inp.name: np.zeros(inp.shape, dtype='float32')
                for inp in inputs}
