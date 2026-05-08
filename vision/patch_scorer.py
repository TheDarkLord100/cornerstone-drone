import cv2 as cv
import numpy as np
import yaml
import os

def _load_config():
    # Installed via colcon into share directory
    share_path = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', '..',
        'share', 'px4_offboard', 'vision', 'config.yaml'
    )
    # Source directory fallback
    src_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    for path in [share_path, src_path]:
        path = os.path.normpath(path)
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f'config.yaml not found. Tried: {share_path}, {src_path}')

class PatchScorer:
    def __init__(self, patch_size=None, weights=None):
        cfg = _load_config()["patch_scorer"]
        self.patch_size = patch_size or cfg["patch_size"]
        w = weights or cfg["weights"]
        self.w_edge = w["edge"]
        self.w_texture = w["texture"]
        self.w_blob = w["blob"]
        self.w_shadow = w["shadow"]

    def preprocess(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 1.2)
        return gray

    def patchdim(self, img):
        h, w = img.shape[:2]
        ph = h // self.patch_size
        pw = w // self.patch_size
        return ph, pw

    def edge_density(self, gray):
        gx = cv.Sobel(gray, cv.CV_32F, 1, 0)
        gy = cv.Sobel(gray, cv.CV_32F, 0, 1)
        mag = cv.magnitude(gx, gy)
        return mag

    def texture_variance(self, gray):
        mean = cv.blur(gray, (5, 5))
        sqmean = cv.blur(gray.astype(np.float32) ** 2, (5, 5))
        return sqmean - mean.astype(np.float32) ** 2

    def blob_density(self, gray):
        log = cv.GaussianBlur(gray, (0, 0), 2)
        log = cv.Laplacian(log, cv.CV_32F)
        return np.abs(log)

    def shadow_map(self, img):
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        shadow = 1.0 - (hsv[:, :, 2] / 255.0)
        return shadow

    def normalize(self, m):
        m = m.astype(float)
        return (m - m.min()) / (m.max() - m.min() + 1e-6)

    def compute_scores(self, img, obstacle_detector=None):
        gray = self.preprocess(img)
        ph, pw = self.patchdim(gray)

        edge_map = self.normalize(self.edge_density(gray))
        texture_var_map = self.normalize(self.texture_variance(gray))
        blob_map = self.normalize(self.blob_density(gray))
        shadow_map = self.normalize(self.shadow_map(img))

        edge_small = cv.resize(edge_map, (pw, ph), interpolation=cv.INTER_AREA)
        tex_small = cv.resize(texture_var_map, (pw, ph), interpolation=cv.INTER_AREA)
        shadow_small = cv.resize(shadow_map, (pw, ph), interpolation=cv.INTER_AREA)
        blob_small = cv.resize(blob_map, (pw, ph), interpolation=cv.INTER_AREA)

        scores = (
            self.w_edge * (1 - edge_small)
            + self.w_texture * (1 - tex_small)
            + self.w_shadow * (1 - shadow_small)
            + self.w_blob * (1 - blob_small)
        )

        if obstacle_detector is not None:
            obstacle_mask = obstacle_detector.obstacle_map(img)
            obstacle_mask_small = cv.resize(obstacle_mask, (pw, ph), interpolation=cv.INTER_AREA)
            scores[obstacle_mask_small > 0] = 0.0

        return scores
