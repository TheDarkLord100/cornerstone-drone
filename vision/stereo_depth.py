import cv2
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


class StereoDepth:
    def __init__(
        self,
        num_disparities=None,
        block_size=None,
        uniqueness_ratio=None,
        speckle_range=None,
        speckle_window_size=None,
    ):
        cfg = _load_config()["stereo_depth"]

        self.num_disparities = num_disparities or cfg["num_disparities"]
        self.block_size = block_size or cfg["block_size"]
        self.uniqueness_ratio = uniqueness_ratio or cfg["uniqueness_ratio"]
        self.speckle_range = speckle_range or cfg["speckle_range"]
        self.speckle_window_size = speckle_window_size or cfg["speckle_window_size"]

        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            uniquenessRatio=self.uniqueness_ratio,
            speckleRange=self.speckle_range,
            speckleWindowSize=self.speckle_window_size,
            disp12MaxDiff=1,
            P1=8 * 3 * self.block_size ** 2,
            P2=32 * 3 * self.block_size ** 2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)

        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(
            matcher_left=self.left_matcher
        )
        self.wls_filter.setLambda(cfg.get("wls_lambda", 8000))
        self.wls_filter.setSigmaColor(cfg.get("wls_sigma", 1.5))

        self.fx = cfg["fx"]
        self.baseline = cfg["baseline_mm"]
        self.depth_scale = cfg.get("depth_scale", 1000.0)

    def compute(self, left_gray, right_gray):
        left_disp = self.left_matcher.compute(left_gray, right_gray)
        right_disp = self.right_matcher.compute(right_gray, left_gray)

        filtered_disp = self.wls_filter.filter(
            left_disp, left_gray, None, right_gray
        )

        filtered_disp = cv2.normalize(
            filtered_disp, filtered_disp, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        depth = np.zeros_like(filtered_disp, dtype=np.float32)
        valid = filtered_disp > 0
        depth[valid] = (self.fx * self.baseline) / (
            filtered_disp[valid].astype(np.float32) * self.depth_scale
        )

        depth_m = depth.astype(np.float32) / 1000.0
        depth_m[depth_m <= 0] = 0
        depth_m[depth_m > 50] = 0

        return filtered_disp, depth_m

    def compute_pointcloud(self, left_gray, depth_m):
        h, w = left_gray.shape
        k = np.array([
            [self.fx, 0, _load_config()["stereo_camera"]["cx"]],
            [0, self.fx, _load_config()["stereo_camera"]["cy"]],
            [0, 0, 1],
        ], dtype=np.float32)

        points = []
        colors = []
        for v in range(0, h, 4):
            for u in range(0, w, 4):
                d = depth_m[v, u]
                if d <= 0 or d > 50:
                    continue
                x = (u - k[0, 2]) * d / k[0, 0]
                y = (v - k[1, 2]) * d / k[1, 1]
                points.append([x, y, d])
                colors.append(left_gray[v, u])

        return np.array(points), np.array(colors)
