import cv2 as cv
import numpy as np
import yaml
import os


def _load_config():
    share_path = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', '..',
        'share', 'px4_offboard', 'vision', 'config.yaml'
    )
    src_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    for path in [share_path, src_path]:
        path = os.path.normpath(path)
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f'config.yaml not found. Tried: {share_path}, {src_path}')


class ObstacleDetection:
    def __init__(self, model_path=None, obstacle_classes=None, imgsz=None, conf=None):
        # Lazy import — keeps torch from loading at module import time
        try:
            from ultralytics import YOLO
            cfg = _load_config()["obstacle_detection"]
            model_path = model_path or cfg["model_path"]
            self.model = YOLO(model_path)
            self.obstacle_class_id = obstacle_classes or cfg["obstacle_classes"]
            self.imgsz = imgsz or cfg["imgsz"]
            self.conf = conf or cfg["conf"]
            self._enabled = True
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f'ObstacleDetection disabled — YOLO failed to load: {e}')
            self.model = None
            self._enabled = False

        self.obj = {}

    def obstacle_map(self, img):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if not self._enabled:
            return mask

        self.results = self.model(
            img, classes=self.obstacle_class_id,
            imgsz=self.imgsz, conf=self.conf,
        )
        for result in self.results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                if cls_id in self.obstacle_class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cv.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    self.obj[self.model.names[cls_id]] = {
                        "conf": conf,
                        "cords": [x1, y1, x2, y2],
                    }
        return mask

    def obstacle_vis(self, img):
        for cls in self.obj:
            x1, y1, x2, y2 = self.obj[cls]["cords"]
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{cls} {self.obj[cls]['conf']:.2f}"
            cv.putText(img, label, (x1, y1 - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return imgx