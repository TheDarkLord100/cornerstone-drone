from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import numpy as np
import yaml
import os
from collections import deque


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

class Terrain:
    CLASS_NAMES = {
        0: "grassy_terrain",
        1: "marshy_terrain",
        2: "rocky_terrain",
        3: "sandy_terrain",
        4: "urban",
    }

    def __init__(self, frame, model_path=None):
        cfg = _load_config()["terrain"]
        model_path = model_path or cfg["model_path"]
        self.org_img = Image.fromarray(frame) if not isinstance(frame, Image.Image) else frame
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.probs = None

        self.safety_voting_window = cfg.get("safety_voting_window", 5)
        self.confidence_threshold = cfg.get("terrain_confidence_frames", 3)
        self.safety_history = deque(maxlen=self.safety_voting_window)
        self.class_history = deque(maxlen=self.safety_voting_window)
        self.confidence_history = deque(maxlen=self.safety_voting_window)

        self.last_safe_result = None
        self.consecutive_unsafe_frames = 0
        self.consecutive_safe_frames = 0

    def preprocess(self, size=(128, 128)):
        img = self.org_img.resize(size)
        arr = np.array(img).astype(np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def top_match(self):
        img = self.preprocess()
        self.interpreter.set_tensor(self.input_details[0]["index"], img)
        self.interpreter.invoke()
        self.probs = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        top_id = int(np.argmax(self.probs))
        return Terrain.CLASS_NAMES[top_id], round(self.probs[top_id], 2)

    def draw(self):
        if self.probs is None:
            raise ValueError("Run top_match() first to compute probabilities.")

        draw = ImageDraw.Draw(self.org_img)
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except OSError:
            font = ImageFont.load_default()

        y_offset = 10
        for i, name in Terrain.CLASS_NAMES.items():
            conf = self.probs[i]
            text = f"{name}: {conf:.2f}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.rectangle([(8, y_offset - 2), (8 + text_w + 4, y_offset + text_h)], fill="black")
            draw.text((10, y_offset), text, fill="white", font=font)
            y_offset += text_h + 5

        self.org_img.show()

    def _check_temporal_consistency(self, name, prob):
        self.class_history.append(name)
        self.confidence_history.append(prob)

        if len(self.class_history) < 2:
            return True

        return True

    def ifsafe(self):
        cfg = _load_config()["terrain"]
        name, prob = self.top_match()

        self.class_history.append(name)
        self.confidence_history.append(prob)

        is_safe_by_class = name not in cfg["unsafe_terrains"]
        is_safe_by_prob = prob >= cfg["safe_prob_threshold"]

        current_frame_safe = is_safe_by_class and is_safe_by_prob

        self.safety_history.append(current_frame_safe)

        if current_frame_safe:
            self.consecutive_safe_frames += 1
            self.consecutive_unsafe_frames = 0
        else:
            self.consecutive_unsafe_frames += 1
            self.consecutive_safe_frames = 0

        safe_votes = sum(self.safety_history)
        total_votes = len(self.safety_history)

        require_confidence = total_votes >= self.confidence_threshold

        if require_confidence:
            if safe_votes >= self.confidence_threshold:
                self.last_safe_result = "safe"
                return "safe"
            else:
                self.last_safe_result = "unsafe"
                return "unsafe"
        else:
            if current_frame_safe:
                return "safe"
            else:
                return "unsafe"

    def get_confidence_info(self):
        safe_votes = sum(self.safety_history)
        total_votes = len(self.safety_history)

        return {
            "consecutive_safe": self.consecutive_safe_frames,
            "consecutive_unsafe": self.consecutive_unsafe_frames,
            "safe_votes": safe_votes,
            "total_votes": total_votes,
            "last_result": self.last_safe_result,
            "recent_classes": list(self.class_history)[-3:] if self.class_history else [],
        }

    def reset_history(self):
        self.safety_history.clear()
        self.class_history.clear()
        self.confidence_history.clear()
        self.consecutive_safe_frames = 0
        self.consecutive_unsafe_frames = 0
        self.last_safe_result = None
