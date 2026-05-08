import cv2 as cv
import numpy as np


class Visualizer:
    def __init__(self, img, score):
        self.score_resized = cv.resize(score, (img.shape[1], img.shape[0]))
        heat = cv.applyColorMap((self.score_resized * 255).astype(np.uint8), cv.COLORMAP_JET)
        self.blended = cv.addWeighted(img, 0.6, heat, 0.4, 0)
        max_val = np.max(self.score_resized)
        self.thre = 0.97 * max_val

    def contour(self):
        safe = np.uint8((self.score_resized >= self.thre) * 255)
        cnts, _ = cv.findContours(safe, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        net = []
        if cnts:
            cmax = max(cnts, key=cv.contourArea)
            cmaxar = cv.contourArea(cmax)
            for c in cnts:
                if cv.contourArea(c) < 0.5 * cmaxar:
                    continue
                epsilon = 0.01 * cv.arcLength(c, True)
                poly = cv.approxPolyDP(c, epsilon, True)
                cv.polylines(self.blended, [poly], True, (0, 255, 0), 2)

                M = cv.moments(c)
                cx, cy = 0, 0
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv.putText(self.blended, "Safe Zone", (cx, cy),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                mask = np.zeros(self.score_resized.shape, dtype=np.uint8)
                cv.fillPoly(mask, [poly.astype(np.int32)], 1)
                values = self.score_resized[mask == 1]

                net.append([cx, cy, poly, values.mean(), values.var()])
        return net

    def show(self, window_name="Landing Safety Heatmap"):
        cv.imshow(window_name, self.blended)
        cv.waitKey(1)
