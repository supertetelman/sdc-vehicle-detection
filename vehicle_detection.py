'''Pipeline class in charge of detecting and tracking vehicles'''
import numpy as np
import cv2

import car_helper
from pipeline import Pipeline


class VehicleDetection(Pipeline):
    def __init__(self):
        super().__init__()

    def pipeline(self, img):
        img_copy = np.copy(img)

        assert img_copy.shape == img.shape
        return img