'''Pipeline class in charge of detecting and tracking vehicles'''
import numpy as np
import cv2

import car_helper
from pipeline import Pipeline
from car_data import CarData


class VehicleDetection(Pipeline):
    def __init__(self):
        super().__init__()
        data = CarData()

    def pipeline(self, img):
        img_copy = np.copy(img)

        assert img_copy.shape == img.shape
        return img