'''Pipeline class in charge of detecting and tracking vehicles'''
import car_helper
from pipeline import Pipeline


class VehicleDetection(Pipeline):
    def __init__(self):
        super().__init__()

    def pipeline(self, img):
        return img