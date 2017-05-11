
'''The CarWorld Class it the main class for the car project.
This class will 
  process the input (video or image)
  run the data through various detection pipelines (lane/vehicle detection)
  return annoted output
'''

from moviepy.editor import VideoFileClip

import lane_lines
import vehicle_detection
import car_helper

from pipeline import Pipeline


class CarWorld(Pipeline):
    def __init__(self):
        super().__init__()
        self.lanes = lane_lines.LaneLines()
        self.vehicles = vehicle_detection.VehicleDetection()
        self.calibrate() # Calibration cannot occur until after super init


    def process_video(self, input_vid, output_vid):
        '''Run an video through the pipelines'''
        print("Running %s through pipeline and outputting to %s" %(input_vid, output_vid))
        clip = VideoFileClip(input_vid)
        output = clip.fl_image(self.pipeline)
        output.write_videofile(output_vid, audio = False)

    def pipeline(self, img):
        '''Run an image through the pipeline
        pipline is an overlay of lane detection and vehicle detection
        '''
        # Correct for camera distortion
        img = self.correct_distortion(img)

        # Identify Lanes
        lanes_img = self.lanes.pipeline(img)
        assert lanes_img.shape == img.shape

        # Identify Cars
        vehicles_img = self.vehicles.pipeline(img)
        assert vehicles_img.shape == img.shape

        # Combine Results
        img = car_helper.overlay_img(lanes_img, vehicles_img)

        return img