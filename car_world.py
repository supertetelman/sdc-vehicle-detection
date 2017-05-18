
'''The CarWorld Class it the main class for the car project.
This class will 
  process the input (video or image)
  run the data through various detection pipelines (lane/vehicle detection)
  return annoted output
'''
from moviepy.editor import VideoFileClip
import os

import lane_lines
import vehicle_detection
import car_helper

from pipeline import Pipeline

import matplotlib.pyplot as plt
import numpy as np

class CarWorld(Pipeline):
    def __init__(self):
        super().__init__()
        self.calibrate() # Calibration cannot occur until after super init
        self.lanes = lane_lines.LaneLines()
        self.vehicles = vehicle_detection.VehicleDetection(True, "big")

    def process_video(self, input_vid, output_vid, debug_all=False):
        '''Run an video through the pipeline, allow debug options'''
        print("Running %s through pipeline and outputting to %s" %(input_vid, output_vid))
        clip = VideoFileClip(input_vid)
        func = self.pipeline
        if debug_all:
            func = self.debug_pipeline
        output = clip.fl_image(func)
        output.write_videofile(output_vid, audio = False)

    def pipeline(self, img, debug_all=False):
        '''Run an image through the pipeline
        pipline is an overlay of lane detection and vehicle detection
        '''
        # Correct for camera distortion
        img = self.correct_distortion(img)

        # Create a list of images for each stage
        if debug_all:
            imgs = {'img': img}

        '''
        # Identify Lanes if debug return all images, else verify size
        lanes_img = self.lanes.pipeline(np.copy(img), debug_all=debug_all)
        if debug_all:
            for key in lanes_img.keys():
                lanekey = "lanes-%s" %key
                imgs[lanekey] = lanes_img[key] 
        else:
            assert lanes_img.shape == img.shape
        '''

        # Identify Cars, if debug return all images, else verify size
        vehicles_img = self.vehicles.pipeline(np.copy(img), debug_all=debug_all)
        if debug_all:
            for key in vehicles_img.keys():
                vehicleskey = "vehicles-%s" %key
                imgs[vehicleskey] = vehicles_img[key] 
        else:
            assert vehicles_img.shape == img.shape

        # Combine Results for debug/non-debug image
        if debug_all:
            #img = car_helper.overlay_img(lanes_img['final'], vehicles_img['final'])
            imgs['final'] = vehicles_img['final']
            return imgs
        else:
            # img = car_helper.overlay_img(lanes_img, vehicles_img)
            img = car_helper.overlay_img(img, vehicles_img)
        return img

if __name__ == '__main__':
    cw = CarWorld()
    # Run test videos through pipeline
    input_vid = os.path.join("test_vid",'tiny-1.mp4')
    output_vid = os.path.join(cw.results_dir, "tiny_1_output.mp4")
    cw.process_video(input_vid, output_vid)

    # Run test videos through pipeline
    input_vid = os.path.join("test_vid",'tiny-1.mp4')
    output_vid = os.path.join(cw.results_dir, "tiny_1_output_debug.mp4")
    cw.process_video(input_vid, output_vid, debug_all=True)

    # Run test videos through pipeline
    input_vid = os.path.join("test_vid",'tiny-2.mp4')
    output_vid = os.path.join(cw.results_dir, "tiny_2_output.mp4")
    cw.process_video(input_vid, output_vid)

    # Run test videos through pipeline
    input_vid = os.path.join("test_vid", "short-2.mp4")
    output_vid = os.path.join(cw.results_dir, "small_2_output.mp4")
    cw.process_video(input_vid, output_vid)

    # Run test videos through pipeline
    input_vid = os.path.join("test_vid",'project_video.mp4')
    output_vid = os.path.join(cw.results_dir, "project_video_output.mp4")
    cw.process_video(input_vid, output_vid)

    # Run test videos through pipeline
    input_vid = os.path.join("test_vid",'project_video.mp4')
    output_vid = os.path.join(cw.results_dir, "project_video_output_debug.mp4")
    cw.process_video(input_vid, output_vid, debug_all=True)

    input_vid = os.path.join("test_vid",'challenge_video.mp4')
    output_vid = os.path.join(cw.results_dir, "challenge_video_output.mp4")
    cw.process_video(input_vid, output_vid)

    input_vid = os.path.join("test_vid",'harder_challenge_video.mp4')
    output_vid = os.path.join(cw.results_dir, "harder_challenge_video_output.mp4")
    cw.process_video(input_vid, output_vid)
  