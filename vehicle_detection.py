'''Pipeline class in charge of detecting and tracking vehicles'''

import pickle
import os
from collections import deque

import cv2
from skimage.feature import hog

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

import car_helper
from pipeline import Pipeline
from car_data import CarData


class VehicleDetection(Pipeline):
    def __init__(self, data_file=False, data="small", n=10):
        super().__init__()

        # The Simple Vector Machine trained to detect cars
        self.svm = None

        # The scaler trained to scale X input
        self.X_scaler = None

        # Load pre-trained models or load training data
        if data_file:
            self.load_pickle(data_file) # Updates scaler and svm
            self.data = None
        else:
            self.data = CarData(data)

        # fit-sized queue to store box coordinates of cars detected last n imgs
        self.cars =  deque(maxlen = n)

        # The currently identifed car points
        self.current_cars = None

        # The blocks where a car has been detected
        self.current_blocks = None

    def load_pickle(self, data_file):
        models = pickle.load(open(data_file, 'rb'))
        self.svm = models['svm']
        self.X_scaler = models['X_scaler']

    def save_pickle(self, data_file):
        pickle.dumpe({'svm': self.svm, 'X_scaler': self.X_scaler},
                open(data_file, 'wb'))

    def train(self):
        # Initialize models
        self.svc = LinearSVC()
        self.X_scaler = StandardScaler()

        # Split the data into train, validation, and test data
        self.split_data()

        # Extract features from the data
        self.extract_data_features()

        # Train the scaler
        self.train_scaler()

        # Train the svm
        self.train_svm()

    def split_data(self):
        # TODO:
        pass

    def extract_data_features(self):
        # TODO:
        pass

    def train_scaler(self):
        # TODO:
        pass

    def train_svm(self):
        # TODO:
        pass

    def pipeline(self, img):
        '''Given an image return an image with boxes drawn around all vehicles
        It is assumed that the incoming image is undistorted.
        '''
        img_copy = np.copy(img)

        # Detect cars in this image
        self.detect_blocks(img)

        # Create a heat map based on the detected car blocks
        self.calculate_heat(img)

        # Use the detected blocks, heat map, and previous data to detect cars
        self.detect_cars(img)

        # Outline the currently detected cars
        img = car_helper.draw_boxes(img, self.current_cars)

        # Return the annoted image
        assert img_copy.shape == img.shape
        return img

    def detect_blocks(self, img, color='YCrCb', hist_bins=32, spatial_size=(32, 32)):
        # Verify we have been trained
        assert self.svm is not None
        assert self.X_scaler is not None

        # Initialize the current list of detected car blocks
        self.current_blocks = []

        # TODO: Don't scan the horizon for cars; they don't fly yet.
        ystart = 0 #Should be top of image

        # Set some window size params
        pix_per_cell = 8 # Size of square windows TODO: tune
        cell_per_block = 2 # TODO: tune
        window_count = 64 # Total number of windows # TODO: tune
        cells_per_step = 2 # How many cells to step during window slide # TODO: tune
        orient = 9

        # TODO: Verify image is scaled to 255 not 1
        scale = 1

        # Convert from BRG to color
        plt.imshow(img)
        plt.show()
        search_image = car_helper.convert_img(img, color)
        plt.imshow(search_image)
        plt.show()
        # TODO: rescale image based on input?

        # Extract individual color channels
        ch1 = search_image[:,:,0]
        ch2 = search_image[:,:,1]
        ch3 = search_image[:,:,2]

        # Define blocks and steps based on img size
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
        nfeat_per_block = orient * cell_per_block**2

        # Calculate number of steps in the y/x directions and # blocks per window
        window_blocks = (window_count // pix_per_cell) - cell_per_block + 1
        nxsteps = (nxblocks - window_blocks) // cells_per_step
        nysteps = (nyblocks - window_blocks) // cells_per_step

        # Compute image-wide hog features for each channel
        hog1 = self.get_hog_features(ch1, orient, pix_per_cell, cell_per_block)
        hog2 = self.get_hog_features(ch2, orient, pix_per_cell, cell_per_block)
        hog3 = self.get_hog_features(ch3, orient, pix_per_cell, cell_per_block)
    
        # Iterate over each x/y block pair
        for xb in range(nxsteps):
            for yb in range(nysteps):
                # caluclate current y/x position and left/top
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract and stack HOG features for this patch
                hog_feat1 = hog1[ypos:ypos + window_blocks,
                                 xpos:xpos + window_blocks].ravel() 
                hog_feat2 = hog2[ypos:ypos + window_blocks,
                                 xpos:xpos + window_blocks].ravel() 
                hog_feat3 = hog3[ypos:ypos + window_blocks,
                                 xpos:xpos + window_blocks].ravel() 
                hog_X = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                # Extract the image patch for this block
                subimg = cv2.resize(search_image[ytop:ytop + window_count, 
                        xleft:xleft + window_count], (64,64)) # TODO: Why is this 64?

                # Get spatial and color features from the image patch
                spatial_X = self.bin_spatial(subimg, spatial_size)
                hist_X = self.color_hist(subimg, hist_bins)
          
                # Stack and flatten features, then scale them
                X = self.X_scaler.transform(
                        np.hstack((spatial_X, hist_X, hog_X)).reshape(1, -1))  
                
                # Predict on the flattened, scaled X
                prediction = self.svc.predict(X)

                # IF a car was detected
                if predcition == 1:
                    # calculate the scaled window size
                    window_size = np.int(window_count * scale)

                    # Calculate the top/bottom/left/right corner points
                    xl = np.int(xleft * scale)
                    xr = xl + window_size
                    yt = np.int(ytop * scale)
                    yb = y_t + ystart + window_size

                    # Create box coordinates with topleft/bottomright points
                    box = ((xl,yt), (xr, yb))

                    # Add box to current car_blocks list
                    self.current_car_blocks.append(box)

    def calculate_heat(self, img):
    # TODO: Call some sort of heat function to clean up noise
        pass

    def detect_cars(self, img):
    # TODO: Call some sort of function to base current detections on history
        pass

    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block, vis=False):
        '''Given an image return the hog features
        vis: set to true to get (features, vis_img) as response
        '''
        # TODO: test this works with vis = false/true
        return hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=False)

    def bin_spatial(self, img, spatial_size=(32, 32)):
        '''Given an img return a resized and flattened vector'''
        return cv2.resize(img, spatial_size).ravel() 

    def color_hist(self, img, hist_bins=32, bins_range=(0, 256)):
        '''Given a 3 channel img return a vectorized histogram of the channels'''
        ch1_hist = np.histogram(img[:,:,0], bins=hist_bins, range=bins_range)
        ch2_hist = np.histogram(img[:,:,1], bins=hist_bins, range=bins_range)
        ch3_hist = np.histogram(img[:,:,2], bins=hist_bins, range=bins_range)
        return np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0]))


def bin_spatial(img, size=(32, 32)):
    return cv2.resize(img, size).ravel() 


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = cv2.imread(os.path.join("test_img", "test3.jpg"))
    vd = VehicleDetection()
    vd.train()
    img = vd.pipeline(img)
    plt.imshow(img)
    plt.show()
