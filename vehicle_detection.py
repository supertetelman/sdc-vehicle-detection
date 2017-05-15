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
from scipy.ndimage.measurements import label

import car_helper
from pipeline import Pipeline
from car_data import CarData


class VehicleDetection(Pipeline):
    '''The Pipeline is designed to detect and track objects over time. Specifically cars.
    The first group of functions are related to training the models used in identification.
    The second group of funtions are related to feature extraction and detecting cars.
    The third group of functions are related to tracking and error correction over time.

    Initialize with <data_file> as a trained model pickle file to load it or True to use the default,
            Setting this fault will result in the <data> ("small" or "big") to load. This may take some time.

    <n> says how many previous cars to remember, unrelated to heatmap, currently unimplemented
    '''
    def __init__(self, data_file=False, data="small", n=1):
        super().__init__() # Initialize folder info, etc.

        # The Simple Vector Machine trained to detect cars
        self.svm = None

        # The scaler trained to scale X input
        self.X_scaler = None

        # The location to save any new data files
        self.data_file = os.path.join(self.results_dir, "detection_models.p")

        # Load pre-trained models or load training data
        if data_file:
            # Use default file if they did not specify
            if isinstance(data_file, str):
                self.load_pickle(data_file) # Updates scaler and svm
            else:
                self.load_pickle(self.data_file)
            self.data = None
        else:
            self.data = CarData(data)
            # Set some HOG specific params for the model 
            self.pix_per_cell = 8 # Number of pixels in an individual HOG cell XXX: Tunable
            self.cell_per_block = 2 # Number of cells in a single block used by HOG XXX: Tunable
            self.orient = 9 # Number of orientation bins. XXX: Tunable


        # fit-sized queue to store box coordinates of cars detected last n imgs
        self.cars =  deque(maxlen = n)

        # The currently identifed car boxes/points
        self.current_cars = None

        # The blocks where a car has currently been detected
        self.current_blocks = None

        # The raw data loaded from self.data and corresponding classes
        self.img_data = None # XXX: gets reset to none after the data split
        self.img_class = None

        # The split train/cv/test data
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None

        # A heatmap that is update each time a new image is processed
        self.heatmap = None # Heatmap image that persists across instance life
        self.threshold = 3 # The threshold at  which if n blocks overlap it is a car XXX: Tunable
        self.heat_dec = -1 # The amount to reduce each pixel for a new heat map XXX: Tunable

        # Model parameters
        self.spatial_size = (32, 32) # size for spacial features XXX: Tunable
        self.hist_bins = 32 # Number of hist_bins to use XXX: Tunable
        self.color = 'YCrCb' # Color space to convert images to XXX: Tunable


    def load_pickle(self, data_file):
        '''Load a pickle file and extract the model data'''
        models = pickle.load(open(data_file, 'rb'))
        self.svm = models['svm']
        self.X_scaler = models['X_scaler']
        self.pix_per_cell = models['pix_per_cell']
        self.orient = models['orient']
        self.cell_per_block = models['cell_per_block']
        assert self.svm is not None
        assert self.X_scaler is not None
        assert isinstance(self.cell_per_block, int)
        assert isinstance(self.orient, int)
        assert isinstance(self.pix_per_cell, int)
        assert self.orient >= 2
        assert self.pix_per_cell >= 1
        assert self.cell_per_block >= 1

    def save_pickle(self, data_file):
        '''Properly save the model data to a pickle file'''
        assert self.svm is not None
        assert self.X_scaler is not None
        pickle.dump({'svm': self.svm, 'X_scaler': self.X_scaler,
            'pix_per_cell': self.pix_per_cell, 'orient': self.orient,
            'cell_per_block': self.cell_per_block }, open(data_file, 'wb'))

    def train(self):
        '''Top level function to create, train, and save the models'''
        print("Initializing models.")
        self.svm = LinearSVC()
        self.X_scaler = StandardScaler()

        print("Reading in all image data")
        self.get_data()

        print("Extracting features from image data")
        self.extract_data_features()

        print("Training the X scaler")
        self.X_scaler.fit(np.asarray(self.X).astype(np.float64))

        print("Scaling the %d X values" %(len(self.X)))
        self.X_scaler.transform(self.X)

        print("Splitting X, y features into train and test")
        self.split_data()

        print("Training the SVM with train data of length %d" %(len(self.train_X)))
        self.svm.fit(self.train_X, self.train_y)

        print("Scoring the SVM")
        acc = self.svm.score(self.test_X, self.test_y)
        print("SVM test accuracy of %0.4f" %acc)

        print("Saving data to %s" %(self.data_file))
        self.save_pickle(self.data_file)

    def get_data(self):
        '''Read in all the files stored in self.data as raw pixels'''
        assert self.data is not None

        # Initialize class variables to empty lists
        self.img_data = []
        self.img_class = []

        # Append all images and classes to the lists
        X, Y = self.data.get_class_data()
        for x, y in zip(X, Y):
            img = cv2.imread(x)
            # TODO: resize image
            assert img is not None
            assert len(img > 0)
            self.img_class.append(y)
            self.img_data.append(img)

    def extract_data_features(self):
        '''Convert the raw pixel data into handy-dandy features.
        All of the raw data is under 300 MB, if we are duplicating this threefold
        we can still expect everything to fit in under 1GB of ram.
        '''
        # Initialize Feature variable
        self.X = []

        # Iterate over all images
        for img in self.img_data:
            # Convert color space
            img = car_helper.convert_img(img, self.color)

            # Compute hog features for each color channel
            hog1 = self.get_hog_features(img[:,:,0], self.orient,
                    self.pix_per_cell, self.cell_per_block)
            hog2 = self.get_hog_features(img[:,:,1], self.orient,
                    self.pix_per_cell, self.cell_per_block)
            hog3 = self.get_hog_features(img[:,:,2], self.orient,
                    self.pix_per_cell, self.cell_per_block)

            # Get spatial, color, and hog features from the image 
            spatial_X = self.bin_spatial(img, self.spatial_size)
            hist_X = self.color_hist(img, self.hist_bins)
            hog_X = np.concatenate((hog1, hog2, hog3)).ravel()

            # Stack and flatten everything into a single feature
            X = np.concatenate((spatial_X, hist_X, hog_X))

            # Set class variables  
            self.X.append(X)

        # Remove the raw data once features have been extraced
        self.img_data = None
        assert len(self.img_class) == len(self.X)

    def split_data(self, test=0.33):
        '''Split data into train/test/validation'''
        count = len(self.X)

        # Split the data into train, cv, and tesst
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.X,
            self.img_class, test_size = test)
        print("train dataset (%0.2f %%, %d), \
            \ntest dataset (%0.2f %%, %d)" %(100 - (100 * test),
                len(self.train_X), 100*test,
                len(self.test_X)))

        # Remove unsplit features and classes
        self.X = None
        self.img_class = None

        #Verify nothing was lost
        assert count == len(self.train_X) + len(self.test_X)

    def pipeline(self, img):
        '''Given an image return an image with boxes drawn around all vehicles
        It is assumed that the incoming image is undistorted.
        '''
        img_shape = img.shape

        # Detect cars in this image
        self.detect_blocks(img)

        # Create a heat map based on the detected car blocks
        self.calculate_heat(img)

        # Use the detected blocks, heat map, and previous data to detect cars
        self.detect_cars(img)

        # Outline the currently detected cars
        img = car_helper.draw_boxes(img, self.current_cars)

        # Return the annoted image
        assert img_shape == img.shape
        return img

    def detect_blocks(self, img, scale=1, debug=False):
        '''Generates hog features for entire img, slides over each window and 
        uses subset of hog features in addition to spatial and color histogram 
        features to predict whether a car is present using pretrained 
        simple vector machine model and pretrained X value scaler.

        HOG will generate a single gradient value for every <self.pix_per_cell> 
            by <self.pix_per_cell> square  Blocks are used to navigate the HOG features
        A total of <window_count> windows will be created
            Each window will be <step_size> cells (<step_size> * self.pix_per_cell pixels) 
                    right/down from the previous window
        '''
        # Verify we have been trained
        assert self.svm is not None
        assert self.X_scaler is not None

        # Initialize the current list of detected car blocks
        self.current_blocks = []

        # Don't scan the horizon for cars; they don't fly yet.
        ystart = 390 # removes most of the horizion XXX: Tunable
        yend = 690 # removes the front of the car XXX: Tunable
        search_img = img[ystart:yend,:,:]

        # Set the number of windows and steps between windows
        window_count = 64 # Total number of windows XXX: Tunable
        step_size = 2 # How man cells to slide right/down for each new window XXX: Tunable

        # Verify image is scaled to 255 not 1
        if np.max(search_img) <= 1:
            search_img = search_img * 255

        # TODO: Automatically  scale up towards the front and down towards the back
        # Scale the image up/down to account for different sized objects up/down the horizon.
        patch_shape = search_img.shape
        if scale != 1: # Save some compute if no scaling is to be done
            search_img = cv2.resize(search_img, (np.int(patch_shape[1]/scale),
                                                 np.int(patch_shape[0]/scale)))

        # Convert from BRG to color
        search_image = car_helper.convert_img(search_img, self.color)

        # Extract individual color channels
        ch1 = search_image[:,:,0]
        ch2 = search_image[:,:,1]
        ch3 = search_image[:,:,2]

        # Compute image-wide hog features for each channel
        hog1 = self.get_hog_features(ch1, self.orient,
                self.pix_per_cell, self.cell_per_block)
        hog2 = self.get_hog_features(ch2, self.orient,
                self.pix_per_cell, self.cell_per_block)
        hog3 = self.get_hog_features(ch3, self.orient,
                self.pix_per_cell, self.cell_per_block)
    
        # Define blocks and steps based on img size
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1 
        nfeat_per_block = self.orient * self.cell_per_block**2

        # Calculate number of steps in the y/x directions and # blocks per window
        window_blocks = (window_count // self.pix_per_cell) - self.cell_per_block + 1
        nxsteps = (nxblocks - window_blocks) // step_size
        nysteps = (nyblocks - window_blocks) // step_size

        # Iterate over each x/y block pair
        for xb in range(nxsteps):
            for yb in range(nysteps):
                # Iterate from 0 to x/y and multipl
                ypos = yb * step_size
                xpos = xb * step_size

                # Calculate positions used in img (hog collapsed px x px cells into single values)
                xleft = xpos * self.pix_per_cell
                ytop = ypos * self.pix_per_cell

                # Extract and stack HOG features for this patch
                hog_feat1 = hog1[ypos:ypos + window_blocks,
                                 xpos:xpos + window_blocks].ravel() 
                hog_feat2 = hog2[ypos:ypos + window_blocks,
                                 xpos:xpos + window_blocks].ravel() 
                hog_feat3 = hog3[ypos:ypos + window_blocks,
                                 xpos:xpos + window_blocks].ravel() 
                hog_X = np.hstack((hog_feat1, hog_feat2, hog_feat3)).flatten()

                # Extract the image patch for this block
                subimg = cv2.resize(search_image[ytop:ytop + window_count, 
                        xleft:xleft + window_count], (64,64)) # TODO: Why is this 64?

                # Get spatial and color features from the image patch
                spatial_X = self.bin_spatial(subimg, self.spatial_size)
                hist_X = self.color_hist(subimg, self.hist_bins)
          
                # Stack and flatten features, then scale them
                X = self.X_scaler.transform(
                        np.hstack((spatial_X, hist_X, hog_X)).reshape(1, -1))  
                
                # Predict on the flattened, scaled X
                prediction = self.svm.predict(X)

                # For debug runs, add every box to the current_blocks list
                if debug: 
                    prediction = True

                # IF a car was detected
                if prediction == 1:
                    # calculate the scaled window size
                    window_size = np.int(window_count * scale)

                    # Calculate the top/bottom/left/right corner points
                    xl = np.int(xleft * scale)
                    xr = xl + window_size
                    yt = np.int(ytop * scale) + ystart
                    yb = yt + window_size

                    # Create box coordinates with topleft/bottomright points
                    box = ((xl,yt), (xr, yb))

                    # Add box to current car_blocks list
                    self.current_blocks.append(box)

    def reset_heat(self, img):
        '''Reset the heatmap image'''
        self.heatmap = np.zeros_like(img).astype(np.float)

    def calculate_heat(self, img, debug=False):
        '''Calculate a heat map based on all the blocks with cars in them'''
        # Generate the heatmap if it does not exist, or decrement all values if it does
        if self.heatmap is None:
            self.reset_heat(img)
        else:
            self.heatmap[self.heatmap > 0] -= self.heat_dec

        # Increment every pixel within a block by 1
        for box in self.current_blocks:
            self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Reset every pixel that did not meet the threshold
        self.heatmap[self.heatmap <= self.threshold] = 0

        # Clip anything with a value over 255
        np.clip(self.heatmap, 0, 255)

        if debug:
            plt.imshow(self.heatmap, cmap='gray')
            plt.show()

    def detect_cars(self, img, debug=False):
        '''Use the heatmap to label cars and generate more accurate bounding boxes'''
        # Initialize cars list
        self.current_cars = []

        # Label cars in the heatmap
        labels = label(self.heatmap)

        # Iterate over each car and calculate box coordinates
        for car in range(1, labels[1] + 1): # add 1 because 0 is the non-car
            car_y, car_x, car_z = (labels[0] == car).nonzero()
            xt = np.min(car_x)
            xb = np.max(car_x)
            yl = np.min(car_y)
            yr = np.max(car_y)
            box = ((xt, yl),(xb, yr))
            self.current_cars.append(box)

        self.cars.extendleft(self.current_cars)

        # Return labels[0] as an image
        if debug:
            return np.asarray(labels[0]).astype(np.float64)

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import glob
    import time

    imgs = glob.glob(os.path.join("test_img",  "*"))
    vd = VehicleDetection()
    vd.train()

    # Iterate over each test image and show each step in the process, then run the whole pipeline.
    i = 0
    for img_file in imgs:
        i += 1
        print(img_file)
        # Create a figure for all 6 images
        f = plt.figure()
        plt.title("Steps of vehicle Detection")

        # Load initial image
        f.add_subplot(2,3,1)
        img = cv2.imread(img_file)
        plt.imshow(img)

        # Show all the Windows we are searching
        f.add_subplot(2,3,2)
        vd.detect_blocks(img, debug=True)
        window_img = car_helper.draw_boxes(img, vd.current_blocks)
        vd.reset_heat(img) # Reset this because the debug data made it bogus
        plt.imshow(window_img)

        # Show what the block detector found
        f.add_subplot(2,3,3)
        vd.detect_blocks(img, debug=False)
        block_img = car_helper.draw_boxes(img, vd.current_blocks)
        plt.imshow(block_img)

        # Detect the image a few times to make the heat map more interesting, then show it
        f.add_subplot(2,3,4)
        vd.calculate_heat(img, debug=False)
        plt.imshow(vd.heatmap, cmap='hot')

        # Show the labels that were detected from the heat map data
        f.add_subplot(2,3,5)
        labels_img = vd.detect_cars(img, debug=True)
        plt.imshow(labels_img, cmap='gray')
        
        # Show the final car outlines
        f.add_subplot(2,3,6)
        final_img = car_helper.draw_boxes(img, vd.current_cars)
        plt.imshow(final_img)
        plt.savefig(os.path.join(vd.results_dir, "%d-debug-img.jpg" %i))
        plt.close()

        # Run through the actual pipeline and time it
        vd.reset_heat(img) # Reset this because the debug data made it bogus
        s = time.time()
        pipeline = vd.pipeline(img)    
        plt.imshow(pipeline)
        plt.title("Vehicles Detected in %0.2f seconds" %(time.time()-s))
        plt.savefig(os.path.join(vd.results_dir, "%d-final-output.jpg" %i))
        plt.close()
