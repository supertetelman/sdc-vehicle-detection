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

import pdb

import matplotlib.pyplot as plt

# TODO: Implement using Deep Learning rather than SVM
class VehicleDetection(Pipeline):
    '''The Pipeline is designed to detect and track objects over time. Specifically cars.
    The first group of functions are related to training the models used in identification.
    The second group of functions are related to feature extraction and detecting cars.
    The third group of functions are related to tracking and error correction over time.

    Initialize with <data_file> as a trained model pickle file to load it or True to use the default,
            Setting this fault will result in the <data> ("small" or "big") to load. This may take some time.

    <n> says how many previous cars to remember, unrelated to heatmap, currently unimplemented

    TODO: Move all feature extraction to a more testable/central spot.
    TODO: Differentiate different cars with different colors for advanced tracking.
    '''
    def __init__(self, data_file=False, data="small", n=15):
        super().__init__() # Initialize folder info, etc.
        ###### Initialize Instance Variables
        # The location to save any new data files
        self.data_file = os.path.join(self.results_dir, "detection_models.p")

        # The raw data loaded from self.data and corresponding classes
        self.img_data = None # XXX: gets reset to none after the data split
        self.img_class = None

        # The split train/cv/test data
        self.X = None
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None

        # The Simple Vector Machine trained to detect cars
        self.svm = None

        # The scaler trained to scale X input
        self.X_scaler = None

        # The blocks where a car has currently been detected
        self.current_blocks = None

        # The currently identified car boxes (based on blocks))
        self.current_cars = None

        # fit-sized queue to store box coordinates of cars detected last n imgs
        self.cars =  deque(maxlen = n) # TODO: Currently storing the cars in here and not using them

        # Store the last n heatmaps where the car detection met the threshold and average them into heatmap
        self.heatlist = deque(maxlen = n) 
        self.heatmap = None # Heatmap image that persists across instance life

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
            ###### Start of tunable params ######
            ###### ###### Model dependent tunables
            # Set some HOG specific params for the model 
            self.pix_per_cell = 8 # Number of pixels in an individual HOG cell XXX: Tunable
            self.cell_per_block = 2 # Number of cells in a single block used by HOG XXX: Tunable
            self.orient = 8 # Number of orientation bins. XXX: Tunable

            # Channels to use for hog features
            self.hog_channels = [0, 1, 2] # XXX: Tunable

            # Channels to use for histogram colors
            self.hist_channels = [0, 1, 2] # XXX: Tunable

            # Model parameters
            self.spatial_size = (16, 16) # size for spacial features XXX: Tunable - makes sense to make this train_size
            self.hist_bins = 64 # Number of hist_bins to use XXX: Tunable
            self.color = 'YCrCb' # Color space to convert images to XXX: Tunable

            # Disable features
            self.hist_dis = False # XXX: Tunable
            self.spatial_dis = False # XXX: Tunable
            self.hog_dis = True # XXX: Tunable

            # The shape to resize training images to
            self.train_shape = (64, 64, 3) # XXX: Tunable
            self.train_blur = False # XXX: Tunable, blur kernel applied to all images

        ######m ###### Non-Training/Model dependent tunables
        self.threshold = 2.3 # The threshold at  which if n blocks overlap it is a car XXX: Tunable

        # Sliding window variables
        self.window_size = 64 # Length/Width of windows XXX: Tunable
        self.step_size = 2 # This did well at 2 and 3 # How many cells to slide right/down for each new window XXX: Tunable
        ###### End of tunable params ######

    def validate_data(self):
        '''Validate class data is correct'''
        assert self.svm is not None
        assert self.X_scaler is not None
        assert isinstance(self.cell_per_block, int)
        assert isinstance(self.orient, int)
        assert isinstance(self.pix_per_cell, int)
        assert self.orient >= 2
        assert self.pix_per_cell >= 1
        assert self.cell_per_block >= 1
        assert isinstance(self.hog_channels, list) and len(self.hist_channels) > 0 and set(self.hog_channels).issubset([0,1,2])
        assert isinstance(self.hist_channels, list) and len(self.hist_channels) > 0 and set(self.hist_channels).issubset([0,1,2])
        assert isinstance(self.spatial_dis, bool)
        assert isinstance(self.hist_dis, bool) 
        assert isinstance(self.hog_dis, bool) 
        assert not (self.hog_dis and self.spatial_dis and self.hist_dis)
        assert self.color is not None
        assert self.spatial_size is not None and len(self.spatial_size) == 2
        assert self.hist_bins is not None and self.hist_bins >= 1
        assert self.train_blur is not None and (self.train_blur == False or len(self.train_blur) == 2)
        assert self.train_shape is not None and len(self.train_shape) == 3

    def load_pickle(self, data_file):
        '''Load a pickle file and extract the model data'''
        print("Loading vehicle detection data from %s" %data_file)
        models = pickle.load(open(data_file, 'rb'))
        self.svm = models['svm']
        self.X_scaler = models['X_scaler']
        self.pix_per_cell = models['pix_per_cell']
        self.orient = models['orient']
        self.cell_per_block = models['cell_per_block']
        self.hog_channels = models['hog_channels']
        self.hist_channels = models['hist_channels']
        self.hist_dis = models['hist_dis']
        self.spatial_dis = models['spatial_dis']
        self.hog_dis = models['hog_dis']
        self.color = models['color']
        self.hist_bins = models['hist_bins']
        self.spatial_size = models['spatial_size']
        self.train_blur = models['train_blur']
        self.train_shape = models['train_shape']
        self.validate_data()

    def save_pickle(self, data_file):
        '''Properly save the model data to a pickle file'''
        self.validate_data()
        pickle.dump({ 'svm': self.svm, 'X_scaler': self.X_scaler,
            'pix_per_cell': self.pix_per_cell, 'orient': self.orient,
            'cell_per_block': self.cell_per_block, 
            'hog_channels': self.hog_channels, 
            'hist_dis': self.hist_dis, 'spatial_dis': self.spatial_dis,
            'hog_dis': self.hog_dis,
            'hist_channels': self.hist_channels,
            'color': self.color, 'spatial_size': self.spatial_size,
            'hist_bins': self.hist_bins, 'train_blur': self.train_blur,
            'train_shape': self.train_shape }, open(data_file, 'wb'))

    def train(self):
        '''Top level function to create, train, and save the models'''
        print("Initializing models.")
        self.init_models()
        print("Reading in all image data")
        self.get_data()
        print("Extracting features from image data")
        self.extract_data_features()
        print("Training the X scaler and scaling all %d X values" %(len(self.X)))
        self.scale_data()
        print("Splitting X, y features into train and test")
        self.split_data()
        print("Training the SVM with train data of length %d" %(len(self.train_X)))
        self.train_model()
        print("Scoring the SVM")
        self.score_model()
        print("Saving data to %s" %(self.data_file))
        self.save_pickle(self.data_file)

    def init_models(self):
        '''Initialize Models'''
        self.svm = LinearSVC()
        self.X_scaler = StandardScaler()

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
            # Resize everything to be consistent
            if img.shape != self.train_shape:
                img = cv2.resize(img, self.train_shape[0:2])
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
            # Convert color space and apply blur
            img = car_helper.convert_img(img, self.color, src="BGR",
                    blur=self.train_blur) # training images are read with cv2 and thus BGR

            # Compute hog features for each color channel
            hogs = self.get_hog_features(img, self.hog_channels, self.orient,
                    self.pix_per_cell, self.cell_per_block, disabled=self.hog_dis)

            # Get spatial, color, and hog features from the image 
            spatial_X = self.bin_spatial(img, self.spatial_size,
                    disabled=self.spatial_dis)
            hist_X = self.color_hist(img, self.hist_channels,
                    self.hist_bins, disabled=self.hist_dis)
            hog_X = self.concat_ftrs(hogs)

            # Stack and flatten everything into a single feature
            X = self.get_enabled_ftrs((spatial_X, hist_X, hog_X))

            # Set class variables  
            self.X.append(X)

        # Remove the raw data once features have been extracted
        self.img_data = None
        assert len(self.img_class) == len(self.X)
        self.print_data_stats()

    def print_data_stats(self):
        '''print some data about the training features'''
        print("color histogram (%d) features disabled: %r" %(self.hist_bins,
                self.hist_dis))
        print("spatial features (%s) disabled: %r" %(str(self.spatial_size),
                self.spatial_dis))
        print("HOG features size-%d orient-%d disabled: %r" %(self.window_size,
                self.orient, self.hog_dis))
        if self.X is not None:
            print("Feature shape: %s color: %s" %(str(self.X[0].shape), self.color))
            print("Total # of samples: %d" %len(self.X))

    def scale_data(self):
        '''Fit and scale later
        TODO: Update this to better suit unit tests
        '''
        self.X_scaler.fit(np.asarray(self.X).astype(np.float64))
        self.X_scaler.transform(self.X)

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

    def train_model(self):
        '''Train the model on the loaded/fit data'''
        self.svm.fit(self.train_X, self.train_y)

    def score_model(self):
        '''Score the trained model on the loaded/split data'''
        acc = self.svm.score(self.train_X, self.train_y)
        print("SVM train accuracy of %0.4f" %acc)
        acc = self.svm.score(self.test_X, self.test_y)
        print("SVM test accuracy of %0.4f" %acc)

    def pipeline(self, img, video=True, blocks=False, debug_all=False):
        '''Given an image return an image with boxes drawn around all vehicles
        It is assumed that the incoming image is undistorted.
        '''
        original_img = np.copy(img)
        img_shape = img.shape
        if debug_all:
            imgs = {'img': original_img}

        # Convert from BGR (for cv2 images) or RGB (for video) to self.color
        src = 'RGB' if video else 'BGR'
        img = car_helper.convert_img(img, self.color, src, blur=self.train_blur)
        if debug_all:
            imgs[self.color] = img

        # Reset detected blocks
        self.current_blocks = []

        # Detect cars in this image
        self.scaling_detect_blocks(img)
        if debug_all:
            blocks_img = car_helper.draw_boxes(original_img,
                    self.current_blocks,  (52, 255, 20) ,thick=10)
            imgs['blocks'] = blocks_img
        elif blocks:
            return car_helper.draw_boxes(original_img, self.current_blocks,
                    (52, 255, 20) ,thick=15)

        # Create a heat map based on the detected car blocks
        self.calculate_heat(img)
        if debug_all:
            imgs['heat'] = self.heatmap

        # Use the detected blocks, heat map, and previous data to detect cars
        self.detect_cars(img)

        # Outline the currently detected cars
        img = car_helper.draw_boxes(original_img, self.current_cars,  (52, 255, 20) ,thick=15)
        if debug_all:
            imgs['final'] = img
            return imgs

        # Return the annoted image
        assert img_shape == img.shape
        return img

    def scaling_detect_blocks(self, img, debug=False):
        '''Iterate from top to bottom of the "lane range" and detect cars of
        increasing size. Return the number of windows searched.
        '''
        # TODO: Make this less hard coded but still effective/intuitive.
        count = self.detect_blocks(img, debug=debug, ystart=360, yend=480,
                scale=1)
        count = self.detect_blocks(img, debug=debug, ystart=400, yend=615,
                scale=2, count=count)
        count = self.detect_blocks(img, debug=debug, ystart=400, yend=720,
                scale=3, count=count)
        return count # Total number of windows searched

    def detect_blocks(self, img, ystart=None, yend=None, scale=1, count=0, debug=False):
        '''Generates hog features for entire img, slides over each window and 
        uses subset of hog features in addition to spatial and color histogram 
        features to predict whether a car is present using pretrained 
        simple vector machine model and pretrained X value scaler.

        Returns the number of windows searched.

        HOG will generate a single gradient value for every <self.pix_per_cell> 
            by <self.pix_per_cell> square  Blocks are used to navigate the HOG features
        A total of <window_size> windows will be created
            Each window will be <step_size> cells (<step_size> * self.pix_per_cell pixels) 
                    right/down from the previous window
        '''
        # Verify we have been trained
        assert self.svm is not None
        assert self.X_scaler is not None
        assert ystart is not None and yend is not None
        assert yend <= img.shape[0] and ystart >= 0

        # Snip out the suitable section of the image (non-horizon))
        img = img[ystart:yend,:,:]

        # Don't operate on a 0 chunk of image (this can be the case with transparent backgrounds)
        if img.shape[0] == 0 or img.shape[1] == 0:
            print("WARN: Invalid image size passed to detect_blocks")
            return count # TODO: clean this up

        # Verify image is scaled to 255 not 1
        if np.max(img) <= 1:
            img = img * 255

        # Scale the image up/down to account for different sized objects up/down the horizon.
        if scale != 1: # Save some compute if no scaling is to be done
            img = cv2.resize(img, (np.int(img.shape[1]/scale),
                                                 np.int(img.shape[0]/scale)))

        # Compute image-wide hog features for each channel
        hogs = self.get_hog_features(img, self.hog_channels, self.orient,
                self.pix_per_cell, self.cell_per_block, disabled=self.hog_dis)
    
        # Define blocks and steps based on img size
        # TODO: Increment these by +2 to get the last part of the screen, then fix the bug where this causes sizing issues in the transform
        nxblocks = (img.shape[1] // self.pix_per_cell) - self.cell_per_block + 1 # Round up +1, account for exclusive range +1
        nyblocks = (img.shape[0] // self.pix_per_cell) - self.cell_per_block + 1 # Round up +1, account for exclusive range +1
        nfeat_per_block = self.orient * self.cell_per_block**2

        # Calculate number of steps in the y/x directions and # blocks per window
        window_blocks = (self.window_size // self.pix_per_cell) - self.cell_per_block + 1
        nxsteps = (nxblocks - window_blocks) // self.step_size + 1 # Round up
        nysteps = (nyblocks - window_blocks) // self.step_size + 1 # Round up

        # Iterate over each x/y block pair
        for xb in range(nxsteps):
            for yb in range(nysteps):
                # Increment window count
                count += 1

                # Iterate from 0 to x/y and multipl
                ypos = yb * self.step_size
                xpos = xb * self.step_size

                # Calculate positions used in img (hog collapsed px x px cells into single values)
                xleft = xpos * self.pix_per_cell
                ytop = ypos * self.pix_per_cell

                # Extract and stack HOG features for this patch from pre-calculated values
                if hogs is None:
                    hog_features = None
                else:
                    hog_features = []
                    for hog_ftr in hogs:
                        hog_ftr = hog_ftr[ypos:ypos + window_blocks,
                                  xpos:xpos + window_blocks].ravel()
                        hog_features.append(hog_ftr)

                # Extract the image patch for this block and resize it to model size
                subimg = cv2.resize(img[ytop:ytop + self.window_size,
                        xleft:xleft + self.window_size], self.train_shape[0:2])

                # Get hog, spatial and color features from the image patch
                spatial_X = self.bin_spatial(subimg, self.spatial_size,
                        disabled=self.spatial_dis)
                hist_X = self.color_hist(subimg, self.hist_channels,
                        self.hist_bins, disabled=self.hist_dis)
                hog_X = self.concat_ftrs(hog_features)

                # Stack and flatten features, then scale them
                X = self.X_scaler.transform(self.get_enabled_ftrs(
                            (spatial_X, hist_X, hog_X)).reshape(1, -1))
                
                # Predict on the flattened, scaled X
                prediction = self.svm.predict(X)

                # For debug runs, add every box to the current_blocks list
                if debug: 
                    prediction = True

                # If a car was detected
                if prediction == 1:
                    # calculate the scaled window size
                    window_size = np.int(self.window_size * scale)

                    # Calculate the top/bottom/left/right corner points
                    xl = np.int(xleft * scale)
                    xr = xl + window_size
                    yt = np.int(ytop * scale) + ystart
                    yb = yt + window_size

                    # Create box coordinates with topleft/bottomright points
                    box = ((xl,yt), (xr, yb))

                    # Add box to current car_blocks list
                    self.current_blocks.append(box)
        return count

    def reset_heat(self, img):
        '''Reset the heatmap image'''
        self.heatmap = np.zeros_like(img).astype(np.float)

    def calculate_heat(self, img, debug=False):
        '''Calculate a heat map based on all the blocks with cars in them'''
        # Create a zero image size of <imgs>
        heatmap = np.zeros_like(img).astype(np.float)

        # Increment every pixel within a block by 1
        for box in self.current_blocks:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Add the new heatmap to the last-n list of heatmaps
        self.heatlist.appendleft(heatmap)

        # Update the heatmap to be an average of the past n heatmaps
        self.heatmap = np.average(self.heatlist, axis=0).astype(np.float)

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

        self.cars.appendleft(self.current_cars)

        # Return labels[0] as an image
        if debug:
            return np.asarray(labels[0]).astype(np.float64)

    def get_hog_features(self, img, channels, orient, pix_per_cell,
            cell_per_block, vis=False, disabled=False):
        '''Given an <img> return a list of hog features for the specified <channels>
        vis: set to true to get (features, vis_img) as response
        '''
        if disabled:
            return None
        hogs = []
        for ch in channels:
            hog_ftr = hog(img[:,:,ch], orientations=orient,
                    pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block),
                    transform_sqrt=True,
                    visualise=vis, feature_vector=False)
            hogs.append(hog_ftr)
        return hogs

    def bin_spatial(self, img, spatial_size=(32, 32), debug=False, disabled=False):
        '''Given an img return a resized and flattened vector
        debug returns a displayable 2D image rather than flat X vector.
        '''
        if disabled:
            return None
        ftr = cv2.resize(img, spatial_size)
        if debug:
            return ftr
        return ftr.ravel()

    def color_hist(self, img, channels, hist_bins=32,
            bins_range=(0, 256), disabled=False, debug=False):
        '''Given a 3 channel img return a vectorized histogram of the channels'''
        if disabled:
            return None
        hist_features = []
        for ch in channels:
            hist = np.histogram(img[:,:,ch], bins=hist_bins, range=bins_range)
            if debug: # we need all the data in debug
                hist_features.append(hist)
                continue
            hist_features.append(hist[0])
        if debug:
            return  hist_features
        return self.concat_ftrs(hist_features, ravel=False)

    def concat_ftrs(self, feature_list, ravel=True):
        '''Provide a consistent way of concatenating and flattening feature lists'''
        if feature_list is None: # Used for HOG features
            return None
        ftrs = np.concatenate(feature_list)
        if ravel:
            return ftrs.ravel()
        return ftrs

    def get_enabled_ftrs(self, feature_list):
        '''Given a list of features remove the disabled None features and concat'''
        valid_features = list(filter(lambda x: x is not None, feature_list))
        return self.concat_ftrs(valid_features)


# Test code and writeup supporting code
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import glob
    import time

    # Simple training/debug options
    do_hog = True
    do_features = True
    pretrained = True
    size = "big"

    imgs = glob.glob(os.path.join("test_img",  "*"))

    vd = VehicleDetection(pretrained, size)
    if not pretrained:
        vd.train()
    vd.print_data_stats()

    # Explore the HOG Feature params
    if do_hog:
        img_file = imgs[1]
        img = cv2.imread(img_file)
        cv2.imwrite(os.path.join(vd.results_dir, "test-image.jpg"), img)
        color_img = car_helper.convert_img(img, vd.color, src='BGR', blur=vd.train_blur)
        cv2.imwrite(os.path.join(vd.results_dir, "test-image-color.jpg"), color_img)

        # Explore a range of HOG orient values
        for orient in range(6,12):
            # Compute hog features for each color channel and write it to an image
            hogs_debug = vd.get_hog_features(color_img, [0, 1, 2], orient,
                    vd.pix_per_cell, vd.cell_per_block, vis=True)
            for idx in range(0, 3):
                hog_img = hogs_debug[idx][1]
                cv2.imwrite(os.path.join(vd.results_dir, "hog-ch%d-orient%d.jpg" %(idx, orient)), hog_img*255)

    # Iterate over each test image and show each step in the process, then run the whole pipeline.
    i = 0
    for img_file in imgs:
        print(img_file)
        i += 1
        # Create a figure for all 6 images
        f = plt.figure()
        plt.title("Feature Extraction Steps")

        # Load initial image
        img = cv2.imread(img_file)

        # Create a 2x3 plot with original img, spatial features, hist features and hog features
        f.add_subplot(3,4,1)
        plt.imshow(img)

        # Conver img to different color_space
        f.add_subplot(3,4,2)
        color_img = car_helper.convert_img(img, vd.color, src='BGR', blur=vd.train_blur)
        plt.imshow(color_img)

        if do_features: # Quick flag to remove timely debug
            # Plot spatial features
            spatial_X = vd.bin_spatial(color_img, vd.spatial_size, debug=True)
            f.add_subplot(3,4,3)
            plt.imshow(spatial_X)

            # Plot seperated color_histogram geatures
            hist_features = vd.color_hist(color_img, vd.hist_channels, vd.hist_bins, debug=True)
            for idx in range(0, len(vd.hist_channels)):
                f.add_subplot(3,4,5 + idx)
                # Format histogram data
                bin_edges = hist_features[idx][1]
                bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges) - 1])/2
                plt.bar(bin_centers, hist_features[idx][0])

            # Compute hog features for each color channel and add it to subplot
            # Get a list of each hog feature and a corresponding visualization
            hogs_debug = vd.get_hog_features(color_img, vd.hog_channels, vd.orient,
                    vd.pix_per_cell, vd.cell_per_block, vis=True)

            # Plot each visualization and add the features to a list
            hogs = []
            for idx in range(0, len(vd.hog_channels)):
                f.add_subplot(3,4,9 + idx)
                hogs.append(hogs_debug[idx][0])
                plt.imshow(hogs_debug[idx][1])

            # Combine color_hist_features
            f.add_subplot(3,4,8)        
            hist_X = vd.concat_ftrs((hist_features[0][0], hist_features[1][0], hist_features[2][0]))
            plt.plot(hist_X) 

            # Combine hog features and plot them
            f.add_subplot(3,4,12)
            hog_X = vd.concat_ftrs((hogs[0][0], hogs[1][0], hogs[2][0]))
            plt.plot(hog_X)

            # Show X_scaled feature image
            f.add_subplot(3,4,3)
            # TODO: This debug is working on a full image, the transform only works on output from the vd.train_shape image
            # scaled_ftrs = vd.X_scaler.transform(vd.concat_ftrs((spatial_X, hist_X, hog_X)).reshape(1, -1))
            # plt.plot(scaled_ftrs)
            plt.savefig(os.path.join(vd.results_dir, "%d-debug-features.jpg" %i))
            plt.close()

            # Create a 2x3 image with original img and block/car/heatmap/labels detection
            f = plt.figure()
            plt.title("Steps of vehicle Detection")
            f.add_subplot(2,3,1)
            plt.imshow(img)

            # Show all the Windows we are searching
            f.add_subplot(2,3,2)
            vd.current_blocks = [] # Reset this
            window_count = vd.scaling_detect_blocks(img, debug=True)
            window_img = car_helper.draw_boxes(img, vd.current_blocks)
            cv2.imwrite(os.path.join(vd.results_dir, "%d-debug-windows.jpg"%i), window_img)
            vd.reset_heat(img) # Reset this because the debug data made it bogus
            plt.imshow(window_img)

            # Show all the Windows we are searching, but with bigger gaps to make the scaling visible
            vd.current_blocks = [] # Reset this
            tmp = vd.step_size # Save this
            vd.step_size = 100 # Set this to something big to make the block size easier to see
            vd.scaling_detect_blocks(img, debug=True)
            window_size_img = car_helper.draw_boxes(img, vd.current_blocks)
            cv2.imwrite(os.path.join(vd.results_dir, "%d-debug-windows_size.jpg"%i), window_size_img)
            vd.reset_heat(img) # Reset this because the debug data made it bogus
            vd.step_size = tmp # Reset this

            # Show what the block detector found
            f.add_subplot(2,3,3)
            vd.current_blocks = [] # Reset this
            vd.scaling_detect_blocks(img, debug=False)
            block_img = car_helper.draw_boxes(img, vd.current_blocks)
            cv2.imwrite(os.path.join(vd.results_dir, "%d-debug-blocks.jpg"%i), block_img)
            plt.imshow(block_img)

            # Detect the image a few times to make the heat map more interesting, then show it
            f.add_subplot(2,3,4)
            vd.scaling_detect_blocks(img, debug=False)
            vd.scaling_detect_blocks(img, debug=False)
            vd.calculate_heat(img, debug=False)
            cv2.imwrite(os.path.join(vd.results_dir, "%d-debug-heat.jpg"%i), vd.heatmap)
            plt.imshow(vd.heatmap, cmap='hot')

            # Show the labels that were detected from the heat map data
            f.add_subplot(2,3,5)
            labels_img = vd.detect_cars(img, debug=True)
            cv2.imwrite(os.path.join(vd.results_dir, "%d-debug-labels.jpg"%i), labels_img)
            plt.imshow(labels_img, cmap='gray')
            
            # Show the final car outlines
            f.add_subplot(2,3,6)
            final_img = car_helper.draw_boxes(img, vd.current_cars)
            plt.imshow(final_img)
            plt.savefig(os.path.join(vd.results_dir, "%d-debug-img.jpg" %i))
            plt.close()
        else:
            plt.close()
            window_count = -1
        # Run through the actual pipeline and time it
        vd.reset_heat(img) # Reset this because the debug data made it bogus
        s = time.time()
        pipeline = vd.pipeline(img, blocks = False, video = False)
        plt.imshow(pipeline)
        plt.title("Vehicles Detected in %0.2f seconds. Searched over %d Windows." %(time.time()-s, window_count))
        plt.savefig(os.path.join(vd.results_dir, "%d-final-output.jpg" %i))
        plt.close()
