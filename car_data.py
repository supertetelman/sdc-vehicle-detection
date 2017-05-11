import numpy as np
import cv2
import os
import glob


class CarData(object):
    def __init__(self, size="small", gti=True, kitti=True, car_cat=[], non_cat=[]):
        '''Read in file names atinitialization time
        size: "small" for small dataset, else big dataset
        gti: flag to use GTI data
        kitti: flag to use KITTI data
        car_cat: a list of strings, name of GTI image type - empty list for default
            list of ints for small dataset
        non_cat: a list of strings, name of directoryes - empty list for default
            list of ints for small dataset
        '''
        self.data_dir = "data"
        self.car_files = []
        self.non_car_files = []

        # Verify vehicle category vars are a lists
        assert isinstance(non_cat, list)
        assert isinstance(car_cat, list)

        if size == "small":
            self.vehicle_data_dir = os.path.join(self.data_dir, "vehicles_smallset")
            self.non_vehicle_data_dir = os.path.join(self.data_dir, "non-vehicles_smallset")

            # Set default categories
            if len(non_cat) <= 0:
                non_cat = [1, 2, 3]
            if len(car_cat) <= 0:
                car_cat = [1, 2, 3]

            # Add all non_car filenames
            for cat in non_cat:
                dname = "notcars%d" %cat
                new_files = self.get_dir_images(self.non_vehicle_data_dir,dname)
                self.non_car_files += new_files
            # Add all car filenames
            for cat in car_cat:
                dname = "cars%d" %cat
                new_files = self.get_dir_images(self.vehicle_data_dir,dname)
                self.car_files += new_files

        else: # Larger dataset
            self.vehicle_data_dir = os.path.join(self.data_dir, "vehicles")
            self.non_vehicle_data_dir = os.path.join(self.data_dir, "non-vehicles")

            # Set default  categories
            if len(non_cat) <= 0:
                non_cat = ["GTI", "Extras"]
            if len(car_cat) <= 0:
                car_cat = ["Far", "Left", "MiddleClose", "Right"]

            # Parse directory names from flags/categories
            dnames = []
            if kitti:
                dnames.append("KITTI_extracted")
            if gti:
                for cat in car_cat:
                    dnames.append("GTI_%s" %cat) 
            # Add all non_car filenames
            for dname in non_cat:
                new_files = self.get_dir_images(self.non_vehicle_data_dir,dname)
                self.non_car_files += new_files
            # Add all car filenames
            for dname in dnames:
                new_files = self.get_dir_images(self.vehicle_data_dir,dname)
                self.car_files += new_files

        # Shuffle dataset # TODO: add a seed value here 
        np.random.shuffle(self.car_files)
        np.random.shuffle(self.non_car_files)

    def get_dir_images(self, top_dir, dname, f_types=["*.jpeg", "*.png", "*.jpg"]):
        '''Given a top_level directory, subdirectory, and file types return glob'''
        file_list = []
        for f_type in f_types:
            path = os.path.join(top_dir, dname, f_type)
            file_list += glob.glob(path)
        return file_list

    def data_generator(self, val):
        '''Generator to return image files'''
        files = self.car_files
        if val == 0:
            files = self.non_car_files

        while True:
            for fname in files:
                img = cv2.imread(fname)
                yield img
    
    def sample(self, n, val):
        '''Create a n-sized sample of cars (1), or non-cars (0)'''
        objs = []
        for obj in self.data_generator(val):
            if n <= 0:
                break
            objs.append(obj)
            n -= 1
        return objs

    def __repr__(self):
        return str({'cars': self.car_files, 'non-cars': self.non_car_files})


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create datasets from different types of images
    datas = []
    car_data = CarData()
    datas.append(car_data)
    car_data_small = CarData("big")
    datas.append(car_data_small)
    car_data_kitti = CarData(size = "big", gti = False)
    datas.append(car_data_kitti)
    car_data_far = CarData(size = "big", car_cat = ["Far"], kitti = False)
    datas.append(car_data_far)
    car_data_left = CarData(size = "big", car_cat = ["Left"], kitti = False)
    datas.append(car_data_left)

    # Create a seperate samples figure for each dataset
    for data in datas:
        car_samples = data.sample(3, 1)
        non_car_samples = data.sample(3,0)

        # Create a 3x2 figure with cars on top, non_cars on bottom
        f = plt.figure("Sample data")
        i = 1
        for sample in car_samples + non_car_samples:
            f.add_subplot(2, 3,i)
            plt.imshow(sample)
            i += 1
        plt.show()
