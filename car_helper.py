'''This module contains several common helper functions and classes.
These helpers are used throught the modules in the CarWorld package
'''
import cv2


def overlay_img(orig, update):
    '''Overlay an image over another image'''
    return cv2.addWeighted(orig, 1, update, 0.3, 0)
