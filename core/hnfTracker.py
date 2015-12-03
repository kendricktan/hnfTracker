#!/usr/bin/python
import cv2
import numpy as np


class hnfTracker:
    # Constructor
    def __init__(self, max_width, max_height):
        self.im_raw = None

        # Image dimension
        self.IM_MAX_WIDTH = max_width
        self.IM_MAX_HEIGHT = max_height

        # The region of interests (ROI) (adjust to your likings)
        self.ROIbox_const = 5 # constant to define ROI's width and height
        self.ROI_no = 7 # number of regions
        self.ROI_coord = [] # List of coordinates

        # Coordinates of the ROI
        self.ROI_coord.append((((int(max_width/2)-self.ROIbox_const),int(max_height/2)-self.ROIbox_const), ((int(max_width/2)+self.ROIbox_const),int(max_height/2)+self.ROIbox_const)))
        self.ROI_coord.append((((int(max_width/1.95)-self.ROIbox_const),int(max_height/5)-self.ROIbox_const), ((int(max_width/1.95)+self.ROIbox_const),int(max_height/5)+self.ROIbox_const)))
        self.ROI_coord.append((((int(max_width/2.5)-self.ROIbox_const),int(max_height/3.25)-self.ROIbox_const), ((int(max_width/2.5)+self.ROIbox_const),int(max_height/3.25)+self.ROIbox_const)))
        self.ROI_coord.append((((int(max_width/1.95)-self.ROIbox_const),int(max_height/1.35)-self.ROIbox_const), ((int(max_width/1.95)+self.ROIbox_const),int(max_height/1.35)+self.ROIbox_const)))
        self.ROI_coord.append((((int(max_width/2.7)-self.ROIbox_const),int(max_height/1.7)-self.ROIbox_const), ((int(max_width/2.7)+self.ROIbox_const),int(max_height/1.7)+self.ROIbox_const)))
        self.ROI_coord.append((((int(max_width/1.6)-self.ROIbox_const),int(max_height/1.3)-self.ROIbox_const), ((int(max_width/1.6)+self.ROIbox_const),int(max_height/1.3)+self.ROIbox_const)))
        self.ROI_coord.append((((int(max_width/1.6)-self.ROIbox_const),int(max_height/3.75)-self.ROIbox_const), ((int(max_width/1.6)+self.ROIbox_const),int(max_height/3.75)+self.ROIbox_const)))

    # Return boxes around the ROI in the frame
    # This is to provide user feedback
    # so the user knows where the ROI are
    def drawn_ROI(self, im_input):
        for x in range(0, self.ROI_no):
            cv2.rectangle(im_input, self.ROI_coord[x][0], self.ROI_coord[x][1], (0, 255, 0), 1)


    # Analyzes frames and gets the median value
    # in order to perform gaussian blurring
    # Median value is then user to get mean value
    def extract_mean(self, im_input):
        # Get the individual images and append them to a list
        im_ROI = []

        for x in range(0, self.ROI_no):
            im_ROI.append(im_input[self.ROI_coord[x][0][0]:self.ROI_coord[x][1][0], self.ROI_coord[x][0][1]:self.ROI_coord[x][1][1]])

        # Calculate mean and median
        return np.mean(np.median([im_ROI[x].copy() for x in range(0, self.ROI_no)], axis=0))


