#!/usr/bin/python
import cv2
import numpy as np


class hnfTracker:
    # Constructor
    def __init__(self, max_width, max_height):
        
        # Image dimension
        self.IM_MAX_WIDTH = max_width
        self.IM_MAX_HEIGHT = max_height

        # [The green boxes]
        # The region of interests (ROI) used to calculate
        # the mean value to threshold(adjust to your likings)
        self.ROI_BOX_CONST = 5 # constant to define ROI's width and height
        self.ROI_NO = 5 # number of regions
        self.ROI_COORD = [] # List of coordinates

        # Coordinates of the ROI (green box)
        self.ROI_COORD.append((((int(max_width/1.3)-self.ROI_BOX_CONST),int(max_height/2)-self.ROI_BOX_CONST), ((int(max_width/1.3)+self.ROI_BOX_CONST),int(max_height/2)+self.ROI_BOX_CONST)))
        self.ROI_COORD.append((((int(max_width/1.25)-self.ROI_BOX_CONST),int(max_height/5)-self.ROI_BOX_CONST), ((int(max_width/1.25)+self.ROI_BOX_CONST),int(max_height/5)+self.ROI_BOX_CONST)))
        self.ROI_COORD.append((((int(max_width/1.2)-self.ROI_BOX_CONST),int(max_height/3.25)-self.ROI_BOX_CONST), ((int(max_width/1.2)+self.ROI_BOX_CONST),int(max_height/3.25)+self.ROI_BOX_CONST)))
        self.ROI_COORD.append((((int(max_width/1.55)-self.ROI_BOX_CONST),int(max_height/2.35)-self.ROI_BOX_CONST), ((int(max_width/1.55)+self.ROI_BOX_CONST),int(max_height/2.35)+self.ROI_BOX_CONST)))
        self.ROI_COORD.append((((int(max_width/1.4)-self.ROI_BOX_CONST),int(max_height/1.7)-self.ROI_BOX_CONST), ((int(max_width/1.4)+self.ROI_BOX_CONST),int(max_height/1.7)+self.ROI_BOX_CONST)))

        # [The red boxes]
        # Hand Region
        # To isolate hand region where we'll track hand
        self.HAND_REGION = [(int(max_width-(max_width/2)),int(0)), (int(max_width),int(max_height/1.25))]
        
        # Has the mean value been set
        self.MEAN_VAL_SET = False

    # Draws boxes around the interested regions
    def drawn_ROI(self, im_input):
        # Hand region
        cv2.rectangle(im_input, self.HAND_REGION[0], self.HAND_REGION[1], (0, 0, 255), 1)

        # ROI region
        if self.MEAN_VAL_SET is False:
            for x in range(0, self.ROI_NO):
                cv2.rectangle(im_input, self.ROI_COORD[x][0], self.ROI_COORD[x][1], (0, 255, 0), 1)


    # Gets the mean value of the frames
    def extract_mean(self, im_input):
        # Get the individual images and append them to a list
        im_ROI = []

        for x in range(0, self.ROI_NO):
            # To crop images its img[y:y+h, x:x+w]
            im_ROI.append(im_input[self.ROI_COORD[x][0][1]:self.ROI_COORD[x][1][1], self.ROI_COORD[x][0][0]:self.ROI_COORD[x][1][0]])

        # Calculate mean
        return np.mean(np.median([im_ROI[x].copy() for x in range(0, self.ROI_NO)], axis=0))

    # Sets the mean value of the frames
    def set_mean(self, mean_val):
        self.MEAN_VAL_SET = True
        self.IM_MEAN = mean_val

    # Gets the stored mean value
    def get_mean(self):
        if self.MEAN_VAL_SET is True:
            return self.IM_MEAN
        return None

    # Resets mean value
    def reset(self):
        self.MEAN_VAL_SET = False
        self.IM_MEAN = None


