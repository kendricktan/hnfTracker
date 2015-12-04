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

    # Returns hand region of the image provided
    def get_hand_region(self, im_input):
        return im_input[self.HAND_REGION[0][1]:self.HAND_REGION[1][1], self.HAND_REGION[0][0]:self.HAND_REGION[1][0]]

    # Smoothens and optimizes image for
    # analyzing later
    def optimize_image(self, im_input):
        return cv2.bilateralFilter(im_input, 9, 75, 75)

    # Analyzes image and draws the crucial
    # points it found on the raw image
    # that resembles a hand
    def analyze_image(self, im_thresh, im_raw):
        # Gets the contours
        img_thresh, contours, hierarchy = cv2.findContours(im_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Maxiumum area
        # used to find largest contour
        m_area = 0
        m_x = None # Maximum x (largest contour)

        # Find largest contour
        for x in range(len(contours)):
            cur_cont = contours[x]
            cur_area = cv2.contourArea(cur_cont)

            if(cur_area > m_area):
                m_area = cur_area
                m_x = x

        try:
            # Gets largest contour
            max_cont = contours[m_x]

            # Gets the convex hull from contour
            hull = cv2.convexHull(max_cont, returnPoints = False)

            # Gets the defects
            defects = cv2.convexityDefects(max_cont, hull)

            # Draw the current (largest) contour and its hull
            # onto the input image
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(max_cont[s][0])
                end = tuple(max_cont[e][0])
                far = tuple(max_cont[f][0])
                cv2.line(im_raw[self.HAND_REGION[0][1]:self.HAND_REGION[1][1], self.HAND_REGION[0][0]:self.HAND_REGION[1][0]],start,end,(0,255,0),2)
                cv2.circle(im_raw[self.HAND_REGION[0][1]:self.HAND_REGION[1][1], self.HAND_REGION[0][0]:self.HAND_REGION[1][0]],far,5,(0,0,255),2)

        except:
            pass


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


