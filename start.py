import cv2
from core.hnfTracker import hnfTracker

# Make webcam consistent
# sudo v4l2-ctl --set-ctrl exposure_auto=1 --device=/dev/video1

# Video capturing instance
# -1 specifies it to get the first avaliable video instance
video_capture = cv2.VideoCapture(1)

# hnfTracker core instance
hnftracker = hnfTracker(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH), video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

while(True):
    # Gets frame from video input
    ret, frame = video_capture.read()

    # Smoothens image
    frame = cv2.medianBlur(frame, 5)

    # Mirrors image
    cv2.flip(frame, 1, frame)

    # Converts to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Draws Region of Interest (ROI) onto frame
    hnftracker.drawn_ROI(frame)

    # Gets mean frame
    gray_mean = hnftracker.extract_mean(frame)

    # Gets thresholdde image
    ret, img_thresh = cv2.threshold(gray, gray_mean, 255, cv2.THRESH_BINARY)

    # Display image
    if (frame is not None):
        cv2.imshow('hnfTracker v1.0 (thresholded)', img_thresh)
        cv2.imshow('hnfTracker v1.0 (original)', frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
video_capture.release()
