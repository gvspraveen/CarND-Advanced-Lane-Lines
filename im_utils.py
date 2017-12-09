import cv2

def read_image(path):
    return cv2.imread(path)

def convert_gray(cn2_bgr_image):
    return cv2.cvtColor(cn2_bgr_image, cv2.COLOR_BGR2GRAY)

