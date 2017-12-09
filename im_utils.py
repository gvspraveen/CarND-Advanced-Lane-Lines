import cv2

def read_image(path):
    """
    Reads the image from path and converts to rgb space
    :param path:
    :return:
    """
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def convert_gray(cn2_rgb_image):
    return cv2.cvtColor(cn2_rgb_image, cv2.COLOR_RGB2GRAY)

