import cv2
import numpy as np

def binary_image(img, thresh_min, thresh_max):
    binary_output = np.zeros_like(img)
    binary_output[(img >= thresh_min) & (img <= thresh_max)] = 1
    return binary_output

# Sobel threshold
def sobel_threshold(img_rgb, orient='x', sobel_kernel=3):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    return sobel

def mag_thresh(img_rgb, sobel_kernel=3, thresh_min=0, thresh_max=255):
    sobelx = sobel_threshold(img_rgb, orient='x', sobel_kernel=sobel_kernel)
    sobely = sobel_threshold(img_rgb, orient='y', sobel_kernel=sobel_kernel)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))
    return binary_image(scaled_sobel, thresh_min, thresh_max)

def dir_threshold(img_rgb, sobel_kernel=3, thresh_min=0, thresh_max=np.pi/2):
    sobelx = sobel_threshold(img_rgb, orient='x', sobel_kernel=sobel_kernel)
    sobely = sobel_threshold(img_rgb, orient='y', sobel_kernel=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    return binary_image(absgraddir, thresh_min, thresh_max)

def r_threshold(img_rgb, thresh_min=0, thresh_max=255):
    r = img_rgb[:,:,2]
    return binary_image(r, thresh_min, thresh_max)

def s_hls_threshold(img_rgb, thresh_min=0, thresh_max=255):
    hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]
    return binary_image(s, thresh_min, thresh_max)

def l_hls_threshold(img_rgb, thresh_min=0, thresh_max=255):
    hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
    l = hls[:,:,1]
    return binary_image(l, thresh_min, thresh_max)

def yellow_threshold(img_rgb):
    # Reference https://medium.com/@tjosh.owoyemi/finding-lane-lines-with-colour-thresholds-beb542e0d839
    lower = np.array([225,160,0],dtype = "uint8")
    upper = np.array([255, 255, 170],dtype = "uint8")
    mask = cv2.inRange(img_rgb, lower, upper)

    rgb_y = cv2.bitwise_and(img_rgb, img_rgb, mask = mask).astype(np.uint8)
    rgb_y = cv2.cvtColor(rgb_y, cv2.COLOR_RGB2GRAY)
    rgb_y = binary_image(rgb_y, 20, 255)

    hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
    hls_lower = np.array([20,90,80],dtype = "uint8")
    hls_upper = np.array([45, 200, 255],dtype = "uint8")
    mask = cv2.inRange(hls, hls_lower, hls_upper)
    hls_y = cv2.bitwise_and(img_rgb, img_rgb, mask = mask).astype(np.uint8)
    hls_y = cv2.cvtColor(hls_y, cv2.COLOR_HLS2RGB)
    hls_y = cv2.cvtColor(hls_y, cv2.COLOR_RGB2GRAY)
    hls_y = binary_image(hls_y, 20, 255)

    binary_output = np.zeros_like(hls_y)
    binary_output [(hls_y == 1)|(rgb_y==1)]= 1
    return binary_output

def white_threshold(img_rgb):
    # Reference https://medium.com/@tjosh.owoyemi/finding-lane-lines-with-colour-thresholds-beb542e0d839
    lower = np.array([120,120,200],dtype = "uint8")
    upper = np.array([255, 255, 255],dtype = "uint8")
    mask = cv2.inRange(img_rgb, lower, upper)
    rgb_w = cv2.bitwise_and(img_rgb, img_rgb, mask = mask).astype(np.uint8)
    rgb_w = cv2.cvtColor(rgb_w, cv2.COLOR_RGB2GRAY)
    return binary_image(rgb_w, 20, 255)
