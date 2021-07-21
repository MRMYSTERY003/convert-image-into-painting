import cv2
import numpy as np


def read_file(filename):
   img = cv2.imread(filename)
   return img

filename = 'test.jpg'
img = read_file(filename)

def edge_mask(img, line_size, blur_value):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    return edges

line_size = 5
blur_value = 5
edges = edge_mask(img, line_size, blur_value)



def color_quantization(img, k):
    # Transform the image
    data = np.float32(img).reshape((-1, 3))

    # Determine criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    # Implementing K-Means
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result


total_color = 9
paint = color_quantization(img, total_color)
blurred = cv2.bilateralFilter(img, d=8, sigmaColor=100,sigmaSpace=100)
cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)


cv2.imshow('original', img)
cv2.imshow('painting', paint)
cv2.waitKey(0)