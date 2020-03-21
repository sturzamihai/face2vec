import cv2
import numpy as np

BATCH_SIZE = 32
IMG_SIZE = (512, 512)

def load_image(path):
    image1 = cv2.imread(path)
    image1 = cv2.resize(image1, IMG_SIZE)
    cv2.imshow("Test", image1)
    cv2.waitKey(0)
    image1 = np.reshape(image1[::-1], (1,*IMG_SIZE,3))
    return image1