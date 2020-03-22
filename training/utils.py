import cv2
import numpy as np

def find_closest(vect):
    for idx,img in enumerate(vect):
        dist = []
        for tidx,trgt in enumerate(vect):
            if idx!=tidx:
                dist.append(np.linalg.norm(img-trgt))
        print(dist.index(min(dist)))

def load_image(path, size):
    image1 = cv2.imread(path)
    image1 = cv2.resize(image1, size)
    cv2.imshow("Test", image1)
    cv2.waitKey(0)
    image1 = np.reshape(image1[::-1], (1,*size,3))
    return image1