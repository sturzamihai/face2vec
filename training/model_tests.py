import cv2
import keras
import numpy as np
from utils import BATCH_SIZE, IMG_SIZE, load_image

def find_closest(vect):
    vects = []
    for idx,img in enumerate(vect):
        dist = []
        for tidx,trgt in enumerate(vect):
            if idx!=tidx:
                dist.append(np.linalg.norm(img-trgt))
        print(dist)
        print(dist.index(min(dist)))
        vects.append(dist.index(min(dist)))


model = keras.models.load_model('./checkpoints/model-01.hdf5')

image1 = load_image("../dataset/image0.jpg")
image2 = load_image("../dataset/image1.jpg")
image3 = load_image("../dataset/image2.png")
image4 = load_image("../dataset/image3.jpg")

bottleneck = keras.models.Model(
    model.input, model.get_layer('bottleneck').output)

bottleneck_value1 = bottleneck.predict(image1)
bottleneck_value2 = bottleneck.predict(image2)
bottleneck_value3 = bottleneck.predict(image3)
bottleneck_value4 = bottleneck.predict(image4)

find_closest([bottleneck_value1,bottleneck_value3,bottleneck_value2,bottleneck_value4])