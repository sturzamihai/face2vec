import cv2
import keras
import numpy as np
from utils import load_image, find_closest

IMG_SIZE = (32, 32)

model = keras.models.load_model('./checkpoints/dae-model-01.hdf5')

image1 = load_image("../dataset/image0.jpg",IMG_SIZE)
image2 = load_image("../dataset/image1.jpg",IMG_SIZE)
image3 = load_image("../dataset/image2.png",IMG_SIZE)
image4 = load_image("../dataset/image3.jpg",IMG_SIZE)

bottleneck = keras.models.Model(
    model.input, model.get_layer('bottleneck').output)

bottleneck_value1 = bottleneck.predict(image1)
bottleneck_value2 = bottleneck.predict(image2)
bottleneck_value3 = bottleneck.predict(image3)
bottleneck_value4 = bottleneck.predict(image4)

find_closest([bottleneck_value1,bottleneck_value3,bottleneck_value2,bottleneck_value4])