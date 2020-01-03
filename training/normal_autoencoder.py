import os
import keras
import numpy as np
from PIL import Image

# CelebA Dataset
PATH = '/Users/mihaisturza/Downloads/img_align_celeba/'

entry = keras.layers.Input((512, 512, 3))
output = keras.layers.Conv2D(
    32, (3, 3), activation='relu', padding='same')(entry)
output = keras.layers.AveragePooling2D(padding='same')(output)
output = keras.layers.Conv2D(
    32, (3, 3), activation='relu', padding='same')(output)
output = keras.layers.AveragePooling2D(padding='same')(output)
output = keras.layers.Conv2D(
    32, (3, 3), activation='relu', padding='same')(output)
output = keras.layers.AveragePooling2D(padding='same')(output)
output = keras.layers.Conv2D(
    32, (3, 3), activation='relu', padding='same')(output)
output = keras.layers.AveragePooling2D(padding='same')(output)
output = keras.layers.Conv2D(
    32, (3, 3), activation='relu', padding='same', name='bottleneck')(output)
output = keras.layers.UpSampling2D()(output)
output = keras.layers.Conv2D(
    32, (3, 3), activation='relu', padding='same')(output)
output = keras.layers.UpSampling2D()(output)
output = keras.layers.Conv2D(
    32, (3, 3), activation='relu', padding='same')(output)
output = keras.layers.UpSampling2D()(output)
output = keras.layers.Conv2D(
    32, (3, 3), activation='relu', padding='same')(output)
output = keras.layers.UpSampling2D()(output)
output = keras.layers.Conv2D(
    32, (3, 3), activation='sigmoid', padding='same')(output)

model = keras.models.Model(entry, output)
model.compile(optimizer='adam', loss='binary_crossentropy')

model.summary()

# model.fit() ... 

image1 = np.reshape(np.random.random((512, 512, 3)), (1, 512, 512, 3))
image2 = np.reshape(np.random.random((512, 512, 3)), (1, 512, 512, 3))
bottleneck = keras.models.Model(
    model.input, model.get_layer('bottleneck').output)
bottleneck_value1 = bottleneck.predict(image1)
bottleneck_value2 = bottleneck.predict(image2)

print(np.linalg.norm(bottleneck_value1-bottleneck_value2))
