import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from utils import BATCH_SIZE, IMG_SIZE

entry = keras.layers.Input((*IMG_SIZE, 3))

# encoder
encoder = keras.layers.Conv2D(
    32, (3, 3), activation='relu', padding='same')(entry)
encoder = keras.layers.AveragePooling2D(padding='same')(encoder)
encoder = keras.layers.Conv2D(
    32, (3, 3), activation='relu', padding='same')(encoder)
encoder = keras.layers.AveragePooling2D(padding='same')(encoder)
encoder = keras.layers.Conv2D(
    32, (3, 3), activation='relu', padding='same')(encoder)
encoder = keras.layers.AveragePooling2D(padding='same')(encoder)

# bottleneck
bottleneck = keras.layers.Conv2D(
    32, (3, 3), activation='relu', padding='same', name='bottleneck')(encoder)

# decoder
decoder = keras.layers.UpSampling2D()(bottleneck)
decoder = keras.layers.Conv2D(
    32, (3, 3), activation='relu', padding='same')(decoder)
decoder = keras.layers.UpSampling2D()(decoder)
decoder = keras.layers.Conv2D(
    32, (3, 3), activation='relu', padding='same')(decoder)
decoder = keras.layers.UpSampling2D()(decoder)
decoder = keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoder)

model = keras.models.Model(entry, decoder)
model.compile(optimizer='adam', loss='binary_crossentropy')

model.summary()

data = ImageDataGenerator(validation_split=0.2,preprocessing_function=lambda x:x/255)
train = data.flow_from_directory('../dataset', target_size=IMG_SIZE,
                                 class_mode='input', batch_size=BATCH_SIZE, subset='training')
valid = data.flow_from_directory('../dataset', target_size=IMG_SIZE,
                                 class_mode='input', batch_size=BATCH_SIZE, subset='validation')

model.fit_generator(train, steps_per_epoch=1, epochs=1,
                    callbacks=[ModelCheckpoint('./checkpoints/ae-model-{epoch:02d}.hdf5', verbose=1)], validation_data=valid, validation_steps=1)

