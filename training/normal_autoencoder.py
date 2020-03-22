import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from functools import reduce

IMG_SIZE = (32, 32)
BATCH_SIZE = 32
HIDDEN_SIZE = 256
BOTTLE_SIZE = 128
INPUT_SIZE = reduce(lambda x, y: x*y, [*IMG_SIZE,3])

entry = keras.layers.Input((*IMG_SIZE,3))
flatten = keras.layers.Flatten()(entry)

# encoder
encoder = keras.layers.Dense(
    HIDDEN_SIZE*3, activation='relu')(flatten)
encoder = keras.layers.Dense(
    HIDDEN_SIZE*2, activation='relu')(encoder)
encoder = keras.layers.Dense(
    HIDDEN_SIZE, activation='relu')(encoder)

# bottleneck
bottleneck =keras.layers.Dense(
    BOTTLE_SIZE, activation='relu', name='bottleneck')(encoder)

# decoder
decoder = keras.layers.Dense(
    HIDDEN_SIZE*2, activation='relu')(bottleneck)
decoder = keras.layers.Dense(
    HIDDEN_SIZE*3, activation='relu')(decoder)
decoder = keras.layers.Dense(
    INPUT_SIZE, activation='sigmoid')(decoder)
decoder = keras.layers.Reshape((*IMG_SIZE, 3))(decoder)

model = keras.models.Model(entry, decoder)
model.compile(optimizer='adam', loss='binary_crossentropy')

model.summary()

data = ImageDataGenerator(validation_split=0.2,preprocessing_function=lambda x:x/255)
train = data.flow_from_directory('../dataset', target_size=IMG_SIZE,
                                 class_mode='input', batch_size=BATCH_SIZE, subset='training')
valid = data.flow_from_directory('../dataset', target_size=IMG_SIZE,
                                 class_mode='input', batch_size=BATCH_SIZE, subset='validation')


model.fit_generator(train, steps_per_epoch=train.n//train.batch_size, epochs=1,
                    callbacks=[ModelCheckpoint('./checkpoints/dae-model-{epoch:02d}.hdf5', verbose=1)], validation_data=valid, validation_steps=valid.n//valid.batch_size)

