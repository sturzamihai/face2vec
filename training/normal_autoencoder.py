import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# CelebA Dataset
PATH = '/Users/mihaisturza/Downloads/img_align_celeba/'
BATCH_SIZE = 32
IMG_SIZE = (512, 512)

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

model.fit_generator(train, steps_per_epoch=train.n//train.batch_size, epochs=1,
                    callbacks=[ModelCheckpoint('./checkpoints', verbose=1)], validation_data=valid, validation_steps=valid.n//valid.batch_size)

image1 = np.reshape(np.random.random((512, 512, 3)), (1, 512, 512, 3))
image2 = np.reshape(np.random.random((512, 512, 3)), (1, 512, 512, 3))
bottleneck = keras.models.Model(
    model.input, model.get_layer('bottleneck').output)
bottleneck_value1 = bottleneck.predict(image1)
bottleneck_value2 = bottleneck.predict(image2)

print(np.linalg.norm(bottleneck_value1-bottleneck_value2))
