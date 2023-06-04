import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from datetime import datetime


# load the arrays from files
images = np.load('train/images.npy')
masks = np.load('train/images.npy')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Print the shapes of the training and validation sets
print("Training set - Images:", X_train.shape)
print("Training set - Masks:", y_train.shape)
print("Validation set - Images:", X_val.shape)
print("Validation set - Masks:", y_val.shape)

# Apply random transformations like rotation, scaling, flipping, or adjusting brightness to augment images and their corresponding masks
# Data augmentation increase the diversity and size of your dataset

# Define the data generator
datagen = ImageDataGenerator(
    rotation_range=20,  # randomly rotate images
    horizontal_flip=True  # randomly flip images horizontally
)


# Define the data generator function
def data_generator(images, masks, batch_size):
    num_samples = len(images)
    while True:
        # Generate random indices for the batch
        indices = np.random.choice(num_samples, batch_size, replace=False)

        # Generate a batch of images and masks
        batch_images = images[indices]
        batch_masks = masks[indices]

        # Preprocess the images and masks using the data generator
        batch_images = datagen.flow(batch_images, batch_size=batch_size, shuffle=False).next()
        batch_masks = datagen.flow(batch_masks, batch_size=batch_size, shuffle=False).next()

        yield batch_images, batch_masks

def unet_model(input_shape):
    # Input layer
    inputs = tf.keras.layers.Input(input_shape)

    # Contracting path (Encoder)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottom of the UNet
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)

    # Expanding path (Decoder)
    up5 = tf.keras.layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv4)
    concat5 = tf.keras.layers.concatenate([up5, conv3], axis=3)
    conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(concat5)
    conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv5)
    concat6 = tf.keras.layers.concatenate([up6, conv2], axis=3)
    conv6 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(concat6)
    conv6 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv6)
    concat7 = tf.keras.layers.concatenate([up7, conv1], axis=3)
    conv7 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(concat7)
    conv7 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)

    # Output layer
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv7)

    # Create the model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model

def dice_coefficient(y_true, y_pred, smooth=1e-7):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

# define model architecture
model = unet_model(input_shape=(64, 64, 3))

# compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[dice_coefficient])

# define hyperparameters
epochs = 1
batch_size = 32

# create a data generator
train_generator = data_generator(X_train, y_train, batch_size=batch_size)
validation_generator = data_generator(X_val, y_val, batch_size=batch_size)

# train the model using fit_generator
model.fit_generator(generator=train_generator,
                    steps_per_epoch=len(X_train) // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=len(X_val) // batch_size)

# save model
model.save(f'model/')
