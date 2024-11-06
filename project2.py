# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:40:08 2024

@author: Santiagp
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


input_shape = (500, 500, 3) #Desired image shape (Height, Width, Channel)
train_dir = 'Data/train' 
validation_dir = 'Data/valid'
bsize = 32  #Set batch size

'''Data Processing'''
train_datagen = ImageDataGenerator(rescale = 1.0/255., shear_range = 0.4, 
                                   zoom_range = 0.4, horizontal_flip = True)
validation_datagen = ImageDataGenerator(rescale = 1.0/255.)

train_gen = train_datagen.flow_from_directory(train_dir, batch_size=bsize,
                                             class_mode = 'categorical', 
                                             target_size =input_shape[0:2])
validation_gen = validation_datagen.flow_from_directory(validation_dir,
                                                        batch_size=bsize,
                                                        class_mode = 'categorical',
                                                        target_size = input_shape[0:2])
#print("Train generator class indices:", train_generator.class_indices) #Checking classes
#print("Validation generator class indices:", validation_gen.class_indices)

'''CNN Design'''
model1 = Sequential()
model1.add(tf.keras.layers.Conv2D(filters = 16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model1.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(600, activation="relu"),
    tf.keras.layers.Dropout(0.1, seed=2019),
    tf.keras.layers.Dense(400, activation="relu"),
    tf.keras.layers.Dropout(0.3, seed=2019),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dropout(0.4, seed=2019),
    tf.keras.layers.Dense(200, activation="relu"),
    tf.keras.layers.Dropout(0.2, seed=2019),
    tf.keras.layers.Dense(4, activation="softmax")
])