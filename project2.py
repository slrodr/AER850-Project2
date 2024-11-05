# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:40:08 2024

@author: Santiagp
"""

import tensorflow as tf
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
