# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:40:08 2024

@author: Santiagp
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


input_shape = (500, 500, 3) #Desired image shape (Height, Width, Channel)
train_dir = r"D:\AER850\AER850-Project2\Data\train"
validation_dir = r"D:\AER850\AER850-Project2\Data\valid"
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

#Defining a function for model evaluation plotting
def model_eval_plot (history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs_model = range(1, len(val_loss)+1)
    
    plt.figure()
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_model, train_loss, label='Training Loss', color='blue')
    plt.plot(epochs_model, val_loss, label='Validation Loss', color='orange')
    plt.xlabel('Epochs') 
    plt.ylabel('Loss')
    plt.label()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_model, train_accuracy, label='Training Accuracy', color='blue')
    plt.plot(epochs_model, val_accuracy, label='Validation Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


'''First CNN Design'''
#Design
model1 = Sequential()
model1.add(tf.keras.layers.Conv2D(filters = 16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model1.add(tf.keras.layers.MaxPooling2D((2,2)))
model1.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model1.add(tf.keras.layers.MaxPooling2D((2,2)))
model1.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(tf.keras.layers.MaxPooling2D((2,2)))
model1.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model1.add(tf.keras.layers.MaxPooling2D((2,2)))
model1.add(tf.keras.layers.Flatten())
model1.add(tf.keras.layers.Dense(512, activation='relu'))
model1.add(tf.keras.layers.Dropout(0.1, seed=2000))
model1.add(tf.keras.layers.Dense(128, activation='relu'))
model1.add(tf.keras.layers.Dropout(0.2, seed=2000))
model1.add(tf.keras.layers.Dense(512, activation='relu'))
model1.add(tf.keras.layers.Dropout(0.1, seed=2000))
model1.add(tf.keras.layers.Dense(128, activation='relu'))
model1.add(tf.keras.layers.Dropout(0.2, seed=2000))
model1.add(tf.keras.layers.Dense(128, activation='relu'))
model1.add(tf.keras.layers.Dropout(0.5, seed=2000))
model1.add(tf.keras.layers.Dense(3, activation='softmax'))
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model1.summary()

#Training
history1 = model1.fit(train_gen, steps_per_epoch = 60, epochs = 50, 
                      validation_data=validation_gen, 
                      validation_steps = 20, verbose = 2)

#Evaluation
model_eval_plot(history1)