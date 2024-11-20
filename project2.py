# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:40:08 2024

@author: Santiagp
"""
import time
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


input_shape = (500, 500, 3) #Desired image shape (Height, Width, Channel)
train_dir = r"D:\AER850\AER850-Project2\Data\train"
validation_dir = r"D:\AER850\AER850-Project2\Data\valid"
bsize = 32  #Set batch size

'''Data Processing'''
train_datagen = ImageDataGenerator(rescale = 1.0/255., shear_range = 0.4, 
                                   zoom_range = 0.4, rotation_range=15,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2, 
                                   horizontal_flip = True)
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
def model_eval_plot (history, title):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    
    plt.figure()
    plt.suptitle(title, fontsize=16)
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='Training Accuracy', color='blue')
    plt.plot(val_accuracy, label='Validation Accuracy', color='orange')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


'''First CNN Design'''
#Design
model1 = Sequential()
model1.add(Conv2D(filters = 16, kernel_size=(3, 3), activation='relu',
                              	input_shape=input_shape, padding = 'same'))
model1.add(MaxPooling2D((2,2)))
model1.add(Conv2D(32, (3, 3), activation='relu', padding = 'same'))
model1.add(MaxPooling2D((2,2)))
model1.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
model1.add(MaxPooling2D((2,2)))
model1.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))
model1.add(MaxPooling2D((2,2)))
model1.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
model1.add(MaxPooling2D((2,2)))
model1.add(Flatten())
model1.add(Dense(128, activation='relu'))
model1.add(Dropout(0.2, seed=2000))
model1.add(Dense(64, activation='relu'))
model1.add(Dropout(0.3, seed=2000))
model1.add(Dense(32, activation='relu'))
model1.add(Dropout(0.4, seed=2000))
model1.add(Dense(32, activation='relu'))
model1.add(Dropout(0.5, seed=2000))
model1.add(Dense(3, activation='softmax'))
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model1.summary()

#Training while monitoring how long it takes
start1 = time.time()
history1 = model1.fit(train_gen, steps_per_epoch = 60, epochs = 50, 
                      validation_data=validation_gen, 
                      validation_steps = 20, verbose = 2)
end1 = time.time()
train_time1 = end1-start1
model1.save('model1.h5')
#Evaluation
model_eval_plot(history1, "Model 1 Loss and Accuracy")
print(f"Model 1 training time: {train_time1 / 60:.2f} minutes")
'''Second CNN Design'''
#Design
model2 = Sequential()
model2.add(Conv2D(filters = 16, kernel_size=(3, 3),strides=(1, 1), activation = layers.LeakyReLU,
                              	input_shape=input_shape))
model2.add(MaxPooling2D((2,2)))
model2.add(Conv2D(32, (3, 3),strides=(1, 1), activation = layers.LeakyReLU))
model2.add(MaxPooling2D((2,2)))
model2.add(Conv2D(64, (3, 3),strides=(1, 1), activation = layers.LeakyReLU))
model2.add(MaxPooling2D((2,2)))
model2.add(Flatten())
model2.add(Dense(64, activation='elu'))
model2.add(Dropout(0.3, seed=2000))
model2.add(Dense(32, activation='elu'))
model2.add(Dropout(0.4, seed=2000))
model2.add(Dense(16, activation='elu'))
model2.add(Dropout(0.5, seed=2000))
model2.add(Dense(3, activation='softmax'))
model2.compile(optimizer='nadam', loss='categorical_crossentropy', 
               metrics = ['accuracy'])

model2.summary()

#Training while monitoring how long it takes
start2 = time.time()
history2 = model2.fit(train_gen, steps_per_epoch = 50, epochs = 50,
                      validation_data = validation_gen,
                      validation_steps = 20, verbose = 2)
end2 = time.time()
train_time2 = end2-start2

#Evaluation
model_eval_plot(history2, "Model 2 Loss and Accuracy")
print(f"Model 2 training time: {train_time2 / 60:.2f} minutes")
model2.save('model2.h5')