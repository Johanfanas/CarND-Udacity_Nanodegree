#!/usr/bin/env python

import csv
import cv2
import numpy as np
import math
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def data():
    lines = []
    with open('../training_data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
            
    images = []
    measurements = []
    for line in lines:
        steering_center = float(line[3])

        # create adjusted steering measurements for the side camera images
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        path = '../training_data/IMG/' # fill in the path to your training IMG directory
        
        img_center = path + line[0].split('/')[-1]
        img_left = path + line[1].split('/')[-1]
        img_right = path + line[2].split('/')[-1]

        #add images and angles to data set
        images.append(img_center)
        images.append(img_left)
        images.append(img_right)
        measurements.append(steering_center)
        measurements.append(steering_left)
        measurements.append(steering_right)
    return images, measurements

def generator(samples, batch_size):
    steering_angle = samples[1]
    samples = samples[0]
    num_samples = len(samples)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_steer = steering_angle[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample)
                fliped_image = cv2.flip(center_image, 1)       
                
                images.append(center_image)
                images.append(fliped_image)
                
            for batch_sample in batch_steer:
                center_angle = float(batch_sample)
                fliped_angle = -center_angle  
                
                angles.append(center_angle)
                angles.append(fliped_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield np.array(X_train), np.array(y_train)

def main():
    batch_size = 128
    
    samples, angles = data()
    samples, angles = shuffle(samples, angles)
    X_train, X_valid, y_train, y_valid = train_test_split(samples, angles,test_size=0.2, shuffle=False)
    X_train, y_train = shuffle(X_train, y_train)
    X_valid, y_valid = shuffle(X_valid, y_valid)
    print('Number of training samples:', len(X_train))

    train_generator = generator((X_train, y_train), batch_size=batch_size)
    validation_generator = generator((X_valid, y_valid), batch_size=batch_size)

    from keras.models import Sequential
    from keras.callbacks import EarlyStopping
    from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
    from keras.layers.convolutional import Convolution2D
    from keras.layers.pooling import MaxPooling2D

    model = Sequential()
    model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(15,5,5, activation='relu'))
    model.add(Dropout(0.6)) 
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    model.fit_generator(train_generator, steps_per_epoch=math.floor(len(X_train)/batch_size)\
                        , validation_data=validation_generator, \
                        validation_steps=math.floor(len(X_valid)/batch_size), \
                        epochs=3, verbose=1, callbacks=[EarlyStopping(verbose=1)])
    
    model.save('model.h5')
    exit()
    
try:
    main()
except KeyboardInterrupt:
    pass
