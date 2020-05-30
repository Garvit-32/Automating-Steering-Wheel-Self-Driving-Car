import csv
import numpy as np
import cv2
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Lambda, Dropout,BatchNormalization
from keras.layers.convolutional import Cropping2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import time


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

samples = []
#read data.txt
with open("driving_dataset/data.txt") as f:
    for line in f:
        samples.append(line)
# print(samples)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # for i in range(3): # center, left and rights images
                    name = 'driving_dataset/' + line.split()[0]
                    # print(name)
                    
                    current_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                    current_image = cv2.resize(current_image,(320,160))
                    # current_image = current_image/255.0
                    images.append(current_image)
                    
                    angles.append(float(line.split()[1]) * 3.14159265 / 180)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_samples, validation_samples = train_test_split(samples[:], test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

tensorboard_callback = TensorBoard(log_dir="logs/{}".format(time.time()))


# nVidia model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24,(5,5), strides=(2,2), activation='elu'))
model.add(BatchNormalization())  
model.add(Conv2D(36,(5,5), strides=(2,2), activation='elu'))
model.add(BatchNormalization())  
model.add(Conv2D(48,(5,5), strides=(2,2), activation='elu'))
model.add(BatchNormalization())  
model.add(Conv2D(64,(3,3), activation='elu'))
model.add(BatchNormalization())  
model.add(Conv2D(64,(3,3), activation='elu'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(256, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(25, activation='elu'))
model.add(Dense(1))
model.compile(optimizer=Adam(lr=0.0001), loss='mse',metrics=['accuracy'])


# fit the model
model.fit_generator(train_generator, steps_per_epoch=len(train_samples),
validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1,callbacks=[tensorboard_callback])

model.save('model.h5')