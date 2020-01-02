import cv2
import time
import os
import osascript
import numpy as np
import string
import collections
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical 
from tensorflow.keras import datasets, layers, models

# set numpy options for cleaner output
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

def create_training_data():
    # create empty numpy arrays to store training data
    training_x = []
    training_y = []

    # initialize cam
    cam = cv2.VideoCapture(0)
    cam.set

    while True:
        # read image from cam
        ret_val, img = cam.read()

        # resize to 160x90 and flip
        img = cv2.resize(img, (160, 90))
        img = cv2.flip(img, 1)
        
        # indicating no hand gesture
        if cv2.waitKey(0) == ord('0'):
            training_x.append(img)
            training_y.append(0)
            print('0')
        
        # indicating thumbs up 
        if cv2.waitKey(0) == ord('1'):
            training_x.append(img)
            training_y.append(1)
            print('1')

        # indicating thumbs down
        if cv2.waitKey(0) == ord('2'):
            training_x.append(img)
            training_y.append(2)
            print('2')

        cv2.imshow('Training', img)

        # wait for escape key to exit
        if cv2.waitKey(0) == 27: 
            break  

    cv2.destroyAllWindows()

    return np.array(training_x), np.array(training_y)

def load_model(filepath):
    # attempt to load net from pickled object
    net = joblib.load(filepath)
    
    return net

def create_model(x, y, object_filepath='net.pkl'):
    # create convolutional network
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(90, 160, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    # rescale training data by centering around 0 
    x = (x / 127) - 1

    # break x and y into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # fit network
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    return model

def run_model(model):
    # initialize cam
    cam = cv2.VideoCapture(0)
    cam.set

    # declare empty list to store past most likely predictions
    prediction_history = collections.deque() 

    while True:
        # read image from cam
        ret_val, img = cam.read()

        # resize to 160x90 and flip
        img = cv2.resize(img, (160, 90))
        img = cv2.flip(img, 1)
    
        # get system output volume at each frame
        code, volume, error = osascript.run('get output volume of (get volume settings)')

        # rescale and reshape image to be read by MLP
        img_read = (img / 127) - 1
        img_read = np.expand_dims(img_read, axis=0)
        # img_read = img_read.reshape(1, -1)

        # predict using network
        prediction = model.predict_proba(img_read)

        # find index of most likely value
        index = np.argmax(prediction[0])

        # add most likely index to prediction history
        if len(prediction_history) > 4:
            prediction_history.popleft()
        prediction_history.append(index)

        # incresase volume if net consistantly predicts thumbs up
        if prediction[0][1] > 0.80 and prediction_history.count(1) > 4:
            volume = int(volume) + 5
            osascript.run('set volume output volume ' + str(volume))

        # decrease volume if net consistantly predicts thumbs down
        if prediction[0][2] > 0.80 and prediction_history.count(2) > 4:
            volume = int(volume) - 5
            osascript.run('set volume output volume ' + str(volume))

        # transform image to higher resolution and color for display
        img = cv2.resize(img, (400, 225))

        # add text
        cv2.putText(img, str(prediction), (0, 70), cv2.FONT_HERSHEY_SIMPLEX,  
                     1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Hand Command', img)

        # wait for escape key to exit
        if cv2.waitKey(10) == 27: 
            break  

    cv2.destroyAllWindows()

x, y = create_training_data()
net = create_model(x, y)
run_model(net)

