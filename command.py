import cv2
import time
import os
import osascript
import numpy as np
import string
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from keras.utils.np_utils import to_categorical 

# set numpy options for cleaner output
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

def load_model(filepath):
    # attempt to load net from pickled object
    net = joblib.load(filepath)
    
    return net

def create_model(object_filepath, training_filepath):
    # create file for pickled object if one does not exist yet
    if (not os.path.isfile(object_filepath)):
        open(object_filepath, 'a').close()

    # create MLPClassifier
    net = MLPClassifier(hidden_layer_sizes=(3000, 1000, 300), max_iter=1000)

    # read training data
    training_data = np.genfromtxt(training_filepath, delimiter=',')

    # split training data
    x = training_data[:, :-1]
    y = training_data[:, -1]

    # rescale training data
    x = x / 255

    # convert classification labels to 1 hot encoding
    y = to_categorical(y)

    # fit network
    net.fit(x, y)

    # dump trained MLP to object filepath
    joblib.dump(net, object_filepath)

    return net

def run_model(model, train=False):
    # initialize cam
    cam = cv2.VideoCapture(0)
    cam.set

    while True:
        # read image from cam
        ret_val, img = cam.read()

        # convert image to grayscale, resize to 160x90, and flip
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (80, 45))
        img = cv2.flip(img, 1)
        
        if train == True:
            # indicating no hand gesture
            if cv2.waitKey(0) == ord('0'):
                line = img.flatten()
                line = np.append(line, 0)
                with open('train.csv','ab') as f:
                    np.savetxt(f, [line], fmt='%5d', delimiter=',')
                print('0')
            if cv2.waitKey(0) == ord('1'):
                line = img.flatten()
                line = np.append(line, 1)
                with open('train.csv','ab') as f:
                    np.savetxt(f, [line], fmt='%5d', delimiter=',')
                print('1')
            if cv2.waitKey(0) == ord('2'):
                line = img.flatten()
                line = np.append(line, 2)
                with open('train.csv','ab') as f:
                    np.savetxt(f, [line], fmt='%5d', delimiter=',')
                print('2')

            prediction = 'training'

        else: 
            # get system output volume at each frame
            code, volume, error = osascript.run('get output volume of (get volume settings)')

            # flatten, rescale, and reshape image to be read by MLP
            img_read = img.flatten()
            img_read = img_read / 255
            img_read = img_read.reshape(1, -1)

            # predict using network
            prediction = model.predict_proba(img_read)

            # find index of highest probability
            index = np.argmax(prediction[0])

            # incresase volume if net predicts thumbs up
            if index == 1:
                volume = int(volume) + 1
                osascript.run('set volume output volume ' + str(volume))

            # decrease volume if net predicts thumbs down
            if index == 2:
                volume = int(volume) - 1
                osascript.run('set volume output volume ' + str(volume))

            # transform image to higher resolution and color for display
            img = cv2.resize(img, (400, 225))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # add text
        # cv2.putText(img, str(prediction), (0, 70), cv2.FONT_HERSHEY_SIMPLEX,  
        #             1, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.imshow('Up Down!', img)

        # wait for escape key to exit
        if cv2.waitKey(1) == 27: 
            break  

    cv2.destroyAllWindows()

net = load_model('net.pkl')
run_model(net, train=False)

