# Hand Command

## Description
Python 3 script to control MacOS system volume using hand gestures. A thumbs up increases the volume and a thumbs down decreases the volume. 

## Dependencies
1. opencv-python
2. osascript
3. numpy
4. scikit-learn
5. keras

## Usage
1. Generate training data for the model to run on using the run_model function with the train flag set to true. Press 1 on keyboard to capture an image of a thumbs up, 2 to capture a thumbs down, and 0 to capture the enviroment. This will create a file titled train.csv in the directory it was run from with labeled training data.
2. Run the create_model function, passing the paths of the training data and net.pkl as inputs. When the model is finished training its weights will be saved to net.pkl. 
3. Pass the created model to run_model as an input. If training was performed correctly you will now be able to raise system volume with a thumbs up and decrease it with a thumbs down by running command.py. 
