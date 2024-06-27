import mediapipe as mp
import pandas as pd
import cv2
from pathlib import Path
import tensorflow as tf
import numpy as np
from google.protobuf.json_format import MessageToDict
import json
from keras.models import load_model


model = load_model(f"{Path(__file__).parent}/Matrix model/best_model.h5")

# Directory
parentDirectory = Path(__file__).parent
inputDirectory = parentDirectory.joinpath("Videos")

# Opencv config
resizeRatio = [1280, 720]
resizeInterpolation = cv2.INTER_AREA

# For reading all files in the video folder
supportsExtension = ["**/*.mp4", "**/*.mov"]

# Dataframe
f = open(f"{str(Path(__file__).parent)}/Data/label.json",)
labelList = json.load(f)
videoPaths = {}

for index, directory in enumerate(inputDirectory.iterdir()):
    for file in directory.iterdir():
        videoPaths[file] = index

if __name__ == "__main__":  
    totalFile = len(videoPaths)
    counter = 0
    Data = np.empty((0, 10, 42, 3), dtype=np.float32)
    Label = np.empty((0,), dtype=np.float32)
    
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4) as hands:
        
        for video, index in videoPaths.items(): # loop video
            cap = cv2.VideoCapture(str(video))
            counter += 1
            frameArray = np.empty((0,42,3), dtype=np.float32)
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                  break
              
                image.flags.writeable = True # To improve performance, optionally mark the image as not writeable to
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (resizeRatio[0], resizeRatio[1]), interpolation=resizeInterpolation)
                results = hands.process(image)
                Corrdinates = np.empty((0,3), dtype=np.float32)
                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
                        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
                        if cv2.waitKey(5) & 0xFF == 27:
                            break
                        if len(results.multi_handedness) == 1: # เจอข้างเดียว
                            handedness_dict = MessageToDict(handedness)
                            if handedness_dict["classification"][0]["index"] == 0:
                                for landmark in hand_landmarks.landmark:
                                    Corrdinates = np.concatenate((Corrdinates, [[landmark.x, landmark.y, landmark.z]]), axis=0)
                                for i in range(21): # Fill
                                    Corrdinates = np.concatenate((Corrdinates, [[0 for i in range(3)]]), axis=0)
                            elif handedness_dict["classification"][0]["index"] == 1:
                                for i in range(21): # Fill
                                    Corrdinates = np.concatenate((Corrdinates, [[0 for i in range(3)]]), axis=0)
                                for landmark in hand_landmarks.landmark:
                                    Corrdinates = np.concatenate((Corrdinates, [[landmark.x, landmark.y, landmark.z]]), axis=0)
                            else:
                                continue
                    if len(results.multi_handedness) == 2: # เจอสองข้าง
                        for landmark in results.multi_hand_landmarks[0].landmark:
                            Corrdinates = np.concatenate((Corrdinates, [[landmark.x, landmark.y, landmark.z]]), axis=0)
                        for landmark in results.multi_hand_landmarks[1].landmark:
                            Corrdinates = np.concatenate((Corrdinates, [[landmark.x, landmark.y, landmark.z]]), axis=0)
    
                if len(Corrdinates) == 42:
                    frameArray = np.concatenate((frameArray, [Corrdinates]), axis=0)
                if frameArray.shape == (10,42,3):
                    result = model.predict(frameArray.reshape((1, 10, 42, 3)))
                    print(f"Model results: {result.tolist()[0].index(max(result.tolist()[0]))}\nReal index: {index}")
                    frameArray = np.empty((0,42,3), dtype=np.float32)
    
            cap.release()
            cv2.destroyAllWindows()