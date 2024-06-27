import mediapipe as mp
import pandas as pd
import cv2
from pathlib import Path
import tensorflow as tf
import numpy as np
from google.protobuf.json_format import MessageToDict
import json

# Directory
parentDirectory = Path(__file__).parent
inputDirectory = parentDirectory.joinpath("Videos")

# Opencv config
resizeRatio = [1280, 720]
resizeInterpolation = cv2.INTER_AREA

# For reading all files in the video folder
supportsExtension = ["**/*.mp4", "**/*.mov"]

# Dataframe
labelList = {}
videoPaths = {}

for index, directory in enumerate(inputDirectory.iterdir()):
    labelList[directory.name] = index
    labelList[index] = directory.name
    for file in directory.iterdir():
        videoPaths[file] = index

if __name__ == "__main__":  
    totalFile = len(videoPaths)
    counter = 0
    Data = np.empty((0, 10, 42, 3), dtype=np.float32)
    Label = np.empty((0,), dtype=np.float32)
    
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
              
                image.flags.writeable = False # To improve performance, optionally mark the image as not writeable to
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (resizeRatio[0], resizeRatio[1]), interpolation=resizeInterpolation)
                results = hands.process(image)
                Corrdinates = np.empty((0,3), dtype=np.float32)
                if results.multi_hand_landmarks:
                    if len(results.multi_handedness) == 1: # เจอข้างเดียว
                        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
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
                    Data = np.concatenate((Data, [frameArray]), axis=0)
                    Label = np.concatenate((Label, [index]), axis=0)
                    frameArray = np.empty((0,42,3), dtype=np.float32)
    
            print(f"Progress : {counter} / {totalFile}\nCollected data : {Data.shape[0]}")
            cap.release()
            cv2.destroyAllWindows()
        
    print(f"All data collected: {Data.shape[0]}")

    # saving things
    try:
        np.save(f"{parentDirectory}\Data\Features.npy", Data)
    except Exception:
        np.save(f"Features.npy", Data)
    try:
        np.save(f"{parentDirectory}\Data\label.npy", Label)
    except Exception:
        np.save(f"Label.npy", Label)
    with open('label.json', 'w') as f:
        json.dump(labelList, f)

    