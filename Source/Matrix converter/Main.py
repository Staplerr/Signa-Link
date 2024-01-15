import mediapipe as mp
from mediapipe.tasks.python import vision
import cv2
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from pathlib import Path
import os

parentPath = Path(__file__).parent
print("Parent directory: " + str(parentPath))
inputDirectory = parentPath.joinpath("Input")
outputFile = parentPath.joinpath("output" + ".csv")

modelDirectory = parentPath.joinpath("Model")
handModel = modelDirectory.joinpath("hand_landmarker.task")
minPoseConfidence = 0.5
poseModel = modelDirectory.joinpath("pose_landmarker_full.task")
minHandConfidence = 0.5
#Config
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
poseColumnNameList = ["nose", "left eye (inner)", "left eye", "left eye (outer)", "right eye (inner)",
                      "right eye", "right eye (outer)", "left ear", "right ear", "mouth (left)",
                      "mouth (right)", "left shoulder", "right shoulder", "left elbow", "right elbow",
                      "left wrist", "right wrist", "left pinky", "right pinky", "left index",
                      "right index","left thumb","right thumb","left hip","right hip"]
handColumnNameList = ["wrist", "thumb cmc", "thumb mcp", "thumb ip", "thumb tip",
                      "index finger mcp", "index finger pip", "index finger dip", "index finger tip", "middle finger mcp",
                      "middle finger pip", "middle finger dip", "middle finger tip", "ring finger mcp", "ring finger pip",
                      "ring finger dip", "ring finger tip", "pinky mcp", "pinky pip", "pinky dip",
                      "pinky tip"]

#create the landmarker object
poseOption = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=poseModel),
                                   running_mode=VisionRunningMode.IMAGE,
                                   min_pose_detection_confidence=minPoseConfidence)
handOption = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=handModel),
                                   running_mode=VisionRunningMode.IMAGE,
                                   min_hand_detection_confidence=minHandConfidence,
                                   num_hands=2)
poseLandmarker = PoseLandmarker.create_from_options(poseOption)
HandLandmarker = HandLandmarker.create_from_options(handOption)

#detecting landmarks
imagePATH = inputDirectory.joinpath("Moo/Hand.jpg").resolve()
image = mp.Image.create_from_file(str(imagePATH))
poseResult = poseLandmarker.detect(image)
poseCoordinates = poseResult.pose_landmarks[0][:25]
handResult = HandLandmarker.detect(image)
handCoordinates = handResult.hand_landmarks

#Convert to csv
Moo = []
column = []
columnHolder = []
rowHolder = []
#Pose dframe
i = 0
for landmark in poseCoordinates:
    rowHolder.append([landmark.x, landmark.y, landmark.z])
    columnHolder.append(poseColumnNameList[i])
    i += 1

#Hand dframe
for landmark in handCoordinates[0]:
    rowHolder.append([landmark.x, landmark.y, landmark.z])
for landmark in handCoordinates[1]:
    rowHolder.append([landmark.x, landmark.y, landmark.z])
for word in handColumnNameList:
    columnHolder.append(word + " Right Hand")
for word in handColumnNameList:
    columnHolder.append(word + " Left Hand")

#Setting up dframe
Moo.append(rowHolder)
column.append(columnHolder)
df = pd.DataFrame(Moo)
df.columns = column
df.index = ["หมู"]
df.to_csv(outputFile)
print(df)
print(df.index)
