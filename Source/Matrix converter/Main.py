import mediapipe as mp
from mediapipe.tasks.python import vision
import cv2
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from pathlib import Path

parentPath = Path(__file__).parent
print("Parent directory: " + str(parentPath))
inputDirectory = parentPath.joinpath("Input/")
outputFile = parentPath.joinpath("output" + ".csv")

modelDirectory = parentPath.joinpath("Model")
handModel = modelDirectory.joinpath("hand_landmarker.task")
poseModel = modelDirectory.joinpath("pose_landmarker_full.task")
#Config
minPoseConfidence = 0.5
minHandConfidence = 0.5
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

#initiate dataframe
columnHolder = ["Label"]
for columnName in poseColumnNameList:
    columnHolder.append(columnName)
for columnName in handColumnNameList:
    columnHolder.append(columnName + " Right")
for columnName in handColumnNameList:
    columnHolder.append(columnName + " Left")
df = pd.DataFrame(columns=columnHolder)

#function for adding landmarks on 
def addLandMark(coordinates, index, value, i): #add landmark to list
    for landmark in coordinates[index]:
        value[i] = [landmark.x, landmark.y, landmark.z]
        i += 1
    return value, i
def toDataFrame(imagePATH, index): #convert image path to be added to dataframe
    print("Adding " + imagePATH.name + " to dataframe as " + index + " to index " + str(len(df)))
    image = mp.Image.create_from_file(str(imagePATH))
    poseResult = poseLandmarker.detect(image)
    handResult = HandLandmarker.detect(image)
    poseCoordinates = poseResult.pose_landmarks[0][:25]
    handCoordinates = handResult.hand_landmarks
    value = [0] * 68 #0 = label, 1-25 = pose, 26-46 = right hand 47-67 = left hand
    value[0] = index
    i = 1
    #add landmarks to list
    for landmark in poseCoordinates:
        value[i] = [landmark.x, landmark.y, landmark.z]
        i += 1
    if len(handCoordinates) > 1:
        value, i = addLandMark(handCoordinates, 0, value, i)
        value, i = addLandMark(handCoordinates, 1, value, i)
    elif len(handCoordinates) > 0: #detect if the image does have a hand in the first place
        if handResult.handedness[0][0].category_name == "Left":
            i += 21
            value, i = addLandMark(handCoordinates, 0, value, i)
        else:
            value, i = addLandMark(handCoordinates, 0, value, i)
    #add landmarks to dataframe
    df.loc[len(df)] = value

#"g o o d s t u f f"
imageSubdirectory = inputDirectory.iterdir()
for childDirectory in imageSubdirectory:
    if childDirectory.is_dir():
        for image in childDirectory.glob("**/*.jpg"):   #reading all image in input directory
            index = childDirectory.name                 #saving directory name to use as index name
            toDataFrame(image, index)
print("Output dataframe:")
print(df)
df.to_csv(outputFile)