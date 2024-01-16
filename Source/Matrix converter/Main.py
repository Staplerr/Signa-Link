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
#list for converting label into a number
labelList = {"กรอบ": 0,
             "กระเพรา": 1,
             "ขา": 2,
             "ข้าว": 3,
             "ไข่": 4,
             "คะน้า": 5,
             "เค็ม": 6,
             "โจ๊ก": 7,
             "แดง": 8,
             "ต้ม": 9,
             "แตงโม": 10,
             "น้ำพริกเผา": 11,
             "บะหมี่": 12,
             "เปรี้ยว": 13,
             "ผัด": 14,
             "ฝรั่ง": 15,
             "พริกแกง": 16,
             "มะม่วง": 17,
             "ม้า": 18,
             "มาม่า": 19,
             "ลูกชิ้นปลา": 20,
             "เลือด": 21,
             "สับ": 22,
             "เส้นเล็ก": 23,
             "เส้นใหญ่": 24,
             "หมู": 25,
             "หวาน": 26,
             "องุ่น": 27,
             "แอปเปิ้ล": 28}
#Config
minPoseConfidence = 0.5
minHandConfidence = 0.5
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

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
    columnHolder.append("right " + columnName)
for columnName in handColumnNameList:
    columnHolder.append("left " + columnName)
df = pd.DataFrame(columns=columnHolder)

#function for adding landmarks on 
def addLandMark(coordinates, index, value, i): #add landmark to list
    for landmark in coordinates[index]:
        value[i] = [landmark.x, landmark.y, landmark.z]
        i += 1
    return value, i
def toDataFrame(imagePATH, label): #convert image path to be added to dataframe
    image = mp.Image.create_from_file(str(imagePATH))
    poseResult = poseLandmarker.detect(image)
    handResult = HandLandmarker.detect(image)
    poseCoordinates = poseResult.pose_landmarks
    handCoordinates = handResult.hand_landmarks
    if len(poseCoordinates) > 0 and len(handCoordinates) > 0: #check if the pose and hand could be detect in the first place
        print("Adding " + imagePATH.name + " to dataframe as " + label + " to index " + str(len(df)))
        value = [0] * 68 #0 = label, 1-25 = pose, 26-46 = right hand 47-67 = left hand
        value[0] = labelList[label] #convert label to number to make it easier to use with neural network
        i = 1
        #add landmarks to list
        for landmark in poseCoordinates[0][:25]:
            value[i] = [landmark.x, landmark.y, landmark.z]
            i += 1
        if len(handCoordinates) > 1:
            value, i = addLandMark(handCoordinates, 0, value, i)
            value, i = addLandMark(handCoordinates, 1, value, i)
        else:
            if handResult.handedness[0][0].category_name == "Left":
                i += 21
                value, i = addLandMark(handCoordinates, 0, value, i)
            else:
                value, i = addLandMark(handCoordinates, 0, value, i)
        #add landmarks to dataframe
        df.loc[len(df)] = value
    else:
        print("Fail to add " + imagePATH.name + " to " + label) #unable to detect either pose or hand coordinates

#"g o o d s t u f f"
imageSubdirectory = inputDirectory.iterdir()
for childDirectory in imageSubdirectory:
    if childDirectory.is_dir():
        for image in childDirectory.glob("**/*.jpg"):   #reading all image in input directory
            label = childDirectory.name                 #saving directory name to use as index name
            toDataFrame(image, label)
print("Output dataframe:")
print(df)
df.to_csv(outputFile)