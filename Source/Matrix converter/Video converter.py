import mediapipe as mp
from mediapipe.tasks.python.components.containers import landmark as mpLandmark
from mediapipe.tasks.python import vision
import pandas as pd
import time
import cv2
from pathlib import Path
import time
import os
import numpy as np

parentDirectory = Path(__file__).parent
inputDirectory = parentDirectory.joinpath("Videos")
outputFile = parentDirectory.joinpath("image output" + ".xlsx")
modelDirectory = parentDirectory.joinpath("Model")
handModel = modelDirectory.joinpath("hand_landmarker.task")
poseModel = modelDirectory.joinpath("pose_landmarker_full.task")

supportsExtension = ["*/*.mp4", "*/*.mov"]
sample = 5 #Save frame every n frame
frameBuffer = 10
processes = []
processesCount = 10

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
columnNames = ["Label"]

for i in range(frameBuffer):
    for columnName in poseColumnNameList:
        columnNames.append(f"{columnName}_{i}")
    for columnName in handColumnNameList:
        columnNames.append(f"right_{columnName}_{i}")
    for columnName in handColumnNameList:
        columnNames.append(f"left_{columnName}_{i}")
df = pd.DataFrame(columns=columnNames)

labelList = {"กรอบ": 0,     "กระเพรา": 1,    "ขา": 2,       "ข้าว": 3,
             "ไข่": 4,       "คะน้า": 5,      "เค็ม": 6,       "โจ๊ก": 7,
             "แดง": 8,      "ต้ม": 9,        "แตงโม": 10,    "น้ำพริกเผา": 11,
             "บะหมี่": 12,    "เปรี้ยว": 13,    "ผัด": 14,       "ฝรั่ง": 15,
             "พริกแกง": 16,  "มะม่วง": 17,    "ม้า": 18,       "มาม่า": 19,
             "ลูกชิ้นปลา": 20, "เลือด": 21,     "สับ": 22,       "เส้นเล็ก": 23,
             "เส้นใหญ่": 24,  "หมู": 25,       "หวาน": 26,     "องุ่น": 27,
             "แอปเปิ้ล": 28}

#mediapipe config
minPoseConfidence = 0.5
minHandConfidence = 0.5
baseOptions = mp.tasks.BaseOptions
poseLandmarker = vision.PoseLandmarker
poseLandmarkerOptions = vision.PoseLandmarkerOptions
handLandmarker = vision.HandLandmarker
handLandmarkerOptions = vision.HandLandmarkerOptions
visionRunningMode = vision.RunningMode

#create the landmarker object
poseOption = poseLandmarkerOptions(base_options=baseOptions(model_asset_path=poseModel),
                                   running_mode=vision.RunningMode.IMAGE,
                                   min_pose_detection_confidence=minPoseConfidence)
handOption = handLandmarkerOptions(base_options=baseOptions(model_asset_path=handModel),
                                   running_mode=visionRunningMode.IMAGE,
                                   min_hand_detection_confidence=minHandConfidence,
                                   num_hands=2)
poseLandmarker = poseLandmarker.create_from_options(poseOption)
handLandmarker = handLandmarker.create_from_options(handOption)

def getFilePATHS(directory):
    videoPATHs = []
    for extension in supportsExtension: #Collect file that has mp4 and mov file extension
        for file in directory.glob(extension):
            videoPATHs.append(file)
    print(f"total files: {len(videoPATHs)}")
    return videoPATHs

def splitList(list):
    step = len(list) // processesCount
    remain = len(list) % processesCount
    for i in range(0, len(list), step):
        yield list[i:i + step + remain] #return multiple 1D list
        remain = 0

#Re-written
def addLandmarks(coordinates, array):
    if type(coordinates[0]) == mpLandmark.NormalizedLandmark:
        for landmark in coordinates:
            value = np.array([landmark.x, landmark.y, landmark.z], dtype=np.float16)
            array = np.vstack([array, value])
    else:
        for filler in coordinates:
            array = np.vstack([array, filler])
    return array

def generateFrameLandmarks(frame):
    frame = mp.Image.create_from_file(frame)
    poseResult = poseLandmarker.detect(frame)
    handResult = handLandmarker.detect(frame)
    poseCoordinates = poseResult.pose_landmarks
    handedness = handResult.handedness 
    handCoordinates = handResult.hand_landmarks

    coordinatesArray = np.empty((3, ), dtype=np.float16)
    coordinatesArray = addLandmarks(poseCoordinates[0][:25], coordinatesArray)
    coordinatesArray = np.delete(coordinatesArray, 0, axis=0) #remove the first element that got create when declaire the empty array
    for index, category in enumerate(handedness):
        if len(handCoordinates) < 2: #check if mp detect only one hand
            filler = np.zeros(shape=(len(handColumnNameList), 3), dtype=np.float16)
            if category[index].index == 0: #detect right
                coordinatesArray = addLandmarks(handCoordinates[index], coordinatesArray)
                coordinatesArray = addLandmarks(filler, coordinatesArray)
            else: #detect left
                coordinatesArray = addLandmarks(filler, coordinatesArray)
                coordinatesArray = addLandmarks(handCoordinates[index], coordinatesArray)
            break
        else:
            coordinatesArray = addLandmarks(handCoordinates[index], coordinatesArray)
    print(coordinatesArray)
    print(len(coordinatesArray))
    return coordinatesArray #return np array

bothIMG = parentDirectory.joinpath('Images/กรอบ/IMG_0189_50.png')
leftIMG = parentDirectory.joinpath('Images/กระเพรา/IMG_0199_25.png')
rightIMG =parentDirectory.joinpath('Images/เปรี้ยว/VID_20240123200118_0.png')

path = rightIMG
label = path.parent
generateFrameLandmarks(str(path), str(label))