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

#Directory
parentDirectory = Path(__file__).parent
inputDirectory = parentDirectory.joinpath("Videos")
tempDirectory = parentDirectory.joinpath("Temp")
if not tempDirectory.exists():
    tempDirectory.mkdir(parents=True)
outputFile = parentDirectory.joinpath("Output.csv")

#Opencv config
baseSample = 5 #Save frame every n frame
frameBuffer = 10 #Number of frame that will be included inside the dataframe
resizeRation = 10
resizeInterpolation = cv2.INTER_AREA

#mp models
modelDirectory = parentDirectory.joinpath("Model")
handModel = modelDirectory.joinpath("hand_landmarker.task")
poseModel = modelDirectory.joinpath("pose_landmarker_full.task")

#For reading all file in video folder
supportsExtension = ["*/*.mp4", "*/*.mov"]

#Dataframe
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
        for axis in ["X", "Y", "Z"]:
            columnNames.append(f"{columnName}_{axis}_{i}")
    for columnName in handColumnNameList:
        for axis in ["X", "Y", "Z"]:
            columnNames.append(f"right_{columnName}_{axis}_{i}")
    for columnName in handColumnNameList:
        for axis in ["X", "Y", "Z"]:
            columnNames.append(f"left_{columnName}_{axis}_{i}")
df = pd.DataFrame(columns=columnNames)

labelList = {"กรอบ": 0,     "กิน": 1,    "ข้าว": 2,       "คุณสบายดีไหม": 3,
             "ผัด": 4,       "สวัสดี": 5,      "หมู": 6,       "ไหน": 7,
             "อยู่": 8,}

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
    return videoPATHs


def addLandmarks(coordinates, array): #Function for adding new landmarks to array
    if type(coordinates[0]) == mpLandmark.Landmark:
        for landmark in coordinates:
            value = np.array([landmark.x, landmark.y, landmark.z], dtype=np.float16)
            array = np.vstack([array, value])
    else:
        for filler in coordinates:
            array = np.vstack([array, filler])
    return array #Return 2D np array


def generateFrameLandmarks(frame): #Function for generating landmarks for single frame
    frame = mp.Image.create_from_file(frame)
    poseResult = poseLandmarker.detect(image=frame)
    poseCoordinates = poseResult.pose_world_landmarks
    if len(poseCoordinates) == 0:
        return None
    handResult = handLandmarker.detect(image=frame)
    handedness = handResult.handedness 
    handCoordinates = handResult.hand_world_landmarks

    coordinatesArray = np.empty((3, ), dtype=np.float16)
    coordinatesArray = addLandmarks(poseCoordinates[0][:25], coordinatesArray)
    if len(handedness) == 0: #check if no hand is detect
        for i in range(2):
            filler = np.zeros(shape=(len(handColumnNameList), 3), dtype=np.float16)
            coordinatesArray = addLandmarks(filler, coordinatesArray)
    else: #execute if hand is detect
        for index, category in enumerate(handedness):
            if len(handCoordinates) == 1: #check if mp detect only one hand
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
    coordinatesArray = np.delete(coordinatesArray, 0, axis=0) #remove the first element that got create when declare the empty array

    return coordinatesArray #return 2D np array


def removeExcessFrames(frameList, sample): #Convert from frames to landmarks of video
    landmarksArray = []
    middlePosition = int(np.ceil(len(frameList) / 2)) - 1 #Middle position of input array, median?
    currentPosition = [middlePosition, middlePosition] #Use for calculate which is the next position needed to be add to array.
    addToBack = True

    while len(landmarksArray) < frameBuffer:
        direction = int(addToBack) #0 = front, 1 = back
        positionToAdd = direction * len(landmarksArray) #Position to add element to the array that got return
        valuePosition = currentPosition[direction] + (sample * (direction * 2 - 1)) #Position in the frameList that will be added to return array
        while True:
            try:
                if valuePosition >= 0: #Preventing from adding the value that have been count from the back of frameList.
                    landmark = generateFrameLandmarks(str(frameList[valuePosition]))
                    if type(landmark) == np.ndarray:
                        break
                #print(f"Error raised, value position: {valuePosition}, current position: {currentPosition[direction]}, direction: {direction}, position to add: {positionToAdd}, landmark: {landmark}")
                raise IndexError
            except IndexError: #Only execute when the value position is below zero
                valuePosition -= (direction * 2 - 1)
        landmarksArray.insert(positionToAdd, landmark)
        currentPosition[direction] = valuePosition
        addToBack = not addToBack

    #return array
    return np.array(landmarksArray, dtype=np.float16)


def videoToLandmarks(videoPATH, sample):
    cap = cv2.VideoCapture(str(videoPATH))
    frameList = []
    currentFrame = 0
    totalFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    landmarks = []

    if totalFrame < frameBuffer:
        return None
    while totalFrame < frameBuffer * sample and sample != 1:
        sample -= 1
    print(f"Video: {videoPATH.name} Total frame: {totalFrame} Sample: {sample}")
    #capture all frame the video has
    while cap.isOpened:
        ret, frame = cap.read()
        if ret:
            #np array = impossible to detect hand, image file = really easy to detect
            file = tempDirectory.joinpath(f"{videoPATH.name}_{currentFrame}.png")
            height = int(np.floor(frame.shape[0] / resizeRation))
            width = int(np.floor(frame.shape[1] / resizeRation))
            frame = cv2.resize(frame, (width, height), interpolation=resizeInterpolation)
            cv2.imwrite(str(file), frame)
            frameList.append(file)
            currentFrame += 1
        else:
            break
    cap.release()
    landmarks = removeExcessFrames(frameList, sample)
    for file in frameList:
        os.remove(file)

    return landmarks #Return "3D" np array


startTime = time.perf_counter()

videoPaths = getFilePATHS(inputDirectory)
totalFile = len(videoPaths)
for index, video in enumerate(videoPaths):
    label = labelList[video.parent.name]

    landmarks = videoToLandmarks(video, baseSample)
    if type(landmarks) != np.ndarray:
        print(f"Skip {video.name}")
        continue
    landmarks = landmarks.reshape((-1))
    landmarks = landmarks.tolist()
    landmarks.insert(0, label)

    df.loc[len(df)] = landmarks
    print(f"Progress: {index + 1} / {totalFile}")
dataProcessTime = time.perf_counter()

df = df.sample(frac=1) #Shuffle dataframe
shuffleTime = time.perf_counter()
print(df)
df.to_csv(outputFile, index=False)
finishTime = time.perf_counter()
print(f"Data process time: {dataProcessTime - startTime} second")
print(f"Shuffle time: {shuffleTime - dataProcessTime} second")
print(f"Save time: {finishTime - shuffleTime} second")