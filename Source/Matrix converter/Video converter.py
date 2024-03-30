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
sample = 5 #Save frame every n frame
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
    return videoPATHs


def addLandmarks(coordinates, array):
    if type(coordinates[0]) == mpLandmark.Landmark:
        for landmark in coordinates:
            value = np.array([landmark.x, landmark.y, landmark.z], dtype=np.float16)
            array = np.vstack([array, value])
    else:
        for filler in coordinates:
            array = np.vstack([array, filler])
    return array #Return 2D np array


def generateFrameLandmarks(frame):
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


def removeExcessLandmarks(startArray): #This definitely will break if length of input array is lower than sample * frameBuffer
    array = []
    middlePosition = int(np.ceil(len(startArray) / 2)) - 1 #Middle position of input array, median?
    array.append(startArray[middlePosition])
    currentValuePosition = [middlePosition, middlePosition] #Use for calculate which is the next position needed to be add to array.
    addToBack = True
    
    #Jam landmarks into available spot
    while len(array) < frameBuffer:
        direction = int(addToBack) #0 = front, 1 = back
        positionToAdd = direction * len(array) #Position to add element to the array that got return
        valuePosition = currentValuePosition[direction] + (sample * (direction * 2 - 1)) #Position in the startArray that will be added to return array
        #Check if the index is out of range, if so then it will move closer to last position
        while True:
            try:
                if valuePosition < 0: #Preventing from adding the value that have been count from the back of startArray.
                    raise IndexError
                array.insert(positionToAdd, startArray[valuePosition])
                break
            except IndexError:
                valuePosition -= (direction * 2 - 1)
        currentValuePosition[direction] = valuePosition
        addToBack = not addToBack

    #return array
    return np.array(array, dtype=np.float16)


def videoToLandmarks(videoPATH):
    landmarks = []
    cap = cv2.VideoCapture(str(videoPATH))
    currentFrame = 0

    #capture all frame the video has
    while cap.isOpened:
        ret, frame = cap.read()
        if ret:
            file = tempDirectory.joinpath(f"{videoPATH.name}_{currentFrame}.png")
            #np array = impossible to detect hand, image file = really easy to detect
            height = int(np.floor(frame.shape[0] / resizeRation))
            width = int(np.floor(frame.shape[1] / resizeRation))
            frame = cv2.resize(frame, (width, height), interpolation=resizeInterpolation)
            cv2.imwrite(str(file), frame)

            landmarkResult = generateFrameLandmarks(str(file))
            if type(landmarkResult) == np.ndarray: #Prevent from adding frame that no pose has been detected
                landmarks.append(landmarkResult) #Array of 2D np array
            os.remove(file)
            currentFrame += 1
        else:
            break
    cap.release()
    
    landmarks = removeExcessLandmarks(landmarks)
    return landmarks #Return "3D" np array


startTime = time.perf_counter()

videoPaths = getFilePATHS(inputDirectory)
totalFile = len(videoPaths)
for index, video in enumerate(videoPaths):
    label = labelList[video.parent.name]

    landmarks = videoToLandmarks(video)
    landmarks = landmarks.reshape((-1, 3))
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