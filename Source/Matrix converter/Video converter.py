import mediapipe as mp
from mediapipe.tasks.python import vision
import pandas as pd
import time
import cv2
from pathlib import Path
import time
import multiprocessing

#path variable
parentDirectory = Path(__file__).parent
inputDirectory = parentDirectory.joinpath("Videos")
outputFile = parentDirectory.joinpath("image output" + ".xlsx")
modelDirectory = parentDirectory.joinpath("Model")
handModel = modelDirectory.joinpath("hand_landmarker.task")
poseModel = modelDirectory.joinpath("pose_landmarker_full.task")

#video reading config
supportsExtension = ["*/*.mp4", "*/*.mov"]
sample = 5 #Save frame every n frame
processes = []
processesCount = 10
frameBuffer = 10

#initiate dataframe
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
        columnNames.append(f"columnName_{i}")
    for columnName in handColumnNameList:
        columnNames.append(f"right_{columnName}_{i}")
    for columnName in handColumnNameList:
        columnNames.append(f"left_{columnName}_{i}")
df = pd.DataFrame(columns=columnNames)
#list for converting label into a number
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

#function for adding landmarks on 
def addLandMark(coordinates, index, value, i): #add landmark to list
    for landmark in coordinates[index]:
        value[i] = [landmark.x, landmark.y, landmark.z]
        i += 1
    return value, i
def toDataFrame(frames, label, nextRow): #convert image path to be added to dataframe
    value = [[0, 0, 0]] * len(columnNames)
    value[0] = labelList[label] #convert label to number to make it easier to use with neural network
    i = 1
    for index, frame in enumerate(frames):
        i = 1 + (len(poseColumnNameList) + len(handColumnNameList) * 2) * index
        #print(f"column: {len(poseColumnNameList) + len(handColumnNameList) * 2}")
        #print(f"index: {index}")
        #print(f"column * index: {(len(poseColumnNameList) + len(handColumnNameList) * 2) * index}")
        #print(f"i: {i}")
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=(frame))
        poseResult = poseLandmarker.detect(frame)
        handResult = HandLandmarker.detect(frame)
        poseCoordinates = poseResult.pose_landmarks
        handCoordinates = handResult.hand_landmarks
        if len(poseCoordinates) > 0: #check if the pose and hand could be detect in the first place
            #add landmarks to list
            for landmark in poseCoordinates[0][:25]:
                value[i] = [landmark.x, landmark.y, landmark.z]
                i += 1
            if len(handCoordinates) == 2:
                value, i = addLandMark(handCoordinates, 0, value, i)
                value, i = addLandMark(handCoordinates, 1, value, i)
            elif len(handCoordinates) == 1:
                if handResult.handedness[0][0].category_name == "Left":
                    i += 21
                value, i = addLandMark(handCoordinates, 0, value, i)
            
    #print(f"Value: {len(value)}")
    #print(f"column: {len(df.columns)}")
                
    #add landmarks to dataframe
    df.loc[nextRow] = value

def saveFrames(videoPATHs, nextrow):
    for videoPATH in videoPATHs:
        flipping = True #don't even know what to name this
        frames = []
        label = videoPATH.parent.name
        print(f"Adding: {videoPATH.name} to dataframe as: {label} to index: {str(len(df))}")

        cap = cv2.VideoCapture(str(videoPATH))
        currentFrame = 0

        while cap.isOpened:
            ret, frame = cap.read()
            if ret:
                if currentFrame / sample - currentFrame // sample == 0: #check if the current frame is the sample frame
                    frames.append(frame)
                    #cv2.imshow(f"name: {videoPATH} frame: {currentFrame}", frame)
                currentFrame += 1
            else:
                break
        while len(frames) > frameBuffer: #remove frame untile the frames list contain only 10 frame
            if flipping:
                frames.pop(0)
            else:
                frames.pop()
            flipping = not flipping
        cap.release()
        toDataFrame(frames, label, nextrow)
        nextrow += 1

startTime = time.perf_counter()

print(inputDirectory)
videoPATHs = getFilePATHS(inputDirectory)
#videoPATHsList = list(splitList(videoPATHs))
#del videoPATHs #hopefully it will freeup some memory

saveFrames(videoPATHs, 0)

#nextRow = 0 #don't even know what to name this
#for i in range(processesCount): #initiate process
#    p = multiprocessing.Process(target=saveFrames, args=[videoPATHsList[i], nextRow])
#    p.start()
#    processes.append(p)
#    nextRow += len(videoPATHsList[i])
#for process in processes: #wait for process to end
#    process.join()
#
#del videoPATHsList
#del processes

df = df.sample(frac=1) #Shuffle dataframe
print("Output dataframe:")
print(df)
df.to_excel(outputFile, index=False)

finishTime = time.perf_counter()
print(f"total time: {finishTime - startTime}")