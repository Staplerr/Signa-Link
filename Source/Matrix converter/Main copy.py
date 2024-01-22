import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
from pathlib import Path
import os
import cv2

#path variable
parentPath = Path(__file__).parent
print("Parent directory: " + str(parentPath))
inputDirectory = parentPath.joinpath("Input/")
outputFile = parentPath.joinpath("output" + ".xlsx")
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
removeUnusableImage = True
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
columnNames = ["Label"]
for columnName in poseColumnNameList:
    columnNames.append(columnName)
for columnName in handColumnNameList:
    columnNames.append("right " + columnName)
for columnName in handColumnNameList:
    columnNames.append("left " + columnName)
df = pd.DataFrame(columns=columnNames)

#function for adding landmarks on 
def addLandMark(coordinates, index, value, i): #add landmark to list
    for landmark in coordinates[index]:
        value[i] = [landmark.x, landmark.y, landmark.z]
        i += 1
    return value, i
def toDataFrame(imagePATH, label): #convert image path to be added to dataframe
    image = mp.Image(str(imagePATH))
    poseResult = poseLandmarker.detect(image)
    handResult = HandLandmarker.detect(image)
    poseCoordinates = poseResult.pose_landmarks
    handCoordinates = handResult.hand_landmarks
    if len(poseCoordinates) > 0 and len(handCoordinates) > 0: #check if the pose and hand could be detect in the first place
        print("Adding " + imagePATH.name + " to dataframe as " + label + " to index " + str(len(df)))
        value = [[0, 0, 0]] * (1 + len(poseColumnNameList) + len(handColumnNameList) * 2) #0 = label, 1-25 = pose, 26-46 = right hand 47-67 = left hand
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
        if removeUnusableImage:
            os.remove(imagePATH)

def process_image(image_path, label):
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_landmarks = mp.solutions.hands.Hands().process(image_rgb)

    if image_landmarks.multi_hand_landmarks:
        hand_landmarks = image_landmarks.multi_hand_landmarks[0]  # Assuming you are processing one hand
        landmarks_list = []
        for landmark in hand_landmarks.landmark:
            landmarks_list.extend([landmark.x, landmark.y, landmark.z])
        
        # Add landmarks to DataFrame
        df.loc[len(df)] = [labelList[label]] + landmarks_list

# Initiate DataFrame
columnNames = ["Label"] + [f"Landmark_{i}" for i in range(1, 22 * 3 + 1)]  # Assuming 21 hand landmarks
df = pd.DataFrame(columns=columnNames)

# Save landmarks for each image
image_folder_path = parentPath.joinpath("Input")  # Assuming images are in the "Input" folder
label = "test"  # Change this label as needed

for image_path in image_folder_path.glob("*.png"):  # Change the extension based on your image format
    process_image(image_path, label)

# Save DataFrame to Excel
output_file = parentPath.joinpath("output.xlsx")
df.to_excel(output_file, index=False)
