import tensorflow as tf
import keras
from keras import layers
import pandas as pd
from pathlib import Path
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

keras.mixed_precision.set_global_policy(keras.mixed_precision.Policy('mixed_float16'))
parentDirectory = Path(__file__).parent
testDirectory = parentDirectory.joinpath("Test image")
model = keras.models.load_model(parentDirectory.joinpath("Matrix model full"))
labelList = {0: "กรอบ",
             1: "กระเพรา",
             2: "ขา",
             3: "ข้าว",
             4: "ไข่",
             5: "คะน้า",
             6: "เค็ม",
             7: "โจ๊ก",
             8: "แดง",
             9: "ต้ม",
             10: "แตงโม",
             11: "น้ำพริกเผา",
             12: "บะหมี่",
             13: "เปรี้ยว",
             14: "ผัด",
             15: "ฝรั่ง",
             16: "พริกแกง",
             17: "มะม่วง",
             18: "ม้า",
             19: "มาม่า",
             20: "ลูกชิ้นปลา",
             21: "เลือด",
             22: "สับ",
             23: "เส้นเล็ก",
             24: "เส้นใหญ่",
             25: "หมู",
             26: "หวาน",
             27: "องุ่น",
             28: "แอปเปิ้ล"}

#Pose/Hand detection model config
mediapipeModelDirectory = parentDirectory.joinpath("Mediapipe model")
removeUnusableImage = True
minPoseConfidence = 0.5
minHandConfidence = 0.5
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode
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
poseOption = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=mediapipeModelDirectory.joinpath("pose_landmarker_full.task")),
                                   running_mode=VisionRunningMode.IMAGE,
                                   min_pose_detection_confidence=minPoseConfidence)
handOption = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=mediapipeModelDirectory.joinpath("hand_landmarker.task")),
                                   running_mode=VisionRunningMode.IMAGE,
                                   min_hand_detection_confidence=minHandConfidence,
                                   num_hands=2)
poseLandmarker = PoseLandmarker.create_from_options(poseOption)
HandLandmarker = HandLandmarker.create_from_options(handOption)

def addLandMark(coordinates, index, matrix, i): #add landmark to list
    for landmark in coordinates[index]:
        matrix[i] = [landmark.x, landmark.y, landmark.z]
        i += 1
    return matrix, i
def pictureToMatrix(imagePATH):
    image = mp.Image.create_from_file(str(imagePATH))
    poseResult = poseLandmarker.detect(image)
    handResult = HandLandmarker.detect(image)
    poseCoordinates = poseResult.pose_landmarks
    handCoordinates = handResult.hand_landmarks
    if len(poseCoordinates) > 0 and len(handCoordinates) > 0: #check if the pose and hand could be detect in the first place
        matrix = [[0, 0, 0]] * (len(poseColumnNameList) + len(handColumnNameList) * 2) #0 = label, 1-25 = pose, 26-46 = right hand 47-67 = left hand
        i = 0
        #add landmarks to list
        for landmark in poseCoordinates[0][:25]:
            matrix[i] = [landmark.x, landmark.y, landmark.z]
            i += 1
        if len(handCoordinates) > 1:
            matrix, i = addLandMark(handCoordinates, 0, matrix, i)
            matrix, i = addLandMark(handCoordinates, 1, matrix, i)
        else:
            if handResult.handedness[0][0].category_name == "Left":
                i += 21
                matrix, i = addLandMark(handCoordinates, 0, matrix, i)
            else:
                matrix, i = addLandMark(handCoordinates, 0, matrix, i)
        return matrix
    else:
        #print("Fail to detect: " + str(imagePATH))
        return None

total = 0
correct = 0
unconfidence = 0
for file in testDirectory.glob("*/*.*"):
    matrix = pictureToMatrix(file)
    if matrix != None:
        matrix = np.array(matrix).flatten()
        matrix = matrix.reshape((-1, 67*3, 1))
        matrix = tf.convert_to_tensor(matrix, dtype=tf.float16)
        prediction = model.predict(matrix, verbose=3)
        
        total += 1
        if max(prediction[0]) > 0.5:
            if labelList[np.argmax(prediction)] == file.parent.name:
                correct += 1
        else:
            unconfidence += 1
print(f"total: {total} corect: {correct} unconfidence: {unconfidence} wrong: {total-unconfidence-correct}")