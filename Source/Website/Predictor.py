import tensorflow as tf
import keras
from pathlib import Path
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from binascii import a2b_base64

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

parentDirectory = Path(__file__).parent
imageDirectory = parentDirectory.joinpath("Image")
model = keras.models.load_model(parentDirectory.joinpath("Model"))
keras.mixed_precision.set_global_policy(keras.mixed_precision.Policy('mixed_float16'))
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

def initiateMediapipeModel():
    #Pose/Hand detection model config
    mediapipeModelDirectory = parentDirectory.joinpath("Mediapipe model")
    minPoseConfidence = 0.5
    minHandConfidence = 0.5
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    VisionRunningMode = vision.RunningMode
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
    return poseLandmarker, HandLandmarker
poseLandmarker, HandLandmarker = initiateMediapipeModel()

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
        matrix = [[0, 0, 0]] * 67
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
        app.logger.info(f"Fail to detect: {imagePATH}")
        return None

@app.route('/predictImage', methods=['POST'])
def predictImage():
    #Convert the dataURL received into png and save the path of the image
    #Convert the dataURL into png because it could be convert to mediapipe image more easily
    imageReceived = request.get_data()
    binaryImage = a2b_base64(imageReceived[22:])
    fd = open(str(imageDirectory.joinpath("image.png")), 'wb')
    fd.write(binaryImage)
    fd.close()
    app.logger.info(f"Image saved")
    imagePATH = imageDirectory.joinpath("image.png")

    #Create dictionary holder
    dataDict = {"label" : None,
                "confidence" : None,
                "inferenceTime" : None}

    startTime = time.perf_counter()
    matrix = pictureToMatrix(imagePATH)
    if matrix != None:
        #Preprocess matrix and save the prediction
        matrix = np.array(matrix).flatten()
        matrix = matrix.reshape((-1, 67*3, 1))
        matrix = tf.convert_to_tensor(matrix, dtype=tf.float16)
        prediction = model.predict(matrix, verbose=3)

        #Add data to dictionary
        dataDict["inferenceTime"] = time.perf_counter() - startTime
        dataDict["predictedLabel"] = labelList[np.argmax(prediction)]
        dataDict["confidence"] = max(prediction[0]) * 100
        app.logger.info(dataDict)
        return jsonify(dataDict) #Convert dictionary to json
    return jsonify(dataDict)

@app.route('/APIpostTest', methods=['POST'])
def APIpostTest():
    data = request.json
    app.logger.info(f"receive POST request with data: {data}")
    return jsonify(data)

@app.route('/APIgetTest', methods=['GET'])
def APIgetTest():
    data = {"data" : "hi",
            "data2" : "hello"}
    app.logger.info(f"receive GET request, returned data: {data}")
    return jsonify(data)

#if __name__ == "__main__":
#    print("running")
#    app.run(debug=True)
#    print("done")