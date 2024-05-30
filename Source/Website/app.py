import tensorflow as tf
import keras
from pathlib import Path
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import landmark as mpLandmark
import time
from datetime import datetime
from flask import Flask, request, jsonify, session, render_template
from flask_session import Session
import logging
from binascii import a2b_base64
import os
import sys

app = Flask(__name__)
#CORS(app)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "cachelib"
Session(app)
logging.basicConfig(level=logging.DEBUG)

#Directory
parentDirectory = Path(__file__).parent
tempDirectory = parentDirectory.joinpath("temp")
if not tempDirectory.exists():
    tempDirectory.mkdir(parents=True)
logDirectory = parentDirectory.joinpath("logs")
if not logDirectory.exists():
    logDirectory.mkdir(parents=True)
logFile = logDirectory.joinpath("log.txt")

#frames config
sample = 5 #Save frame every n frame
frameBuffer = 10 #Number of frame that will be included inside the dataframe
retryChance = 2

#Matrix model stuff static/model/matrix_model.keras
matrixModel = keras.models.load_model(parentDirectory.joinpath("static/model/matrix_model"))
keras.mixed_precision.set_global_policy(keras.mixed_precision.Policy('float32'))
labelList = ["กรอบ",     "กิน",    "ข้าว",       "คุณสบายดีไหม",
             "ผัด",       "สวัสดี",      "หมู",       "ไหน",
             "อยู่",]
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


def initiateMediapipeModel():
    #Pose/Hand detection model config
    mediapipeModelDirectory = parentDirectory.joinpath("static/model/mediapipe_model")
    poseModel = mediapipeModelDirectory.joinpath("pose_landmarker_full.task")
    handModel = mediapipeModelDirectory.joinpath("hand_landmarker.task")

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
    return poseLandmarker, HandLandmarker
poseLandmarker, handLandmarker = initiateMediapipeModel()
    

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
    try:
        frame = mp.Image.create_from_file(frame)
    except: #Occur when saving image gone wrong, don't know how but it did.
        return None

    mpPredictStart = time.perf_counter()
    poseResult = poseLandmarker.detect(image=frame)
    poseCoordinates = poseResult.pose_world_landmarks
    if len(poseCoordinates) == 0:
        return None
    handResult = handLandmarker.detect(image=frame)
    handedness = handResult.handedness
    handCoordinates = handResult.hand_world_landmarks
    mpPredictTime = time.perf_counter() - mpPredictStart

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
    coordinatesArray = np.nan_to_num(coordinatesArray) #Replace nan with 0
    return [coordinatesArray, mpPredictTime] #return 2D np array and float

@app.route('/predictImage', methods=['POST'])
def predictImage():
    currentFrame = session["currentFrame"]
    #Convert the dataURL received into png and save the path of the image
    imageReceived = request.get_data()
    binaryImage = a2b_base64(imageReceived[22:])
    fd = open(str(tempDirectory.joinpath(f"frame_{currentFrame}.png")), 'wb')
    fd.write(binaryImage)
    fd.close()

    #Create dictionary holder
    dataDict = {"label" : None,
                "confidence" : None,
                "inferenceTime" : None}

    if currentFrame / sample - currentFrame // sample == 0:
        imagePATH = tempDirectory.joinpath(f"frame_{currentFrame}.png")
        mpResult = generateFrameLandmarks(str(imagePATH))

        if type(mpResult) != list: #Give it another chance when did not detect body
            for i in range(retryChance):
                imagePATH = tempDirectory.joinpath(f"frame_{currentFrame - i}.png")
                try:
                    mpResult = generateFrameLandmarks(str(imagePATH))
                    if type(mpResult) == list: break
                except: pass

        if type(mpResult) == list:
            session["landmarks"] = np.vstack([mpResult[0], session["landmarks"]], dtype=np.float16)
            session["landmarks"] = session["landmarks"][:-(len(poseColumnNameList) + len(handColumnNameList) * 2)]
            landmarks = session["landmarks"]

            processedLandmark = landmarks.reshape((-1, 3 * frameBuffer * (len(poseColumnNameList) + len(handColumnNameList) * 2)))

            nnPredictStart = time.perf_counter()
            prediction = matrixModel.predict(processedLandmark, verbose=3)

            #Add data to dictionary
            dataDict["inferenceTime"] = {"Neural network" : time.perf_counter() - nnPredictStart,
                                         "Mediapipe" : mpResult[1]}
            dataDict["label"] = labelList[np.argmax(prediction)]
            dataDict["confidence"] = prediction[0][np.argmax(prediction[0])] * 100
            
            app.logger.info(f"Returned: {dataDict}")
            log = open(str(logFile), "a")
            log.write(f"Time: {datetime.now()} Returned: {dataDict} Landmarks: {landmarks}\n\n")
            log.close()

    try:
        os.remove(str(tempDirectory.joinpath(f"frame_{currentFrame - 5}.png")))
    except: pass

    session["currentFrame"] += 1
    return jsonify(dataDict) #Convert dictionary to json


@app.route('/')
def homePage():
    session["landmarks"] = np.empty([frameBuffer * (len(poseColumnNameList) + len(handColumnNameList) * 2), 3], dtype=np.float16)
    session["currentFrame"] = 0
    return render_template('home.html')


@app.route('/about')
def infoPage():
    return render_template('about.html')

@app.route('/dataset')
def datasetPage():
    return render_template('dataset.html')

if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    app.run()