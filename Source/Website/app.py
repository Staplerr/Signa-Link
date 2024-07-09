import tensorflow as tf
import json
import keras
from pathlib import Path
import numpy as np
import mediapipe as mp
import time
from google.protobuf.json_format import MessageToDict
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

#Matrix model stuff
keras.mixed_precision.set_global_policy(keras.mixed_precision.Policy('float32'))
matrixModel = keras.models.load_model(parentDirectory.joinpath("static/model/matrix_model.h5"))
f = open(str(parentDirectory.joinpath("static/model/label.json")))
labels = json.load(f)
f.close
handsLandmarkSpot = 21

mp_hands = mp.solutions.hands
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4) as hands:
    
    def push(
            npArray: np.ndarray,
            value
        ):
        """
            Push new variable into np array and remove the last one
        """
        npArray = np.concatenate(([npArray, value]), axis=0)
        npArray = np.delete(npArray, -1, axis=0)
        return npArray

    array = np.empty((10,42,3), dtype=np.float32)
    print(array)
    print(f"Before shape: {array.shape}")
    array = push(array, np.ones((1,42, 3)))
    print(array)
    print(f"After shape: {array.shape}")

    def preprocessImage():
        pass

    def landmarker(image):
        results = hands.process(image)
        Coordinates = np.empty((0,3), dtype=np.float32)

        if not results.multi_hand_landmarks:
            return None

        if len(results.multi_handedness) == 1: # เจอข้างเดียว
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                handedness_dict = MessageToDict(handedness)
                if handedness_dict["classification"][0]["index"] == 0:
                    for landmark in hand_landmarks.landmark:
                        Coordinates = np.concatenate((Coordinates, [[landmark.x, landmark.y, landmark.z]]), axis=0)
                    for i in range(21): # Fill
                        Coordinates = np.concatenate((Coordinates, [[0 for i in range(3)]]), axis=0)
                elif handedness_dict["classification"][0]["index"] == 1:
                    for i in range(21): # Fill
                        Coordinates = np.concatenate((Coordinates, [[0 for i in range(3)]]), axis=0)
                    for landmark in hand_landmarks.landmark:
                        Coordinates = np.concatenate((Coordinates, [[landmark.x, landmark.y, landmark.z]]), axis=0)
                else:
                    continue
        if len(results.multi_handedness) == 2: # เจอสองข้าง
            for landmark in results.multi_hand_landmarks[0].landmark:
                Coordinates = np.concatenate((Coordinates, [[landmark.x, landmark.y, landmark.z]]), axis=0)
            for landmark in results.multi_hand_landmarks[1].landmark:
                Coordinates = np.concatenate((Coordinates, [[landmark.x, landmark.y, landmark.z]]), axis=0)


        return array

    @app.route('/predictImage', methods=['POST'])
    def prediction(image):
        dataDict = {"label" : None,
                    "confidence" : None,
                    "inferenceTime" : {
                        "neuralNetwork" : None,
                        "mediaPipe" : None
                        }
                    }
        startTime = time.perf_counter()
        return jsonify(dataDict)

    @app.route('/')
    def homePage():
        session["landmarks"] = np.empty([frameBuffer * (len(handsLandmarkSpot) * 2), 3], dtype=np.float16)
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


"""
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
    coordinatesArray = np.nan_to_num(coordinatesArray) #Replace nan with 0
    return coordinatesArray #return 2D np array

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
        startTime = time.perf_counter()
        imagePATH = tempDirectory.joinpath(f"frame_{currentFrame}.png")
        landmarkResult = generateFrameLandmarks(str(imagePATH))

        if type(landmarkResult) != np.ndarray: #Give it another chance when did not detect body
            for i in range(retryChance):
                imagePATH = tempDirectory.joinpath(f"frame_{currentFrame - i}.png")
                try:
                    landmarkResult = generateFrameLandmarks(str(imagePATH))
                    if type(landmarkResult) == np.ndarray: break
                except: pass

        if type(landmarkResult) == np.ndarray:
            session["landmarks"] = np.vstack([landmarkResult, session["landmarks"]], dtype=np.float16)
            session["landmarks"] = session["landmarks"][:-(len(poseColumnNameList) + len(handColumnNameList) * 2)]
            landmarks = session["landmarks"]

            processedLandmark = landmarks.reshape((-1, 3 * frameBuffer * (len(poseColumnNameList) + len(handColumnNameList) * 2)))

            prediction = matrixModel.predict(processedLandmark, verbose=3)

            #Add data to dictionary
            dataDict["inferenceTime"] = time.perf_counter() - startTime
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
"""