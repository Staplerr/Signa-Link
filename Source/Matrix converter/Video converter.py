import mediapipe as mp
import pandas as pd
import time
import cv2
from pathlib import Path
import os
import tensorflow as tf

# Directory
parentDirectory = Path(__file__).parent
inputDirectory = parentDirectory.joinpath("Videos")
tempDirectory = parentDirectory.joinpath("Temp")
if not tempDirectory.exists():
    tempDirectory.mkdir(parents=True)
outputFile = parentDirectory.joinpath("Output.csv")

# Opencv config
resizeRatio = [5, 9]
resizeInterpolation = cv2.INTER_AREA

# For reading all files in the video folder
supportsExtension = ["**/*.mp4", "**/*.mov"]

# Dataframe
labelList = {}
for index, word in enumerate(inputDirectory.iterdir()):
    labelList[word.name] = index
    labelList[index] = word.name

# mediapipe config
minPoseConfidence = 0.5
minHandConfidence = 0.5
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

videoPaths = {}
for index, directory in enumerate(inputDirectory.iterdir()):
    for file in directory.iterdir():
        videoPaths[file] = index
        
totalFile = len(videoPaths)
tensorMatrix = []
counter = 0
Data = tf.constant([],shape=(0,0), dtype=tf.float16)
Label = tf.constant([],shape=(0,), dtype=tf.float16)
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    for video, index in videoPaths.items():
        cap = cv2.VideoCapture(str(video))
        counter += 1
        Frame = 0
        Label = tf.concat([Label, tf.constant([float(index)], dtype=tf.float16)], axis=0)
        videoTensor = tf.constant([],shape=(0,), dtype=tf.float16)
      	while cap.isOpened():
        		success, image = cap.read()
			if not success:
          		print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          		break

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        		image.flags.writeable = False
        		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        		image = cv2.resize(image, (image.shape[0] // resizeRatio[0], image.shape[1] // resizeRatio[1]), interpolation=resizeInterpolation)
        		results = hands.process(image)
        		if results.multi_hand_landmarks:
            		for hand_landmarks in results.multi_hand_landmarks:
                		for handedness in results.multi_handedness:
                    		for i in range(len(hand_landmarks.landmarks)):
                        		if headedness[i].index == 0 and headedness[i-1] != 0:
                        			for corrdinate in hand_landmarks.landmarks[i]: #return x y z
                                		tf.concat(videoTensor, tf.constant([corrdinate.x, corrdinate.y, corrdinate.z], dtype=tf.float16)
                        		elif headedness[i].index == 1 and headedness[i-1] != 1:
                            		pass
                        		else:
                            		break
        # Flip the image horizontally for a selfie-view display.
        if Frame % 10 == 0:
            Data = tf.stack([Data, videoTensor], axis=0)
            Label = tf.stack([Label, tf.constant([float(index]), dtype=tf.float16)], axis=0)
            videoTensor = tf.constant([],shape=(0,), dtype=tf.float16)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
          break
    cap.release()
    cv2.destroyAllWindows()

"""videoPaths = {}
for index, directory in enumerate(inputDirectory.iterdir()):
    for file in directory.iterdir():
        videoPaths[file] = index
        
totalFile = len(videoPaths)
tensorMatrix = []
counter = 0
Data = tf.constant([],shape=(0, 0, 2, 3), dtype=tf.float16)
Label = tf.constant([],shape=(0,), dtype=tf.float16)
for video, index in videoPaths.items():
    cap = cv2.VideoCapture(str(video))
    Label = tf.concat([Label, tf.constant([float(index)], dtype=tf.float16)], axis=0)
    VideoTensor = tf.constant([],shape=(0, 2, 3), dtype=tf.float16)
    print(Label)
    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            file = tempDirectory.joinpath("frame.png")
            frame = cv2.resize(img, (img.shape[0] // resizeRatio[0], img.shape[1] // resizeRatio[1]), interpolation=resizeInterpolation)
            cv2.imwrite(str(file), frame)
            frame = mp.Image.create_from_file(str(file))
            poseResult = poseLandmarker.detect(image=frame)
            handResult = handLandmarker.detect(image=frame)
            poseCoordinates = poseResult.pose_world_landmarks
            handedness = handResult.handedness
            handCoordinates = handResult.hand_world_landmarks
            print(handResult.multi_hand_landmarks)
            print(handResult.multi_handedness)
            #if handCoordinates != []:
            #    print(len(handCoordinates))
            #    print(handedness[0][0])
                #print(handCoordinates[0][0])
                #frameTensor = tf.stack([tf.constant([float("%.4f" % handCoordinates[0][0].x), float("%.4f" % handCoordinates[0][0].y), float("%.4f" % handCoordinates[0][0].z)], dtype=tf.float16),
                #                         tf.constant([float("%.4f" % handCoordinates[0][1].x), float("%.4f" % handCoordinates[0][1].y), float("%.4f" % handCoordinates[0][1].z)], dtype=tf.float16)], axis=0)
                #VideoTensor = tf.concat([VideoTensor, tf.reshape(frameTensor, (1, 2, 3))], axis=0)
                #print(VideoTensor)
        else:
            break

    if VideoTensor.shape[0] != 0:  # Check if VideoTensor is not empty
        VideoTensor = tf.reshape(VideoTensor, (1, VideoTensor.shape[0], VideoTensor.shape[1], VideoTensor.shape[2]))
        Data = tf.concat([Data, VideoTensor], axis=0)
        print(Data)
    cap.release() 
    cv2.destroyAllWindows() """

'''
def addLandmarks(coordinates, tensor):  # Function for adding new landmarks to tensor
    if type(coordinates[0]) == mpLandmark.Landmark:
        for landmark in coordinates:
            value = tf.constant([landmark.x, landmark.y, landmark.z], dtype=tf.float16)
            tensor = tf.concat([tensor, tf.expand_dims(value, 0)], axis=0)
    else:
        for filler in coordinates:
            tensor = tf.concat([tensor, tf.expand_dims(filler, 0)], axis=0)
    return tensor  # Return 2D tf.Tensor


def generateFrameLandmarks(frame):  # Function for generating landmarks for single frame
    frame = mp.Image.create_from_file(frame)
    poseResult = poseLandmarker.detect(image=frame)
    poseCoordinates = poseResult.pose_world_landmarks
    if len(poseCoordinates) == 0:
        return None
    handResult = handLandmarker.detect(image=frame)
    handedness = handResult.handedness
    handCoordinates = handResult.hand_world_landmarks
    coordinatesTensor = tf.zeros([1, 3], dtype=tf.float16)
    coordinatesTensor = addLandmarks(poseCoordinates[0][:25], coordinatesTensor)
    if type(poseCoordinates[0]) == mpLandmark.Landmark:
        for landmark in poseCoordinates:
            value = tf.constant([landmark.x, landmark.y, landmark.z], dtype=tf.float16)
            tensor = tf.concat([tensor, tf.expand_dims(value, 0)], axis=0)
        coordinatesTensor = tensor
    else:
        for filler in poseCoordinates:
            tensor = tf.concat([tensor, tf.expand_dims(filler, 0)], axis=0)
        coordinatesTensor = tensor
    if len(handedness) == 0:  # check if no hand is detected
        for i in range(2):
            filler = tf.zeros([len(handColumnNameList), 3], dtype=tf.float16)
            coordinatesTensor = addLandmarks(filler, coordinatesTensor)
    else:  # execute if hand is detected
        for index, category in enumerate(handedness):
            if len(handCoordinates) == 1:  # check if mp detect only one hand
                filler = tf.zeros([len(handColumnNameList), 3], dtype=tf.float16)
                if category.index == 0:  # detect right
                    coordinatesTensor = addLandmarks(handCoordinates[index], coordinatesTensor)
                    coordinatesTensor = addLandmarks(filler, coordinatesTensor)
                else:  # detect left
                    coordinatesTensor = addLandmarks(filler, coordinatesTensor)
                    coordinatesTensor = addLandmarks(handCoordinates[index], coordinatesTensor)
                break
            else:
                coordinatesTensor = addLandmarks(handCoordinates[index], coordinatesTensor)
    coordinatesTensor = tf.slice(coordinatesTensor, [1, 0], [coordinatesTensor.shape[0] - 1, 3])  # remove the first element that got created when declaring the empty array

    return coordinatesTensor  # return 2D tf.Tensor


def removeExcessFrames(frameList, sample):  # Convert from frames to landmarks of video
    landmarksList = []
    middlePosition = len(frameList) // 2  # Middle position of input array
    currentPosition = [middlePosition, middlePosition]  # Use for calculating which is the next position needed to be added to the array.
    addToBack = True

    while len(landmarksList) < frameBuffer:
        direction = int(addToBack)  # 0 = front, 1 = back
        positionToAdd = direction * len(landmarksList)  # Position to add element to the array that got returned
        valuePosition = currentPosition[direction] + (sample * (direction * 2 - 1))  # Position in the frameList that will be added to return array
        while True:
            try:
                if valuePosition >= 0 and valuePosition < len(frameList):  # Preventing from adding the value that have been counted from the back of frameList.
                    landmark = generateFrameLandmarks(str(frameList[valuePosition]))
                    if landmark is not None:
                        break
                raise IndexError
            except IndexError:  # Only execute when the value position is below zero
                valuePosition -= (direction * 2 - 1)
        landmarksList.insert(positionToAdd, landmark)
        currentPosition[direction] = valuePosition
        addToBack = not addToBack

    # return array
    return tf.stack(landmarksList)  # Return "3D" tf.Tensor


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
    # capture all frames the video has
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # np array = impossible to detect hand, image file = really easy to detect
            file = tempDirectory.joinpath(f"{videoPATH.name}_{currentFrame}.png")
            height = frame.shape[0] // resizeRatio
            width = frame.shape[1] // resizeRatio
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

    return landmarks  # Return "3D" tf.Tensor


startTime = time.perf_counter()

videoPaths = [file for extension in supportsExtension for file in inputDirectory.glob(extension)]
totalFile = len(videoPaths)
for index, video in enumerate(videoPaths):
    label = labelList[video.parent.name]

    landmarks = videoToLandmarks(video, baseSample)
    if not isinstance(landmarks, tf.Tensor):
        print(f"Skip {video.name}")
        continue
    landmarks = tf.reshape(landmarks, [-1])
    landmarks = landmarks.numpy().tolist()
    landmarks.insert(0, label)

    df.loc[len(df)] = landmarks
    print(f"Progress: {index + 1} / {totalFile}")

dataProcessTime = time.perf_counter()

df = df.sample(frac=1)  # Shuffle dataframe
shuffleTime = time.perf_counter()
print(df)
df.to_csv(outputFile, index=False)
finishTime = time.perf_counter()
print(f"Data process time: {dataProcessTime - startTime} seconds")
print(f"Shuffle time: {shuffleTime - dataProcessTime} seconds")
print(f"Save time: {finishTime - shuffleTime} seconds")
'''