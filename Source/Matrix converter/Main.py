import mediapipe as mp
from mediapipe.tasks.python import vision
import cv2
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from pathlib import Path
import os

parentPath = Path(os.getcwd())
print("Parent directory: " + str(parentPath))
inputDirectory = parentPath.joinpath("Input")
outputFile = parentPath.joinpath("output" + ".csv")

modelDirectory = parentPath.joinpath("Model")
handModel = modelDirectory.joinpath("hand_landmarker.task")
minPoseConfidence = 0.5
poseModel = modelDirectory.joinpath("pose_landmarker_full.task")
minHandConfidence = 0.5
#Config
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

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

#detecting landmarks
imagePATH = inputDirectory.joinpath("หมู/Hand.jpg")
image = mp.Image.create_from_file(str(imagePATH))
poseResult = poseLandmarker.detect(image)
poseCoordinates = poseResult.pose_world_landmarks[0][:25]
handResult = HandLandmarker.detect(image)
handCoordinates = handResult.hand_world_landmarks
print(handResult.handedness)
if len(handResult.handedness) > 1:
    print(handCoordinates[0])
    print(handCoordinates[1])

#displaying mask
#from mediapipe.framework.formats import landmark_pb2
#from mediapipe import solutions
#def draw_landmarks_on_image(rgb_image, detection_result):
#  pose_landmarks_list = detection_result.pose_landmarks
#  annotated_image = np.copy(rgb_image)
#
#  # Loop through the detected poses to visualize.
#  for idx in range(len(pose_landmarks_list)):
#    pose_landmarks = pose_landmarks_list[idx]
#
#    # Draw the pose landmarks.
#    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#    pose_landmarks_proto.landmark.extend([
#      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
#    ])
#    solutions.drawing_utils.draw_landmarks(
#      annotated_image,
#      pose_landmarks_proto,
#      solutions.pose.POSE_CONNECTIONS,
#      solutions.drawing_styles.get_default_pose_landmarks_style())
#  return annotated_image
#annotated_image = draw_landmarks_on_image(image.numpy_view(), result)
#cv2.imshow("window", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
#cv2.waitKey()