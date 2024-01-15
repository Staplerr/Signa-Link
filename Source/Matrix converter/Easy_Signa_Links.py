import mediapipe as mp
from mediapipe.tasks.python import vision
import cv2
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from pathlib import Path
import os

parentPath = Path(__file__).parent
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
poseOption = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=poseModel),
                                   running_mode=VisionRunningMode.IMAGE,
                                   min_pose_detection_confidence=minPoseConfidence)
handOption = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=handModel),
                                   running_mode=VisionRunningMode.IMAGE,
                                   min_hand_detection_confidence=minHandConfidence,
                                   num_hands=2)
poseLandmarker = PoseLandmarker.create_from_options(poseOption)
HandLandmarker = HandLandmarker.create_from_options(handOption)

handColumnNameList = ["wrist", "thumb cmc", "thumb mcp", "thumb ip", "thumb tip",
                      "index finger mcp", "index finger pip", "index finger dip", "index finger tip", "middle finger mcp",
                      "middle finger pip", "middle finger dip", "middle finger tip", "ring finger mcp", "ring finger pip",
                      "ring finger dip", "ring finger tip", "pinky mcp", "pinky pip", "pinky dip",
                      "pinky tip"]

hand= []
right_hand = []
for word in handColumnNameList:
    hand.append(word + " Right Hand")
    right_hand.append(word + " Right Hand")
for word in handColumnNameList:
    hand.append(word + " Left Hand")


def get_Coordinates(Coordinates):
    xyz_list = []
        
    for landmark in Coordinates[0]:
        xyz_list.append([landmark.x, landmark.y, landmark.z])
    
    try:
        for landmark in Coordinates[1]:
            xyz_list.append([landmark.x, landmark.y, landmark.z])
    except IndexError:
        pass
    
    return xyz_list

def get_HandCoordinates(path):
    parentPath = Path(__file__).parent
    inputDirectory = parentPath.joinpath("Input")
    imagePATH = inputDirectory.joinpath(path)
    image = mp.Image.create_from_file(str(imagePATH))
    poseResult = poseLandmarker.detect(image)
    poseCoordinates = poseResult.pose_world_landmarks[0][:25]
    handResult = HandLandmarker.detect(image)
    handCoordinates = handResult.hand_world_landmarks
    
    return handCoordinates

class PoomjaiIsNoob:
    def __init__(self, df):
        self.df = df
        
    def __repr__(self):
        return self.df
    
    def add_to_df(self, data, index_name = None):
        list = []
        list.append(data)
        df_to_add = pd.DataFrame(list)
        try:
            if index_name is not None:
                df_to_add.index = index_name
            else:
                pass
            try:
                df_to_add.columns = hand
            except ValueError:
                df_to_add.columns = right_hand
        except TypeError:
            print("\n\ntype Error!\nTry to add Square Bracket\n\n")
            quit()
            
        self.df = pd.concat([self.df, df_to_add], ignore_index = False)
        self.df = self.df.fillna(0)
        
        return self.df
    
    def add(self, Coordinates, index_name = None):
        xyz_list = get_Coordinates(Coordinates)
        list = []
        list.append(xyz_list)
        df_to_add = pd.DataFrame(list)
        try:
            if index_name is not None:
                df_to_add.index = index_name
            else:
                pass
            try:
                df_to_add.columns = hand
            except ValueError:
                df_to_add.columns = right_hand
                print("\n\n   No Left Hand.\n\n")
        except TypeError:
            print("\n\ntype Error!\nTry to add Square Bracket\n\n")
            quit()
            
        self.df = pd.concat([self.df, df_to_add], ignore_index = False)
        self.df = self.df.fillna(0)
        
        return self.df