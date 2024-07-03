import mediapipe as mp
import cv2
from pathlib import Path
import tensorflow as tf
import numpy as np
from google.protobuf.json_format import MessageToDict
import json
from keras.models import load_model

# Directory
parent_directory = Path(__file__).parent

# OpenCV config
resize_ratio = (1280, 720)
resize_interpolation = cv2.INTER_AREA

# Data structures
with open(parent_directory / "Data/labels.json", 'r') as f:
    label_dict = json.load(f)
model = load_model(f"{Path(__file__).parent}/Matrix model/best_model.keras")
# Process videos
mp_holistic = mp.solutions.holistic
with mp_holistic.Holistic(
        model_complexity=2,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4) as holistic:
        cap = cv2.VideoCapture(0)
        frame_array = np.empty((0, 543, 3), dtype=np.float32)
        frame_count = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            frame_count += 1
            if frame_count % 2 == 0:
                continue
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, resize_ratio, interpolation=resize_interpolation)
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp.solutions.drawing_utils.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
            mp.solutions.drawing_utils.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
            cv2.imshow('Test', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            coordinates = np.zeros((543, 3), dtype=np.float32)
            if results.pose_landmarks:
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    coordinates[i] = [landmark.x, landmark.y, landmark.z]
            if results.face_landmarks:
                for i, landmark in enumerate(results.face_landmarks.landmark):
                    coordinates[33 + i] = [landmark.x, landmark.y, landmark.z]
            if results.left_hand_landmarks:
                for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                    coordinates[33 + 468 + i] = [landmark.x, landmark.y, landmark.z]
            if results.right_hand_landmarks:
                for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                    coordinates[33 + 468 + 21 + i] = [landmark.x, landmark.y, landmark.z]
            if np.any(coordinates):
                frame_array = np.concatenate((frame_array, [coordinates]), axis=0)
            if frame_array.shape[0] == 10:
                result = model.predict(frame_array.reshape((1, 10, 543, 3)))
                print(f"Model results: {label_dict[str(result.tolist()[0].index(max(result.tolist()[0])))]}")
                frame_array = np.empty((0, 543, 3), dtype=np.float32)
        cap.release()
        cv2.destroyAllWindows()