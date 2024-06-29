import mediapipe as mp
import cv2
from pathlib import Path
import tensorflow as tf
import numpy as np
from google.protobuf.json_format import MessageToDict
import json
from keras.models import load_model


model = load_model(f"{Path(__file__).parent}/Matrix model/best_model.keras")

# Directory
parentDirectory = Path(__file__).parent
inputDirectory = parentDirectory.joinpath("Videos")

# Opencv config
resizeRatio = [1280, 720]
resizeInterpolation = cv2.INTER_AREA


# Dataframe
f = open(f"{str(Path(__file__).parent)}/Data/label.json",)
labelList = json.load(f)

if __name__ == "__main__":  
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4) as hands:
            cap = cv2.VideoCapture(0)
            print("starting camera")
            frame_array = np.empty((0,42,3), dtype=np.float32)
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                  break
              
                image.flags.writeable = True # To improve performance, optionally mark the image as not writeable to
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (resizeRatio[0], resizeRatio[1]), interpolation=resizeInterpolation)
                results = hands.process(image)
                coordinates = np.zeros((42, 3), dtype=np.float32)
                if results.multi_hand_landmarks:
                    handedness = [MessageToDict(hand) for hand in results.multi_handedness]
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS,
                                                  mp_drawing_styles.get_default_hand_landmarks_style(),
                                                  mp_drawing_styles.get_default_hand_connections_style())
                        start = 0 if handedness[idx]['classification'][0]['index'] == 0 else 21
                        for i, landmark in enumerate(hand_landmarks.landmark):
                            coordinates[start + i] = [landmark.x, landmark.y, landmark.z]

                cv2.imshow('Test', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                if np.any(coordinates):
                    frame_array = np.concatenate((frame_array, [coordinates]), axis=0)

                if frame_array.shape[0] == 10:
                    result = model.predict(frame_array.reshape((1, 10, 42, 3)))
                    print(f"Model results: {labelList[str(result.tolist()[0].index(max(result.tolist()[0])))]}")
                    frame_array = np.empty((0, 42, 3), dtype=np.float32)
               
            
    
            cap.release()
            cv2.destroyAllWindows()