import json
import numpy as np
import cv2
from pathlib import Path
from google.protobuf.json_format import MessageToDict
from keras.models import load_model # type: ignore
from concurrent.futures import ThreadPoolExecutor
import mediapipe as mp

# Directory
parent_directory = Path(__file__).parent
input_directory = parent_directory.joinpath("Videos")

# Load the model
model = load_model(f"{parent_directory}/Matrix model/Complex_best_model.keras")
# model = load_model(f"{parent_directory}/Matrix model/Base_best_model.keras")

# Opencv config
resize_ratio = (256, 144)  # 144p
#resize_ratio = (640, 360)  # 360p
#resize_ratio = (1280, 720) # 720p
resize_interpolation = cv2.INTER_AREA
predicted_threshold = 3

# Load labels
with open(f"{parent_directory}/Data/labels.json", 'r') as f:
    label_list = json.load(f)

def process_frame(hands, frame, last_coordinates):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, resize_ratio, interpolation=resize_interpolation)
    results = hands.process(image)
    coordinates = np.zeros((84, 3), dtype=np.float32)

    if results.multi_hand_landmarks:
        handedness = [MessageToDict(hand) for hand in results.multi_handedness]
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            start = 0 if handedness[idx]['classification'][0]['index'] == 0 else 21
            for i, landmark in enumerate(hand_landmarks.landmark):
                coordinates[start + i] = [landmark.x, landmark.y, landmark.z]
                coordinates[start + 42 + i] = [
                    coordinates[start + i][0] - last_coordinates[start + i][0],
                    coordinates[start + i][1] - last_coordinates[start + i][1],
                    coordinates[start + i][2] - last_coordinates[start + i][2]
                ]

    return coordinates, results

def draw_landmarks(frame, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

if __name__ == "__main__":
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4) as hands:

        cap = cv2.VideoCapture(0)
        print("Starting camera")
        prev_predicted = []
        prev_confident = []
        frame_array = np.empty((0, 84, 3), dtype=np.float32)
        last_coordinates = np.zeros((42, 3), dtype=np.float32)

        with ThreadPoolExecutor() as executor:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                future = executor.submit(process_frame, hands, frame, last_coordinates)
                coordinates, results = future.result()
                draw_landmarks(frame, results)

                cv2.imshow('Test', frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

                if np.any(coordinates):
                    frame_array = np.concatenate((frame_array, [coordinates]), axis=0)
                    last_coordinates = coordinates[:42]

                if frame_array.shape[0] == 10:
                    result = model.predict(frame_array.reshape((1, 10, 84, 3)))
                    predicted_label_index = np.argmax(result, axis=1)[0]
                    confidence = float(result[0][predicted_label_index]) * 100
                    predicted_label = label_list[str(predicted_label_index)]
                    prev_predicted.append(predicted_label)
                    prev_confident.append(confidence)

                    if len(prev_predicted) > predicted_threshold:
                        prev_predicted.pop(0)
                        prev_confident.pop(0)

                    if prev_predicted.count(predicted_label) == predicted_threshold:
                        average_confidence = np.mean(prev_confident)
                        print(f"Model results: {predicted_label} {average_confidence:.2f}%")
                        prev_predicted = []
                        prev_confident = []

                    frame_array = np.empty((0, 84, 3), dtype=np.float32)

        cap.release()
        cv2.destroyAllWindows()
