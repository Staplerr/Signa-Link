from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import io
from PIL import Image
from keras.models import load_model  # type: ignore
import mediapipe as mp
import pathlib
import json
from google.protobuf.json_format import MessageToDict
import threading
import asyncio
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

parent_directory = pathlib.Path(__file__).parent
model = load_model(parent_directory.joinpath("Base_final_model.keras"))

resize_ratio = (256, 144)  # 144p
resize_interpolation = cv2.INTER_AREA
predicted_threshold = 2

with open(parent_directory.joinpath("labels.json"), 'r') as f:
    label_list = json.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# init global var
last_coordinates = np.zeros((42, 3), dtype=np.float32)
frame_array = np.empty((0, 84, 3), dtype=np.float32)
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.4, min_tracking_confidence=0.4)

prev_predicted = []
prev_confident = []
last_prediction_time = 0
lock = threading.Lock()  # Add a lock for thread safety

# variables that storing the last valid prediction and confidence
last_valid_predicted_label = "Unknown"
last_valid_confidence = 0.0

async def process_frame(hands, frame, last_coordinates):
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
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    return coordinates, results

async def predict(frame, hands, last_coordinates, frame_array):
    coordinates, results = await process_frame(hands, frame, last_coordinates)
    if np.any(coordinates):
        frame_array = np.concatenate((frame_array, [coordinates]), axis=0)
        last_coordinates = coordinates[:42]

    if frame_array.shape[0] >= 10:  # Use >= to handle any case where it exceeds 10
        result = model.predict(frame_array.reshape((1, 10, 84, 3)))
        predicted_label_index = np.argmax(result, axis=1)[0]
        confidence = float(result[0][predicted_label_index]) * 100
        predicted_label = label_list.get(str(predicted_label_index), "Unknown")

        last_predicted_label = predicted_label
        prev_predicted.append(predicted_label)
        prev_confident.append(confidence)

        if len(prev_predicted) > predicted_threshold:
            prev_predicted.pop(0)
            prev_confident.pop(0)

        if prev_predicted.count(predicted_label) >= predicted_threshold:
            average_confidence = np.mean(prev_confident)
            prev_predicted.clear()
            prev_confident.clear()
            return last_predicted_label, predicted_label, average_confidence, last_coordinates, np.empty((0, 84, 3), dtype=np.float32)
        else:
            return last_predicted_label, None, None, last_coordinates, np.empty((0, 84, 3), dtype=np.float32)
    
    return None, None, None, last_coordinates, frame_array

@socketio.on('image')
def handle_image(data):
    img_data = base64.b64decode(data)
    img = Image.open(io.BytesIO(img_data))
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    global last_coordinates, frame_array, hands, last_prediction_time, last_valid_predicted_label, last_valid_confidence
    try:
        with lock:  # Ensure thread safety
            last_predicted_label, predicted_label, confidence, last_coordinates, frame_array = asyncio.run(predict(frame, hands, last_coordinates, frame_array))
            if predicted_label:
                last_prediction_time = time.time()
                last_valid_predicted_label = predicted_label
                last_valid_confidence = confidence
            elif time.time() - last_prediction_time > 5:
                last_valid_predicted_label = "Unknown"
                last_valid_confidence = 0.0
                
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    if predicted_label:
        print(f"{predicted_label} | {confidence:.2f}%")

    _, buffer = cv2.imencode('.jpg', frame)
    encoded_frame = base64.b64encode(buffer).decode('utf-8')
    confidence = last_valid_confidence if last_valid_confidence is not None else 0.0
    last_predicted_label = last_valid_predicted_label if last_valid_predicted_label is not None else "Unknown"
    response_data = {
        'image': encoded_frame,
        'prediction': f'{last_predicted_label} ({confidence:.2f}%)'
    }
    emit('response', response_data)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True, port=8001)
