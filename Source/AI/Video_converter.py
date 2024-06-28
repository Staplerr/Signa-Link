import mediapipe as mp
import numpy as np
import cv2
from pathlib import Path
from google.protobuf.json_format import MessageToDict
import json

# Directory
parent_directory = Path(__file__).parent
input_directory = parent_directory / "Videos"

# OpenCV config
resize_ratio = (1280, 720)
resize_interpolation = cv2.INTER_AREA

# Supported file extensions
supported_extensions = ["*.mp4", "*.mov"]

# Data structures
label_list = {}
video_paths = []

# Populate video paths and labels
for index, directory in enumerate(input_directory.iterdir()):
    if directory.is_dir():
        label_list[directory.name] = index
        label_list[index] = directory.name
        for extension in supported_extensions:
            video_paths.extend(directory.glob(extension))

# Process videos
def process_videos():
    total_files = len(video_paths)
    data = np.empty((0, 10, 42, 3), dtype=np.float32)
    labels = np.empty((0,), dtype=np.float32)
    
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.4, min_tracking_confidence=0.4) as hands:
        for counter, video in enumerate(video_paths, start=1):
            cap = cv2.VideoCapture(str(video))
            frame_array = np.empty((0, 42, 3), dtype=np.float32)
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, resize_ratio, interpolation=resize_interpolation)
                results = hands.process(image)
                
                coordinates = np.zeros((42, 3), dtype=np.float32)
                if results.multi_hand_landmarks:
                    handedness = [MessageToDict(hand) for hand in results.multi_handedness]
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        start = 0 if handedness[idx]['classification'][0]['index'] == 0 else 21
                        for i, landmark in enumerate(hand_landmarks.landmark):
                            coordinates[start + i] = [landmark.x, landmark.y, landmark.z]

                if np.any(coordinates):
                    frame_array = np.concatenate((frame_array, [coordinates]), axis=0)

                if frame_array.shape[0] == 10:
                    data = np.concatenate((data, [frame_array]), axis=0)
                    labels = np.concatenate((labels, [label_list[directory.name]]), axis=0)
                    frame_array = np.empty((0, 42, 3), dtype=np.float32)

            cap.release()
            print(f"Progress: {counter}/{total_files}, Collected data: {data.shape[0]}")

        print(f"All data collected: {data.shape[0]}")
    return data, labels

if __name__ == "__main__":
    data, labels = process_videos()

    # Save data
    output_dir = parent_directory / "Data"
    output_dir.mkdir(exist_ok=True)

    np.save(output_dir / "Features.npy", data)
    np.save(output_dir / "Labels.npy", labels)
    with open(output_dir / "labels.json", 'w') as f:
        json.dump(label_list, f)
