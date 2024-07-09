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
def process_videos(frame_skip=1, debug=False):
    total_files = len(video_paths)
    data = np.empty((0, 10, 543, 3), dtype=np.float32)
    labels = np.empty((0,), dtype=np.float32)
    
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
            model_complexity=0,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4) as holistic:
        for counter, video in enumerate(video_paths, start=1):
            cap = cv2.VideoCapture(str(video))
            frame_array = np.empty((0, 543, 3), dtype=np.float32)
            frame_count = 0
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, resize_ratio, interpolation=resize_interpolation)
                results = holistic.process(image)

                # Draw landmarks on the frame for debugging
                if debug:
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
                    
                    cv2.imshow('Debug', image)
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
                    data = np.concatenate((data, [frame_array]), axis=0)
                    labels = np.concatenate((labels, [label_list[str(Path(video).parent.name)]]), axis=0)
                    frame_array = np.empty((0, 543, 3), dtype=np.float32)

            cap.release()
            cv2.destroyAllWindows()
            print(f"Progress: {counter}/{total_files}, Collected data: {data.shape[0]}")
            print(f"label : {labels[-1]}, {labels.shape}")
            print(str(Path(video).parent.name))

    return data, labels

if __name__ == "__main__":
    data, labels = process_videos(debug=False)
    print(f"All data collected: {data.shape[0]}")

    # Save data
    output_dir = parent_directory / "Data"
    output_dir.mkdir(exist_ok=True)

    np.save(output_dir / "Features.npy", data)
    np.save(output_dir / "Labels.npy", labels)
    with open(output_dir / "labels.json", 'w') as f:
        json.dump(label_list, f)
