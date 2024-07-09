import mediapipe as mp
import numpy as np
import cv2
from pathlib import Path
from google.protobuf.json_format import MessageToDict
import json
from concurrent.futures import ThreadPoolExecutor

# Directory
parent_directory = Path(__file__).parent
input_directory = parent_directory.parent / "Videos"

# OpenCV config
resize_ratio = (1280, 720) # 720p
#resize_ratio = (640, 360) # 360p
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

# Process a single video
def process_video(video, label_list, debug=False):
    mp_hands = mp.solutions.hands
    data = np.empty((0, 10, 84, 3), dtype=np.float32)
    labels = np.empty((0,), dtype=np.float32)
    
    with mp_hands.Hands(model_complexity=1, min_detection_confidence=0.4, min_tracking_confidence=0.4) as hands:
        cap = cv2.VideoCapture(str(video))
        frame_array = np.empty((0, 84, 3), dtype=np.float32)
        last_coordinates = np.zeros((42, 3), dtype=np.float32)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, resize_ratio, interpolation=resize_interpolation)
            results = hands.process(image)

            coordinates = np.zeros((84, 3), dtype=np.float32)
            if results.multi_hand_landmarks:
                handedness = [MessageToDict(hand) for hand in results.multi_handedness]
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    start = 0 if handedness[idx]['classification'][0]['index'] == 0 else 21
                    if debug:
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        mp_drawing = mp.solutions.drawing_utils
                        mp_drawing_styles = mp.solutions.drawing_styles
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                            )
                    for i, landmark in enumerate(hand_landmarks.landmark):    
                        coordinates[start + i] = [landmark.x, landmark.y, landmark.z]
                        coordinates[start + 42 + i] = [
                            coordinates[start + i][0] - last_coordinates[start + i][0],
                            coordinates[start + i][1] - last_coordinates[start + i][1],
                            coordinates[start + i][2] - last_coordinates[start + i][2]
                        ]
                last_coordinates = coordinates[:42]
            else:
                last_coordinates = np.zeros((42, 3), dtype=np.float32)
            if debug:
                cv2.imshow('Debug', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

            if np.any(coordinates):
                frame_array = np.concatenate((frame_array, [coordinates]), axis=0)

            if frame_array.shape[0] == 10:
                data = np.concatenate((data, [frame_array]), axis=0)
                labels = np.concatenate((labels, [label_list[str(Path(video).parent.name)]]), axis=0)
                frame_array = np.empty((0, 84, 3), dtype=np.float32)
        print(f"Processed: {Path(video).parent.name} with {data.shape[0]} data")
        cap.release()
        if debug:
            cv2.destroyAllWindows()

    return data, labels

# Process videos
def process_videos(debug=False):
    all_data = []
    all_labels = []
    
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda video: process_video(video, label_list, debug), video_paths)

    for data, labels in results:
        all_data.append(data)
        all_labels.append(labels)

    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"All data collected: {all_data.shape[0]}")

    return all_data, all_labels

if __name__ == "__main__":
    data, labels = process_videos(debug=False)

    # Save data
    output_dir = parent_directory / "Data"
    output_dir.mkdir(exist_ok=True)

    np.save(output_dir / "Features.npy", data)
    np.save(output_dir / "Labels.npy", labels)
    with open(output_dir / "labels.json", 'w') as f:
        json.dump(label_list, f)
