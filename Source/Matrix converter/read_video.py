import cv2
from pathlib import Path

parentDirectory = Path(__file__).parent
inputDirectory = parentDirectory.joinpath("Input")
outputDirectory = parentDirectory.joinpath("Output")
supportsExtension = ["*/*.mp4", "*/*.mov"]
videoPATHs = []

def save_frames(video_path, label):
    try:    
        cap = cv2.VideoCapture(str(video_path))
        success, frame = cap.read()
        frame_number = 0
        labelDirectory = outputDirectory.joinpath(label)
        labelDirectory.mkdir(parents=True, exist_ok=True) #Create directory if not exist

        while success:
            image_path = str(labelDirectory.joinpath(video_path.name.split(".")[0] + "_" + str(frame_number) + ".png"))
            cv2.imwrite(image_path, frame)
            success, frame = cap.read()
            frame_number += 1
        cap.release()
        print("Processed: " + str(video_path))
    except:
        print("Error occur on: " + video_path.name)

for extension in supportsExtension: #Collect file that has mp4 and mov file extension
    for file in inputDirectory.glob(extension):
        videoPATHs.append(file)
for videoPATH in videoPATHs:
    save_frames(videoPATH, videoPATH.parent.name)
