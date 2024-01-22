import cv2

def save_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    success, frame = cap.read()
    frame_number = 0

    while success:
        image_path = fr'C:\Users\max\OneDrive\Desktop\Signa-Link-1\Source\Getdata\Test\Image\frame_{frame_number:04d}.png' #ไปเปลี่ยนเอาเองนะฮาฟ
        cv2.imwrite(image_path, frame)

        success, frame = cap.read()
        frame_number += 1

        if frame_number % 100 == 0:
            print(f'Processed {frame_number} frames')
            
    print(f'\n\nProcessed {frame_number} frames')
    cap.release()

save_frames(r'C:\Users\max\OneDrive\Desktop\Signa-Link-1\Source\Getdata\Test\Video\test.mp4') #ไปเปลี่ยนเอาเองนะฮาฟ
