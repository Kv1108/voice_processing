# made CCTV.py for synchronized audio and video processing

import cv2
import time
import datetime
import os
from utils import ensure_folder_exists

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

video_folder = "recorded_videos"
ensure_folder_exists(video_folder)

frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

while True:
    _, frame = cap.read()


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) + len(bodies) > 0:  
        if detection:
            timer_started = False
        else:
            detection = True
            timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            video_path = os.path.join(video_folder, f"{timestamp}.mp4")
            out = cv2.VideoWriter(video_path, fourcc, 20, frame_size)
            print(f"Started Recording: {video_path}")
    elif detection:  
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                print("Stopped Recording!")
        else:
            timer_started = True
            detection_stopped_time = time.time()

    if detection:
        out.write(frame)

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)
    for (x, y, width, height) in bodies:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord("q"):
        break

if "out" in locals():
    out.release()
cap.release()
cv2.destroyAllWindows()
