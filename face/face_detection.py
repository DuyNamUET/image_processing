import numpy as np
import cv2

if __name__ == "__main__":
    casc_path = "/opt/ros/kinetic/share/OpenCV-3.3.1-dev/haarcascades/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade.load(casc_path)
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face = face_cascade.detectMultiScale(gray, 1.1, 5)

        # draw around the face
        for (x, y, w, h) in face:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # show on screen
        cv2.imshow('', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindow()