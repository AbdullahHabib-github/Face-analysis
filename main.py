import cv2
from gaze_tracking import GazeTracking
import numpy as np
import dlib


def is_mouth_open(landmarks):
    # Calculate the distance between upper and lower lip points
    upper_lip_pts = np.array(landmarks[50:53] + landmarks[61:64])
    lower_lip_pts = np.array(landmarks[65:68] + landmarks[56:59])
    upper_lip_mean = np.mean(upper_lip_pts[:, 1])
    lower_lip_mean = np.mean(lower_lip_pts[:, 1])
    distance = abs(upper_lip_mean - lower_lip_mean)

    # Define a threshold to determine if the mouth is open
    threshold = 30

    return distance > threshold

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

while True:

    score = 0
    # We get a new frame from the webcam
    _, frame = webcam.read()

    if not _:
        break


    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)


    if gaze.is_center():
        score+=1
        text = "Looking center"

    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    chehra = detector(gray_frame)

    # For each face detected, try to detect smiles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Get the region of interest (ROI) for smile detection
        roi_gray = gray_frame[y:y+h, x:x+w]
    
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))


    if len(smiles)!=0: ###Might add idents
        score+=1    


 
    landmarks = predictor(gray_frame, chehra[0])

    # Convert dlib shape object to numpy array for easier manipulation
    landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])
    if not is_mouth_open(landmarks_np):
        score+=1


    # Draw rectangles around the detected smiles

    if len(faces) > 0: ###User is facing the camera.
        score+=1

    print(score)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()