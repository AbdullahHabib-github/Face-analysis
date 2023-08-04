
import cv2
import dlib
import numpy as np


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


def main():
    # Load the pre-trained facial landmark detection model
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = detector(gray)

        for face in faces:
            # Get facial landmarks for the detected face
            landmarks = predictor(gray, face)

            # Convert dlib shape object to numpy array for easier manipulation
            landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])
            if(is_mouth_open(landmarks_np)):
                print("mouth open")
            else:
                print("not open")    
            # Extract the mouth region using the detected facial landmarks
            mouth_points = landmarks_np[48:68]
            mouth_area = cv2.convexHull(mouth_points)

            # Draw a rectangle around the mouth area
            x, y, w, h = cv2.boundingRect(mouth_area)



            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Mouth Area Detection", frame)

        # Exit when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
