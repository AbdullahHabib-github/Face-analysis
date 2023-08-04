# import cv2

# def detect_faces_webcam():
#     # Load pre-trained face detection model from OpenCV
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     # Initialize the webcam
#     webcam = cv2.VideoCapture(0)

#     while True:
#         # Capture each frame from the webcam
#         ret, frame = webcam.read()

#         if not ret:
#             break

#         # Convert the frame to grayscale for face detection
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Detect faces in the frame
#         faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#         # Draw rectangles around the detected faces
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

#         # Display the frame with detected faces
#         cv2.imshow('Face Detection', frame)

#         # Exit the loop when 'q' key is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the webcam and close the window
#     webcam.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     detect_faces_webcam()






import cv2

def detect_faces_webcam():
    # Load pre-trained face and smile detection models from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    # Initialize the webcam
    webcam = cv2.VideoCapture(0)

    while True:
        # Capture each frame from the webcam
        ret, frame = webcam.read()

        if not ret:
            break

        # Convert the frame to grayscale for face and smile detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # For each face detected, try to detect smiles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Get the region of interest (ROI) for smile detection
            roi_gray = gray_frame[y:y+h, x:x+w]
        
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))

            # Draw rectangles around the detected smiles
            if len(smiles)==0:
                print("no smile")
            else:
                print("smile")    
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(frame, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (0, 255, 0), 2)

        # Display the frame with detected faces and smiles
        cv2.imshow('Face and Smile Detection', frame)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces_webcam()
