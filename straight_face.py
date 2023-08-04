import cv2

def is_user_facing_camera(image_path):
    # Load pre-trained face detection model from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


    webcam = cv2.VideoCapture(0)

    while True:
        # Capture each frame from the webcam
        ret, frame = webcam.read()

        if not ret:
            break
    # Read the input image
        # image = cv2.imread(image_path)
        
        # Convert the image to grayscale for face detection
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            print("User is facing the camera.")
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            print("User is not facing the camera.")

        # Display the image with face detection (you can remove this for batch processing)
        cv2.imshow('Face Detection', frame)
         # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"  # Replace with the path to your image
    is_user_facing_camera(image_path)
