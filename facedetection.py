import cv2
from deepface import DeepFace

def detect_emotions():
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize video capture
    video_stream = cv2.VideoCapture(0)

    try:
        while video_stream.isOpened():
            # Read a frame from the video stream
            success, frame = video_stream.read()
            if not success:
                print("Failed to capture frame. Exiting...")
                break

            # Process the frame for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                # Extract the face region of interest (ROI)
                face_region = frame[y:y + h, x:x + w]

                try:
                    # Analyze the face for emotions using DeepFace
                    analysis = DeepFace.analyze(
                        img_path=face_region, actions=['emotion'], enforce_detection=False
                    )
                    dominant_emotion = analysis['dominant_emotion']

                    # Annotate the frame with emotion and bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(
                        frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
                    )
                except Exception as e:
                    print(f"Error analyzing face: {e}")

            # Show the annotated frame
            cv2.imshow('Emotion Detection', frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release resources
        video_stream.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotions()