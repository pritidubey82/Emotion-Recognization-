import cv2
import numpy as np
from keras.models import model_from_json

# Load the pre-trained emotion detection model
with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("facialemotionmodel.weights.h5")  # Ensure correct weights filename

# Load Haar cascade classifier for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define emotion labels
labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

def preprocess_image(image):
    """Preprocess image for model prediction"""
    feature = np.array(image, dtype="float32")
    feature = feature.reshape(1, 48, 48, 1)  # Reshape for model input
    return feature / 255.0  # Normalize pixel values

# Open webcam
webcam = cv2.VideoCapture(0)

while True:
    success, frame = webcam.read()
    if not success:
        print("Failed to capture image")
        break

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = gray_frame[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (48, 48))

        # Process face ROI for prediction
        img = preprocess_image(face_roi)
        pred = model.predict(img)
        predicted_label = labels[np.argmax(pred)]  # Get label with highest probability

        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display emotion label
        cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show the output frame
    cv2.imshow("Facial Emotion Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
