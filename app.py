import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('road_sign_model_updated.h5')
class_names = ['PARKING', 'PED XING', 'STOP']  # Ensure this matches the class indices

def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Perform edge detection
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def extract_contours(edges, frame):
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Select the largest contour with area threshold
    largest_contour = max(contours, key=cv2.contourArea, default=None)
    if largest_contour is not None and cv2.contourArea(largest_contour) > 1500:  # Adjust threshold as needed
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped = frame[y:y+h, x:x+w]
        return cropped
    return None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    edges = preprocess_frame(frame)
    sign_region = extract_contours(edges, frame)

    if sign_region is not None:
        # Resize and normalize for model
        resized = cv2.resize(sign_region, (64, 64))
        normalized = resized / 255.0
        reshaped = np.expand_dims(normalized, axis=0)

        # Predict
        predictions = model.predict(reshaped)
        class_index = np.argmax(predictions)
        label = class_names[class_index]

        # Display label
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the processed frame
    cv2.imshow('Road Sign Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
