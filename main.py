import cv2
import numpy as np
import tensorflow as tf
import json

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="Resources/Model/model_unquant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("Resources/Model/labels.txt", "r") as f:
    labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]

def preprocess_image(frame, input_shape):
    # Resize the frame to the input shape
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    # Normalize the image to the range [0, 1]
    normalized_frame = resized_frame / 255.0
    # Add batch dimension
    input_data = np.expand_dims(normalized_frame, axis=0).astype(np.float32)
    return input_data

def classify_frame(frame):
    input_shape = input_details[0]['shape']
    input_data = preprocess_image(frame, input_shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output_data), np.max(output_data)

# Attempt to capture video using different backends
backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2, cv2.CAP_GSTREAMER]

for backend in backends:
    cap = cv2.VideoCapture(0, backend)
    if cap.isOpened():
        print(f"Using backend: {backend}")
        break
else:
    raise Exception("Could not open video device with any backend")

# Initialize a set to track unique labels
detected_labels = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Classify the frame
    class_id, confidence = classify_frame(frame)
    label = labels[class_id]

    # Track unique labels with confidence above 0.80, excluding "nothing"
    if label.lower() != "nothing" and confidence > 0.80:
        detected_labels.add(label)

    # Display the results on the frame
    cv2.putText(frame, f"{label}: {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Webcam", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

# Convert set to list for JSON serialization
detected_labels_list = list(detected_labels)

# Save the detected labels to a JSON file
with open("detected_items.json", "w") as json_file:
    json.dump(detected_labels_list, json_file, indent=4)

print("Detected labels have been saved to 'detected_items.json'.")
