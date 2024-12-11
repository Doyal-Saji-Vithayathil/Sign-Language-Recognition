import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except (FileNotFoundError, KeyError):
    print("Error: Model file not found or incorrect model format.")
    exit()

# Set up camera feed (try different indexes if this one fails)
cap = cv2.VideoCapture(0)  # Change to 0 or 1 if 2 doesn't work
if not cap.isOpened():
    print("Error: Could not open video feed.")
    exit()

# MediaPipe hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary for predictions
labels_dict = {0: 'A', 1: 'B', 2: 'C',3: 'D',4: 'E',5: 'F', 6: 'G',7: 'H',8: 'I',9:'K',10:'L',11:'M',12:'N',13:'O',14:'P',15:'Q',16:'R',17:'S',18:'T',19:'U',20:'V',21:'W',22:'X',23:'Y'}

while True:
    data_aux = []
    x_, y_ = [], []

    ret, frame = cap.read()
    if not ret:
        print("Warning: Frame capture failed, retrying...")
        continue  # Skip to the next loop iteration if frame capture failed

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Collect landmarks' x and y for normalization
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Normalize coordinates based on the minimum x and y values
            min_x, min_y = min(x_), min(y_)
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min_x)
                data_aux.append(landmark.y - min_y)

            # Predict character based on model
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict.get(int(prediction[0]), "?")

            # Define bounding box for hand
            x1, y1 = int(min_x * W) - 10, int(min_y * H) - 10
            x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
