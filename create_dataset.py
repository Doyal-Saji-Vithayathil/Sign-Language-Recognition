import sys
import os
import pickle
import mediapipe as mp
import cv2

from absl import logging

# Set logging level to only show errors
logging.set_verbosity(logging.ERROR)

# Redirect stderr temporarily to suppress MediaPipe warnings
original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Revert stderr back to original
sys.stderr = original_stderr

# Process images and save landmarks data
DATA_DIR = './data'
data = []
labels = []

if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
    raise FileNotFoundError(f"The directory {DATA_DIR} does not exist or is empty.")

# Process each class directory inside DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    
    if not os.path.isdir(dir_path):
        print(f"Skipping {dir_path} because it is not a directory.")
        continue

    # Process each image in the class directory
    for img_path in os.listdir(dir_path):
        img_file_path = os.path.join(dir_path, img_path)
        data_aux = []
        x_, y_ = [], []

        # Load the image and convert to RGB
        img = cv2.imread(img_file_path)
        if img is None:
            print(f"Image {img_file_path} could not be read. Skipping.")
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to get hand landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Gather x and y coordinates for normalization
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                # Normalize coordinates based on the minimum x and y values
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))
                    
            # Append the processed data and corresponding label
            data.append(data_aux)
            labels.append(dir_)
            print(f"Processed image {img_path} from class {dir_}")
        else:
            print(f"No hand detected in image {img_path}. Skipping.")

# Save data to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data processing complete. Saved to data.pickle.")
