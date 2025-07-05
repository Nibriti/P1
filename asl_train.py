import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# === Parameters ===
DATASET_PATH = "asl_alphabet_train/asl_alphabet_train/asl_alphabet_train"

LABELS = ['A', 'B', 'C', 'D']  # Update with all your sign labels

# === Debugging dataset folders ===
print(f"Checking contents of DATASET_PATH: '{DATASET_PATH}'")
try:
    contents = os.listdir(DATASET_PATH)
    print(f"Folders found: {contents}")
    for label in LABELS:
        path = os.path.join(DATASET_PATH, label)
        print(f"Label folder '{label}' exists? {os.path.exists(path)}")
except Exception as e:
    print(f"Error accessing DATASET_PATH: {e}")


# === Setup MediaPipe ===
mp_hands = mp.solutions.hands

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    base = landmarks[0]  # wrist is index 0
    landmarks -= base     # translate so wrist is origin
    return landmarks.flatten().tolist()

def extract_landmarks(image, hands_detector):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(image_rgb)
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks
    else:
        return None

def load_data(dataset_path, labels, hands_detector):
    X, y = [], []
    for label in labels:
        folder = os.path.join(dataset_path, label)
        if not os.path.exists(folder):
            print(f"Warning: Folder not found: {folder}")
            continue
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image: {img_path}")
                continue

            landmarks = extract_landmarks(img, hands_detector)
            if landmarks is None:
                print(f"No landmarks detected in {img_path}")
                continue

            landmarks = normalize_landmarks(landmarks)
            X.append(landmarks)
            y.append(label)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

    print("Loading and extracting features from images...")
    X, y = load_data(DATASET_PATH, LABELS, hands)
    print(f"Loaded {len(X)} samples.")

    if len(X) == 0:
        print("No data found. Exiting.")
        hands.close()
        exit()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training classifier...")
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save model and labels
    joblib.dump((model, LABELS), "sign_language_model.pkl")
    print("Model saved as sign_language_model.pkl")

    hands.close()

