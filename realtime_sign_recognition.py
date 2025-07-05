import cv2
import mediapipe as mp
import numpy as np
import joblib

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    base = landmarks[0]  # wrist
    landmarks -= base
    return landmarks.flatten().tolist()

def extract_landmarks(image, hands):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (640, 480))  # Resize for better detection
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return normalize_landmarks(landmarks), hand_landmarks
    else:
        return None, None


if __name__ == "__main__":
    # Load model and labels
    model, LABELS = joblib.load("sign_language_model.pkl")
    print("Model loaded. Labels:", LABELS)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks, hand_landmarks = extract_landmarks(frame, hands)

        if landmarks is not None:
            norm_landmarks = normalize_landmarks(landmarks)
            pred = model.predict([norm_landmarks])[0]

            # Draw landmarks and predicted label
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"Predicted: {pred}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 3)

        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
