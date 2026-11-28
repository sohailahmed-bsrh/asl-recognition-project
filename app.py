import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ==========================
# Load Trained Model
# ==========================
model = load_model("asl_model (3).keras")
import time

sentence = ""
last_prediction = ""
last_time = time.time()

# ==========================
# Class Names
# ==========================
class_names = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "del","nothing","space"
]

# ==========================
# Start Webcam
# ==========================
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror Image
    frame = cv2.flip(frame, 1)

    # ROI Coordinates
    x1, y1 = 300, 100
    x2, y2 = 550, 350

    # Draw ROI Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    # Crop ROI
    roi = frame[y1:y2, x1:x2]

    # Preprocess
    roi_resized = cv2.resize(roi, (224,224))
    roi_resized = roi_resized.astype("float32") / 255.0
    roi_resized = np.expand_dims(roi_resized, axis=0)

    # Prediction
    prediction = model.predict(roi_resized, verbose=0)

    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    label = class_names[predicted_index]

    current_time = time.time()

    # Add letter only if confidence is high
    if confidence > 90:

        if label == last_prediction:

            if current_time - last_time >= 1:

                if label == "space":
                    sentence += " "

                elif label == "del":
                    sentence = sentence[:-1]

                elif label != "nothing":
                    sentence += label
                    sentence = sentence[-40:]  # Keep only last 40 characters   

                last_time = current_time

        else:
            last_prediction = label
            last_time = current_time

    # Show Prediction
    cv2.putText(
        frame,
        f"{label} ({confidence:.2f}%)",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Show Typed Text
    cv2.putText(
        frame,
        "Text: " + sentence,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )
    # ==========================================
    # CREATE SIDE PANEL
    # ==========================================

    panel = np.zeros((frame.shape[0], 350, 3), dtype=np.uint8)

    cv2.putText(panel,
                "Prediction",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2)

    cv2.putText(panel,
                label,
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255,255,255),
                2)

    cv2.putText(panel,
                f"Confidence: {confidence:.2f}%",
                (20,130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,0),
                2)

    cv2.putText(panel,
                "Typed Text",
                (20,190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2)

    cv2.putText(panel,
                sentence[-40:],
                (20,230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2)

    # Merge webcam and panel
    output = np.hstack((frame, panel))

    # Show Windows
    cv2.imshow("ROI", roi)
    cv2.imshow("ASL Recognition", output)

    # Keyboard Controls
    key = cv2.waitKey(1) & 0xFF

    # C = Clear typed text
    if key == ord("c"):
        sentence = ""

    # Q = Quit
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()