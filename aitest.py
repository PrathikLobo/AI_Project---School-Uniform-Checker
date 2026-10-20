
import cv2
import csv
import os
from datetime import datetime
from ultralytics import YOLO

MODEL_PATH = "weights.pt"
LOG_FILE = "uniform_id_checks.csv"
CONFIDENCE_THRESHOLD = 0.45

model = YOLO(MODEL_PATH)

CORRECT_UNIFORM = {"correct_uniform", "proper_uniform"}
INCORRECT_UNIFORM = {"incorrect_uniform", "wrong_uniform"}
GENERAL_UNIFORM = {"uniform", "school_uniform"}
ID_CARD = {"id_card", "school_id", "student_id", "identity_card"}


def normalize_class(name):
    return name.lower().strip().replace(" ", "_").replace("-", "_")


def save_result(uniform_status, id_status, overall):
    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Date and Time",
                "Uniform Status",
                "ID Card Status",
                "Overall Result"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            uniform_status,
            id_status,
            overall
        ])

    print("Inspection result saved.")


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit("Error: Could not access the webcam.")

print("School Uniform or ID Checker started.")
print("Press S to save an inspection result.")
print("Press Q to quit.")

try:
    while True:
        success, frame = cap.read()

        if not success:
            print("Error: Could not read webcam frame.")
            break

        result = model.predict(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )[0]

        display_frame = result.plot()

        detected_classes = set()

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = normalize_class(
                result.names[class_id]
            )
            detected_classes.add(class_name)

        has_correct_uniform = bool(
            detected_classes & CORRECT_UNIFORM
        )
        has_incorrect_uniform = bool(
            detected_classes & INCORRECT_UNIFORM
        )
        has_general_uniform = bool(
            detected_classes & GENERAL_UNIFORM
        )

        if has_incorrect_uniform:
            uniform_status = "UNIFORM ISSUE DETECTED"
        elif has_correct_uniform:
            uniform_status = "CORRECT UNIFORM DETECTED"
        elif has_general_uniform:
            uniform_status = "UNIFORM DETECTED - VERIFY"
        else:
            uniform_status = "UNIFORM NOT DETECTED"

        has_id_card = bool(detected_classes & ID_CARD)

        if has_id_card:
            id_status = "ID CARD DETECTED"
        else:
            id_status = "ID CARD NOT DETECTED"

        if has_correct_uniform and has_id_card:
            overall = "PASS"
        else:
            overall = "CHECK REQUIRED"

        cv2.putText(
            display_frame,
            uniform_status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0) if has_correct_uniform else (0, 0, 255),
            2
        )

        cv2.putText(
            display_frame,
            id_status,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0) if has_id_card else (0, 0, 255),
            2
        )

        cv2.putText(
            display_frame,
            "Result: " + overall,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if overall == "PASS" else (0, 0, 255),
            2
        )

        cv2.imshow("School Uniform or ID Checker", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            save_result(uniform_status, id_status, overall)

        elif key == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
