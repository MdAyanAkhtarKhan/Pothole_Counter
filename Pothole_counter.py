from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import math
from sort import *

# Video
cap = cv2.VideoCapture("Videos/Recording_pothole1.mp4")

# Model
model = YOLO("yolo-weights/best_pothole_mlarge.pt")

classNames = ["Pothole"]

# SORT Tracker
tracker = Sort(
    max_age=50,
    min_hits=3,
    iou_threshold=0.3
)

# Counting Line
limits = [0, 500, 1280, 500]

# Counted IDs
totalCount = []

while True:

    success, img = cap.read()

    if not success:
        break

    img = cv2.resize(img, (1280, 720))

    detections = np.empty((0, 5))

    # YOLO Detection
    results = model(img, stream=True)

    for r in results:

        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:

            conf = float(box.conf[0])

            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            w = x2 - x1
            h = y2 - y1

            # Ignore very small distant potholes
            area = w * h

            if area < 2500:
                continue

            currentArray = np.array(
                [x1, y1, x2, y2, conf]
            )

            detections = np.vstack(
                (detections, currentArray)
            )

    # SORT Tracking
    resultsTracker = tracker.update(detections)

    # Draw counting line
    cv2.line(
        img,
        (limits[0], limits[1]),
        (limits[2], limits[3]),
        (0, 0, 255),
        5
    )

    for result in resultsTracker:

        x1, y1, x2, y2, id = result

        x1, y1, x2, y2 = map(
            int,
            [x1, y1, x2, y2]
        )

        id = int(id)

        w = x2 - x1
        h = y2 - y1

        cx = x1 + w // 2
        cy = y1 + h // 2

        # Draw Box
        cvzone.cornerRect(
            img,
            (x1, y1, w, h),
            l=9,
            rt=2,
            colorR=(255, 0, 255)
        )

        cvzone.putTextRect(
            img,
            f'ID {id}',
            (max(0, x1), max(35, y1)),
            scale=1,
            thickness=1
        )

        cv2.circle(
            img,
            (cx, cy),
            5,
            (0, 255, 255),
            cv2.FILLED
        )

        # Count when crossing line
        if (
            limits[0] < cx < limits[2]
            and
            limits[1] - 25 < cy < limits[1] + 25
        ):

            if id not in totalCount:

                totalCount.append(id)

                cv2.line(
                    img,
                    (limits[0], limits[1]),
                    (limits[2], limits[3]),
                    (0, 255, 0),
                    5
                )

    # Counter Display
    cv2.rectangle(
        img,
        (20, 20),
        (320, 90),
        (255, 0, 0),
        -1
    )

    cv2.putText(
        img,
        f'Potholes: {len(totalCount)}',
        (35, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3
    )

    cv2.imshow(
        "Pothole Counter",
        img
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Final Count:", len(totalCount))