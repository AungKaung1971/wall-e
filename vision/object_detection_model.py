from ultralytics import YOLO
import cv2

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")  # or yolov8s.pt for more accuracy


def detect_objects(frame):
    """
    Runs YOLOv8 on this frame.
    Returns a list of detected object class names.
    """
    results = model(frame, verbose=False)[0]

    labels = []
    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls]
        labels.append(label)

        # optional bounding boxes
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return labels


# -----------------------
# TEST MODE
# -----------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        labels = detect_objects(frame)
        print("Detected:", labels)

        cv2.imshow("YOLO Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# run code
# python vision/object_detection_model.py
