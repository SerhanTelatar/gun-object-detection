from ultralytics import YOLO
import cv2
from utils import get_car

cap = cv2.VideoCapture("gun.mp4")

if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

model = YOLO("runs/detect/train17/weights/best.pt")

while True:
    # Read each frame from the video
    ret, frame = cap.read()

    # If the video ends or no frames are returned, break the loop
    if not ret:
        print("End of video or cannot read the frame.")
        break

    results = model.predict(frame)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        conf = box.conf[0]
        cls = int(box.cls[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Class {cls}: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Display the frame
    cv2.imshow("frame", frame)

    # Wait for a key press for 1ms, if 'q' is pressed, exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting on 'q' press.")
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


