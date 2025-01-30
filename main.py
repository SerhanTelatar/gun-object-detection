import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort  # SORT algoritması
from utils import write_csv

cap = cv2.VideoCapture("deadpool.mov")

model = YOLO("runs/detect/train17/weights/best.pt")

tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)

frame_num = 0
all_results = {}

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read the frame.")
        break
    frame_num += 1

    detection_results = model.predict(frame)
    
    detections = []
    for box in detection_results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0].item())
        cls_id = int(box.cls[0].item())
        detections.append([x1, y1, x2, y2, conf, cls_id])

    sort_input = np.array([[d[0], d[1], d[2], d[3], d[4]] for d in detections])
    if len(sort_input) == 0:
        sort_input = np.empty((0, 5))

    tracked_objects = tracker.update(sort_input)

    current_frame_results = []
    for t in tracked_objects:
        x1t, y1t, x2t, y2t, track_id = t
        best_i = None
        best_iou = 0.0
        
        for i, det in enumerate(detections):
            x1d, y1d, x2d, y2d, confd, clsd = det
            
            xx1 = max(x1t, x1d)
            yy1 = max(y1t, y1d)
            xx2 = min(x2t, x2d)
            yy2 = min(y2t, y2d)
            w = max(0., xx2 - xx1 + 1.)
            h = max(0., yy2 - yy1 + 1.)
            overlap = w * h
            area1 = (x2t - x1t + 1.) * (y2t - y1t + 1.)
            area2 = (x2d - x1d + 1.) * (y2d - y1d + 1.)
            iou = overlap / (area1 + area2 - overlap) if (area1+area2-overlap) != 0 else 0
            
            if iou > best_iou:
                best_iou = iou
                best_i = i

        if best_i is not None and best_iou > 0.1:
            conf_final = detections[best_i][4]
            cls_final = detections[best_i][5]
        else:
            conf_final = 0.0
            cls_final = -1

        current_frame_results.append([
            int(track_id),
            int(x1t), int(y1t), int(x2t), int(y2t),
            cls_final,
            float(conf_final),
        ])

    all_results[frame_num] = current_frame_results

    for res in current_frame_results:
        track_id, x1t, y1t, x2t, y2t, cls_id, conf = res
        cv2.rectangle(frame, (x1t, y1t), (x2t, y2t), (0, 255, 0), 2)
        label = f"ID {track_id} Cls {cls_id} Conf {conf:.2f}"
        cv2.putText(frame, label, (x1t, y1t - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting on 'q' press.")
        break

cap.release()
cv2.destroyAllWindows()

# 7. Sonuçları CSV dosyasına yaz
write_csv(all_results, "results.csv")
