import cv2
import numpy as np
import pandas as pd

def draw_border(img,
                top_left,
                bottom_right,
                color=(0, 255, 0),
                thickness=3,
                line_length_x=50,
                line_length_y=50):
    """
    Çerçevenin dört köşesine “L” biçimli çizgiler eklemek için yardımcı fonksiyon.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Top-left
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    # Bottom-left
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    # Top-right
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)

    return img


results = pd.read_csv('results_interpolated.csv')

video_path = 'deadpool.mov'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('out_gun.mp4', fourcc, fps, (width, height))

track_ids = results['track_id'].unique()

guns = {}

for tid in track_ids:
    df_tid = results[results['track_id'] == tid]
    if len(df_tid) == 0:
        continue

    max_conf = df_tid['confidence'].max()
    best_row = df_tid[df_tid['confidence'] == max_conf].iloc[0]

    best_frame = int(best_row['frame_num'])
    x1_best = int(best_row['x1'])
    y1_best = int(best_row['y1'])
    x2_best = int(best_row['x2'])
    y2_best = int(best_row['y2'])

    cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame)
    ret, frame = cap.read()
    if not ret:
        guns[tid] = {"crop": None, "label": f"Gun (ID={tid})"}
        continue

    if x2_best <= x1_best or y2_best <= y1_best:
        guns[tid] = {"crop": None, "label": f"Gun (ID={tid})"}
        continue

    gun_crop = frame[y1_best:y2_best, x1_best:x2_best]
    if gun_crop.size == 0:
        guns[tid] = {"crop": None, "label": f"Gun (ID={tid})"}
        continue

    H0, W0, _ = gun_crop.shape
    target_h = 200
    target_w = int(W0 * (target_h / H0)) if H0 != 0 else W0
    if target_h > 0 and target_w > 0:
        gun_crop = cv2.resize(gun_crop, (target_w, target_h))

    guns[tid] = {
        "crop": gun_crop,
        "label": f"Track={tid} conf={max_conf:.2f}"
    }

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_num = -1

while True:
    ret, frame = cap.read()
    frame_num += 1
    if not ret:
        break

    df_frame = results[results['frame_num'] == frame_num]
    for idx, row in df_frame.iterrows():
        tid = row['track_id']
        x1 = int(row['x1'])
        y1 = int(row['y1'])
        x2 = int(row['x2'])
        y2 = int(row['y2'])

        draw_border(frame,
                    (x1, y1),
                    (x2, y2),
                    color=(0, 255, 0),
                    thickness=3,
                    line_length_x=50,
                    line_length_y=50)

        gun_info = guns.get(tid, None)
        if not gun_info:
            continue
        gun_crop = gun_info["crop"]
        gun_label = gun_info["label"]

        if gun_crop is None:
            continue

        H, W, _ = gun_crop.shape
        top_insert = y1 - H - 60
        left_insert = (x1 + x2 - W) // 2

        if top_insert < 0:
            top_insert = 0
        if left_insert < 0:
            left_insert = 0
        if top_insert + H >= frame.shape[0]:
            continue
        if left_insert + W >= frame.shape[1]:
            continue

        frame[top_insert:top_insert + H,
              left_insert:left_insert + W] = gun_crop

        text_zone_h = 30
        text_zone_top = top_insert - text_zone_h
        if text_zone_top < 0:
            text_zone_top = 0
        if text_zone_top + text_zone_h > frame.shape[0]:
            continue

        frame[text_zone_top:text_zone_top + text_zone_h,
              left_insert:left_insert + W] = (255, 255, 255)

        font_scale = 0.8
        font_thickness = 2
        (tw, th), _ = cv2.getTextSize(
            gun_label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness
        )
        text_x = left_insert + max(0, (W - tw) // 2)
        text_y = text_zone_top + (text_zone_h + th) // 2

        cv2.putText(
            frame,
            gun_label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness
        )

    out.write(frame)


cap.release()
out.release()
cv2.destroyAllWindows()
print("Tamamlandı! 'out_gun.mp4' dosyasını kontrol edebilirsiniz.")
