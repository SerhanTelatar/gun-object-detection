import csv
import numpy as np
from scipy.interpolate import interp1d

def interpolate_detections(data):

    frame_nums = np.array([int(row['frame_num']) for row in data])
    track_ids = np.array([int(row['track_id']) for row in data])
    x1_vals = np.array([float(row['x1']) for row in data])
    y1_vals = np.array([float(row['y1']) for row in data])
    x2_vals = np.array([float(row['x2']) for row in data])
    y2_vals = np.array([float(row['y2']) for row in data])
    class_ids = np.array([int(row['class_id']) for row in data])
    confidences = np.array([float(row['confidence']) for row in data])

    interpolated_rows = []

    unique_ids = np.unique(track_ids)

    for tid in unique_ids:
        mask = (track_ids == tid)
        frames_tid = frame_nums[mask]
        x1_tid = x1_vals[mask]
        y1_tid = y1_vals[mask]
        x2_tid = x2_vals[mask]
        y2_tid = y2_vals[mask]
        cls_tid = class_ids[mask]
        conf_tid = confidences[mask]

        sort_idx = np.argsort(frames_tid)
        frames_tid = frames_tid[sort_idx]
        x1_tid = x1_tid[sort_idx]
        y1_tid = y1_tid[sort_idx]
        x2_tid = x2_tid[sort_idx]
        y2_tid = y2_tid[sort_idx]
        cls_tid = cls_tid[sort_idx]
        conf_tid = conf_tid[sort_idx]

        first_frame = frames_tid[0]
        last_frame = frames_tid[-1]

        
        if len(frames_tid) == 1:
            row = {
                'frame_num': str(frames_tid[0]),
                'track_id': str(tid),
                'x1': str(x1_tid[0]),
                'y1': str(y1_tid[0]),
                'x2': str(x2_tid[0]),
                'y2': str(y2_tid[0]),
                'class_id': str(cls_tid[0]),
                'confidence': str(conf_tid[0])
            }
            interpolated_rows.append(row)
            continue
        else:
            f_x1 = interp1d(frames_tid, x1_tid, kind='linear')
            f_y1 = interp1d(frames_tid, y1_tid, kind='linear')
            f_x2 = interp1d(frames_tid, x2_tid, kind='linear')
            f_y2 = interp1d(frames_tid, y2_tid, kind='linear')

        all_frames = np.arange(first_frame, last_frame + 1)

        for fr in all_frames:
            if fr in frames_tid:

                idx = np.where(frames_tid == fr)[0][0]

                row = {
                    'frame_num': str(frames_tid[idx]),
                    'track_id': str(tid),
                    'x1': str(x1_tid[idx]),
                    'y1': str(y1_tid[idx]),
                    'x2': str(x2_tid[idx]),
                    'y2': str(y2_tid[idx]),
                    'class_id': str(cls_tid[idx]),
                    'confidence': str(conf_tid[idx])
                }
            else:
                x1_i = float(f_x1(fr))
                y1_i = float(f_y1(fr))
                x2_i = float(f_x2(fr))
                y2_i = float(f_y2(fr))

                row = {
                    'frame_num': str(fr),
                    'track_id': str(tid),
                    'x1': str(x1_i),
                    'y1': str(y1_i),
                    'x2': str(x2_i),
                    'y2': str(y2_i),
                    'class_id': '0',
                    'confidence': '0'
                }

            interpolated_rows.append(row)

    return interpolated_rows


if __name__ == "__main__":

    input_csv = 'results.csv'
    output_csv = 'results_interpolated.csv'

    with open(input_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        data_list = list(reader)

    new_data = interpolate_detections(data_list)

    fieldnames = ['frame_num', 'track_id', 'x1', 'y1', 'x2', 'y2', 'class_id', 'confidence']
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_data)

    print(f"Interpolated results written to {output_csv}")
