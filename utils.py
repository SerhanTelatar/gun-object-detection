import csv

def write_csv(results, filename):
    """
    results sözlüğü:
      results[frame_num] = [
          [track_id, x1, y1, x2, y2, class_id, conf],
          [track_id, x1, y1, x2, y2, class_id, conf],
          ...
      ]
    formatında tutulur.

    Bu fonksiyon, sonuçları CSV dosyasına yazar.
    """
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        # CSV sütun başlıkları
        writer.writerow(["frame_num", "track_id", "x1", "y1", "x2", "y2", "class_id", "confidence"])
        
        # Sözlükteki her kareyi ve o karedeki takip sonuçlarını yaz
        for frame_num, detections in results.items():
            for det in detections:
                writer.writerow([frame_num] + det)