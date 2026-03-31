import cv2
import numpy as np

def get_colors(img, sat_threshold=35):
    results = {
        "1": [],
        "2": [],
        "cuivre": [],
        "or": []
    }

    monocolor_data = []

    for coin in img:
        roi = coin['roi']
        mask = coin['mask']
        r = coin['radius']

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

        h, w = roi.shape[:2]
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        center_mask = (dist < 0.6 * r) & (mask > 0)
        ring_mask = (dist >= 0.6 * r) & (dist < 0.95 * r) & (mask > 0)
        full_mask = (dist < 0.95 * r) & (mask > 0)

        sat_center = np.mean(S[center_mask]) if np.any(center_mask) else 0
        sat_ring = np.mean(S[ring_mask]) if np.any(ring_mask) else 0

        sat_diff = abs(sat_center - sat_ring)

        if sat_diff > sat_threshold:
            if sat_center < sat_ring:
                results["1"].append(coin)
            else:
                results["2"].append(coin)
        else:
            h_mean = np.mean(H[full_mask]) if np.any(full_mask) else 0
            monocolor_data.append({'coin': coin, 'h_mean': h_mean})
    
    if monocolor_data:
        h_values = np.array([item['h_mean'] for item in monocolor_data], dtype=np.uint8)
        if np.std(h_values) > 5.0:
            otsu_thresh, _ = cv2.threshold(h_values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            otsu_thresh = 20.0
        for item in monocolor_data:
            if item['h_mean'] <= otsu_thresh:
                results["cuivre"].append(item['coin'])
            else:
                results["or"].append(item['coin'])
        
    final_output = {}
    for group, coins in results.items():
        final_output[group] = [{"radius": c['radius']} for c in coins]
    
    return final_output