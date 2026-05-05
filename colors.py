import cv2
import numpy as np

def class_coins(coin_data, sat_threshold=20):
    """
    Classe les pièces en 4 catégories.
    Attend en entrée : coin_data = [[roi, mask, radius], ...]
    """
    res = {
        "1 Euro": [],
        "2 Euro": [],
        "Cuivre": [],
        "Or": []
    }

    mono_data = []

    for roi, mask, r in coin_data:
        #1. Conversion RGB vers HSV (Insensible à la luminosité)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        #2. Vectorisation spatiale pour éviter les boucles for
        h, w = roi.shape[:2]
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)

        #3. Création des masques radiaux (avec sécurité mask > 0 pour ignorer fond noir)
        center_mask = (dist < 0.5 * r) & (mask > 0)
        ring_mask = (dist >= 0.7 * r) & (dist < 0.95 *r) & (mask > 0)
        full_mask = (dist < 0.95 * r) & (mask > 0)

        #4. Extraction des saturations moyennes
        s_center = np.mean(S[center_mask]) if np.any(center_mask) else 0
        s_ring = np.mean(S[ring_mask]) if np.any(ring_mask) else 0

        sat_diff = abs(s_center - s_ring)
        print(f"Pièce r={r}: sat_center={s_center:.1f}, sat_ring={s_ring:.1f}, " 
              f"sat_diff={sat_diff:.1f}, bicolore={sat_diff > sat_threshold}")

        #BICOLORE VS MONOCOLORE
        if sat_diff > sat_threshold:
            if s_center < s_ring:
                res["1 Euro"].append({"radius": r, "roi": roi})
            else:
                res["2 Euro"].append({"radius": r, "roi": roi})
        else:
            #Calcul de la teinte (H) moyenne pour différencier Cuivre et Or plus tard
            h_mean = np.mean(H[full_mask]) if np.any(full_mask) else 0
            print(f"  → h_mean={h_mean:.1f}")
            mono_data.append({'radius': r, 'roi': roi, 'h_mean': h_mean})
        
        #CUIVRE VS OR (OTSU)
    if mono_data:
        h_values = np.array([item['h_mean'] for item in mono_data], dtype=np.uint8)

        #Sécurité : Otsu uniquement si on a une vraie variance (bimodalité)
        if np.std(h_values) > 5.0:
            otsu_thresh, _ = cv2.threshold(h_values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            otsu_thresh = 15.0 # Seuil de repli empirique 

        print(f"\n[Otsu] std={np.std(h_values):.2f}, threshold={otsu_thresh}")
        print(f"[Otsu] h_values={h_values.tolist()}")
        
        for item in mono_data:
            if item['h_mean'] <= otsu_thresh:
                res["Cuivre"].append({"radius": item['radius'], "roi": item['roi']})
            else:
                res["Or"].append({"radius": item['radius'], "roi": item['roi']})
    return res