import cv2
# Importation de tes modules
from detection_pieces import extract_coin_data, nms_circles
from couleurs import classify_coins

# 1. Lancement de l'étape 1 (Détection)
img = cv2.imread("test5.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (21, 21), 0)

h, w = img.shape[:2]
min_r = int(min(h, w) * 0.04)
max_r = int(min(h, w) * 0.22)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_r*1.5, 
                           param1=80, param2=35, minRadius=min_r, maxRadius=max_r)

if circles is not None:
    import numpy as np
    raw = [(x, y, r) for x, y, r in np.uint16(np.around(circles[0]))]
    filtered = nms_circles(raw, overlap_thresh=0.6)
    
    # Récupération des données formatées : [[roi, mask, radius], ...]
    coin_data = extract_coin_data(img, filtered)
    print(f"Étape 1 terminée : {len(coin_data)} pièces extraites.")

    # 2. Lancement de l'étape 2 (Ta partie : La Couleur)
    classes_pieces = classify_coins(coin_data, sat_threshold=35)
    
    # Affichage du bilan
    print("\n--- RÉSULTATS DE LA CLASSIFICATION ---")
    for categorie, pieces in classes_pieces.items():
        print(f"{categorie} : {len(pieces)} pièce(s)")
else:
    print("Aucune pièce détectée.")