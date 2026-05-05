import cv2
import numpy as np

img = cv2.imread("test5.jpg")
output = img.copy()
h, w = img.shape[:2]
# --- Prétraitement --- #
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Conversion en niveaux de gris (nécessaire pour HoughCircles)
gray = cv2.GaussianBlur(gray, (21, 21), 0) # Flou gaussien pour réduire le bruit et améliorer la détection
# --- Paramètres adaptatifs --- #
min_r = int(min(h, w) * 0.04) # Rayon minimum
max_r = int(min(h, w) * 0.22) # Rayon maximum
min_dist = min_r * 1.5 # Distance minimum entre deux centres de cercles
# --- Détection des cercles ---
circles = cv2.HoughCircles(
    gray,                   # image en niveaux de gris
    cv2.HOUGH_GRADIENT,     # méthode de détection
    dp=1.2,                 # résolution de l'accumulateur
    minDist=min_dist,       # distance minimale entre cercles
    param1=80,              # seuil Canny (bordures)
    param2=35,              # sensibilité (plus haut = moins de faux positifs)
    minRadius=min_r,        # rayon minimum détecté
    maxRadius=max_r         # rayon maximum détecté
)

# --- Suppression des doublons --- #
def nms_circles(circles, overlap_thresh=0.6):
    """
    Supprime les cercles dont le centre est trop proche d'un cercle plus grand (Non-Maximum Suppression)
    """
    if len(circles) == 0:
        return []
    # Trier par rayon décroissant (on garde les plus grands en priorité)
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    kept = []  # Liste des cercles conservés
    for c in circles:
        x1, y1, r1 = c
        dominated = False
        # Comparaison avec les cercles déjà gardés
        for k in kept:
            x2, y2, r2 = k
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2) # Distance entre les centres
            # Si trop proche d'un plus grand → on ignore
            if dist < r2 * overlap_thresh:
                dominated = True
                break
        # Sinon on garde le cercle
        if not dominated:
            kept.append(c)
    return kept


def extract_coin_data(img, filtered_circles):
    """
    Pour chaque cercle détecté, extrait :
      - roi    : sous-image carrée (bounding box) autour du cercle
      - mask   : masque binaire circulaire (même taille que le roi)
      - radius : rayon du cercle (int)

    Retourne une liste de listes : [[roi, mask, radius], ...]
    """
    results = []
    img_h, img_w = img.shape[:2]

    for (x, y, r) in filtered_circles:
        # --- Calcul de la bounding box (clampée aux bords de l'image) --- #
        x1 = max(0, x - r)
        y1 = max(0, y - r)
        x2 = min(img_w, x + r)
        y2 = min(img_h, y + r)
        # --- Extraction du ROI --- #
        roi = img[y1:y2, x1:x2].copy()
        # --- Création du masque circulaire --- #
        roi_h, roi_w = roi.shape[:2]
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        # Centre du cercle dans le repère du ROI
        cx = x - x1
        cy = y - y1
        cv2.circle(mask, (cx, cy), r, 255, thickness=-1)  # Cercle plein blanc
        results.append([roi, mask, r])
    return results

count = 0
coin_data = []
if circles is not None:
    raw = [(x, y, r) for x, y, r in np.uint16(np.around(circles[0]))]
    filtered = nms_circles(raw, overlap_thresh=0.6) # Seuil à ajuster selon la densité des pièces
    coin_data = extract_coin_data(img, filtered)
    # Dessin des cercles détectés
    for x, y, r in filtered:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
        count += 1
print(f"{count} pièce(s) détectée(s)")

# --- Exemple d'accès aux données --- #
for i, (roi, mask, radius) in enumerate(coin_data):
    print(f"Pièce {i+1} : radius={radius}, roi.shape={roi.shape}, mask.shape={mask.shape}")

cv2.namedWindow("Detection pieces", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detection pieces", 800, 600)
cv2.imshow("Detection pieces", output)
cv2.waitKey(0)
cv2.destroyAllWindows()