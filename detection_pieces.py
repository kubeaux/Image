"""
Détection de pièces en euros - Étapes 1 & 2
L3 MIAGE - Projet Image

Étape 1 : Prétraitement de l'image
Étape 2 : Détection des pièces (HoughCircles)
"""

import cv2
import numpy as np
import sys
from pathlib import Path


# ==============================================================
# ÉTAPE 1 : PRÉTRAITEMENT
# ==============================================================

def pretraiter_image(chemin_image: str) -> tuple:
    """
    Charge et prétraite l'image pour faciliter la détection.

    Retourne :
        image_originale  : image BGR originale (pour l'affichage final)
        image_grise      : image en niveaux de gris floutée (pour HoughCircles)
    """
    # Chargement
    image = cv2.imread(chemin_image)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {chemin_image}")

    # Redimensionnement si l'image est trop grande (pour accélérer le traitement)
    # On garde un max de 1200px sur le grand côté
    hauteur, largeur = image.shape[:2]
    max_dim = 1200
    if max(hauteur, largeur) > max_dim:
        facteur = max_dim / max(hauteur, largeur)
        nouvelle_largeur = int(largeur * facteur)
        nouvelle_hauteur = int(hauteur * facteur)
        image = cv2.resize(image, (nouvelle_largeur, nouvelle_hauteur),
                           interpolation=cv2.INTER_AREA)
        print(f"  Image redimensionnée : {largeur}x{hauteur} → {nouvelle_largeur}x{nouvelle_hauteur}")

    # Conversion en niveaux de gris
    image_grise = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Flou gaussien pour réduire le bruit (kernel 9x9)
    # Un kernel plus grand = détection moins sensible aux textures
    image_flouee = cv2.GaussianBlur(image_grise, (9, 9), 2)

    return image, image_flouee


# ==============================================================
# ÉTAPE 2 : DÉTECTION DES PIÈCES (HoughCircles)
# ==============================================================

def detecter_pieces(image_flouee: np.ndarray, image_originale: np.ndarray) -> list:
    """
    Détecte les cercles (pièces) dans l'image prétraitée.

    Paramètres HoughCircles à comprendre :
        dp          : résolution de l'accumulateur (1 = même résolution que l'image)
        minDist     : distance minimale entre deux centres de cercles détectés
        param1      : seuil haut pour la détection de contours (Canny interne)
        param2      : seuil de l'accumulateur (plus bas = plus de faux positifs)
        minRadius   : rayon minimum en pixels
        maxRadius   : rayon maximum en pixels

    Retourne une liste de cercles (x, y, rayon)
    """
    hauteur, largeur = image_flouee.shape[:2]

    # Estimation des rayons min/max selon la taille de l'image
    # On suppose qu'une pièce fait entre 3% et 30% de la largeur de l'image
    rayon_min = int(largeur * 0.03)
    rayon_max = int(largeur * 0.30)
    distance_min = int(rayon_min * 1.8)  # évite les détections doublées

    print(f"  Rayon min : {rayon_min}px | Rayon max : {rayon_max}px")

    cercles = cv2.HoughCircles(
        image_flouee,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=distance_min,
        param1=100,   # seuil Canny (bord fort)
        param2=30,    # seuil accumulateur (à ajuster selon vos images)
        minRadius=rayon_min,
        maxRadius=rayon_max
    )

    if cercles is None:
        print("  Aucun cercle détecté. Essayez d'abaisser param2.")
        return []

    # Conversion en entiers
    cercles = np.round(cercles[0, :]).astype(int)
    print(f"  {len(cercles)} cercle(s) détecté(s)")

    return cercles


# ==============================================================
# AFFICHAGE DES RÉSULTATS
# ==============================================================

def afficher_detection(image_originale: np.ndarray, cercles: list, chemin_sortie: str = None):
    """
    Dessine les cercles détectés sur l'image et l'affiche.
    Sauvegarde optionnelle si chemin_sortie est fourni.
    """
    image_annotee = image_originale.copy()

    for i, (x, y, r) in enumerate(cercles):
        # Cercle détecté (contour en vert)
        cv2.circle(image_annotee, (x, y), r, (0, 200, 0), 2)
        # Centre de la pièce (point rouge)
        cv2.circle(image_annotee, (x, y), 4, (0, 0, 255), -1)
        # Numéro de la pièce
        cv2.putText(image_annotee, f"#{i+1}", (x - 15, y - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    # Affichage
    cv2.imshow("Détection des pièces", image_annotee)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Sauvegarde
    if chemin_sortie:
        cv2.imwrite(chemin_sortie, image_annotee)
        print(f"  Image annotée sauvegardée : {chemin_sortie}")

    return image_annotee


# ==============================================================
# EXTRACTION DES ROIs (Regions of Interest)
# ==============================================================

def extraire_rois(image_originale: np.ndarray, cercles: list) -> list:
    """
    Extrait chaque pièce détectée sous forme de sous-image carrée.
    Ces ROIs seront utilisées à l'étape 3 pour la classification.

    Retourne une liste de dictionnaires :
        { 'roi': image_couleur, 'centre': (x, y), 'rayon': r }
    """
    hauteur, largeur = image_originale.shape[:2]
    rois = []

    for (x, y, r) in cercles:
        # Zone de découpe carrée autour du cercle (avec marge de 5px)
        marge = 5
        x1 = max(0, x - r - marge)
        y1 = max(0, y - r - marge)
        x2 = min(largeur, x + r + marge)
        y2 = min(hauteur, y + r + marge)

        roi = image_originale[y1:y2, x1:x2]

        rois.append({
            'roi': roi,
            'centre': (x, y),
            'rayon': r
        })

    return rois


# ==============================================================
# SCRIPT PRINCIPAL
# ==============================================================

if __name__ == "__main__":
    # Récupération du chemin de l'image depuis les arguments
    if len(sys.argv) < 2:
        print("Usage : python detection_pieces.py <chemin_image>")
        print("Exemple : python detection_pieces.py photo_pieces.jpg")
        sys.exit(1)

    chemin_image = sys.argv[1]
    chemin_sortie = Path(chemin_image).stem + "_detection.jpg"

    print("\n=== ÉTAPE 1 : Prétraitement ===")
    image_originale, image_flouee = pretraiter_image(chemin_image)

    print("\n=== ÉTAPE 2 : Détection des pièces ===")
    cercles = detecter_pieces(image_flouee, image_originale)

    if cercles is not None and len(cercles) > 0:
        print("\n=== Affichage et extraction des ROIs ===")
        afficher_detection(image_originale, cercles, chemin_sortie)
        rois = extraire_rois(image_originale, cercles)
        print(f"  {len(rois)} ROI(s) extraite(s) → prêtes pour la classification (étape 3)")
    else:
        print("\nAucune pièce détectée. Conseils :")
        print("  - Réduire param2 dans HoughCircles (ex: 20)")
        print("  - Vérifier l'éclairage de la photo")
        print("  - Augmenter le flou gaussien (kernel plus grand)")