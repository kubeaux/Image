import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_saturation_histogram(roi_bgr, mask, coin_name="Piece"):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    
    hist_s = cv2.calcHist([hsv], [1], mask, [256], [0, 256])

    plt.figure(figsize=(10, 5))
    plt.plot(hist_s, color='purple', lw=2)
    plt.title(f"Histogramme de Saturation - {coin_name}")
    plt.xlabel("Intensité de Saturation (0-255)")
    plt.ylabel("Nombre de pixels")
    plt.xlim([0, 255])
    plt.grid(True, alpha=0.3)
    
    plt.axvline(x=35, color='red', linestyle='--', label='Seuil empirique (35)')
    plt.legend()
    
    plt.show()