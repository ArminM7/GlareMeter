import cv2
import numpy as np
import random
from pathlib import Path

clean_dir = Path("data/clean")
output_img_dir = Path("data/train/images")
output_mask_dir = Path("data/train/masks")
output_img_dir.mkdir(parents=True, exist_ok=True)
output_mask_dir.mkdir(parents=True, exist_ok=True)

def generate_large_glare_mask(h, w):
    mask = np.zeros((h, w), dtype=np.float32)
    num_blobs = random.randint(1, 4)   # 1-4 große Flecken pro Bild
    for _ in range(num_blobs):
        # Erzeuge eine unregelmäßige Form über mehrere zufällige Punkte
        num_points = random.randint(4, 8)
        points = []
        # Zufälligen Mittelpunkt und Radius für einen groben Bereich
        cx = random.randint(w//4, 3*w//4)
        cy = random.randint(h//4, 3*h//4)
        base_radius = random.randint(100, 400)  # große Ausdehnung
        for _ in range(num_points):
            angle = random.uniform(0, 2*np.pi)
            r = random.uniform(0.6, 1.0) * base_radius
            px = int(cx + r * np.cos(angle))
            py = int(cy + r * np.sin(angle))
            # Begrenzen auf Bildgrenzen
            px = np.clip(px, 0, w-1)
            py = np.clip(py, 0, h-1)
            points.append([px, py])
        
        # Polygon aus Punkten füllen
        pts = np.array(points, dtype=np.int32)
        temp_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(temp_mask, [pts], 255)
        
        # Großen Weichzeichner anwenden, um weichen Übergang zu bekommen
        ksize = random.choice([51, 71, 101])
        if ksize % 2 == 0: ksize += 1  # muss ungerade sein
        soft_mask = cv2.GaussianBlur(temp_mask.astype(np.float32), (ksize, ksize), 0)
        # Optional: zusätzliches Weichzeichnen mit großem Sigma
        soft_mask = cv2.GaussianBlur(soft_mask, (101, 101), 50)
        
        # Maximalwert über alle Flecken nehmen (Überlappungen summieren, aber bei 255 kappen)
        mask = np.maximum(mask, soft_mask)
    
    # Normiere auf [0, 1]
    mask = np.clip(mask, 0, 255)
    return mask.astype(np.float32) / 255.0


def create_glare_mask(h, w):
    """
    Erzeugt eine einzelne Glanzmaske mit Kern + weichem Halo.
    Gibt Float-Maske (0..1) zurück.
    """
    mask = np.zeros((h, w), dtype=np.float32)
    
    # Zufällige Position und Orientierung
    cx = random.randint(w//4, 3*w//4)
    cy = random.randint(h//4, 3*h//4)
    angle = random.randint(0, 180)
    
    # Achsen der inneren Ellipse (Kern) – kleiner
    inner_a = random.randint(30, 120)   # halbe Breite
    inner_b = random.randint(30, 120)   # halbe Höhe
    
    # Achsen der äußeren Ellipse (Halo) – deutlich größer
    outer_a = inner_a + random.randint(80, 250)
    outer_b = inner_b + random.randint(80, 250)
    
    # 1. Äußere Ellipse zeichnen (ganz weiß)
    outer_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(outer_mask, (cx, cy), (outer_a, outer_b), angle, 0, 360, 255, -1)
    
    # 2. Stark weichzeichnen – sigma abhängig von der Größe des Halos
    sigma = (outer_a - inner_a) / 2.0   # bestimmt die Weichheit
    if sigma < 1: sigma = 1
    blurred = cv2.GaussianBlur(outer_mask.astype(np.float32), (0, 0), sigma)
    blurred = np.clip(blurred, 0, 255)
    
    # 3. Innere Ellipse (Kern) zeichnen – ganz weiß
    inner_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(inner_mask, (cx, cy), (inner_a, inner_b), angle, 0, 360, 255, -1)
    
    # 4. Beide kombinieren: Maximum nehmen (Kern auf 255 zwingen)
    combined = np.maximum(blurred, inner_mask.astype(np.float32))
    
    # Normalisieren auf [0, 1]
    return combined / 255.0

def apply_glare(image, soft_mask):
    """Überstrahlt das Bild mit einer weichen Glanzmaske."""
    img_float = image.astype(np.float32)
    # Erzeuge einen weißen Schleier (255), der je nach Maskenintensität eingeblendet wird
    alpha = soft_mask[..., np.newaxis]  # 3 Kanäle
    # Mischung: Original * (1-alpha) + Weiß * alpha
    img_glare = img_float * (1 - alpha) + 255 * alpha
    img_glare = np.clip(img_glare, 0, 255).astype(np.uint8)
    
    # Für das Label: Schwellwert bei z.B. 0.3 (30% Helligkeit) -> alles darüber ist "Glanz"
    binary_mask = (soft_mask > 0.3).astype(np.uint8) * 255
    return img_glare, binary_mask


def generate_multiple_glare_masks(h, w, count=2):
    final_mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(count):
        mask = create_glare_mask(h, w)
        final_mask = np.maximum(final_mask, mask)   # überlappen, aber nicht heller als 1
    return final_mask


# Verarbeitung
for img_file in clean_dir.glob("*.jpg"):
    img = cv2.imread(str(img_file))
    if img is None: continue
    h, w = img.shape[:2]
    for i in range(8):  # 8 Varianten pro Original
        soft_mask = generate_multiple_glare_masks(h, w, count=random.randint(1,4))
        glare_img, bin_mask = apply_glare(img, soft_mask)
        name = f"{img_file.stem}_glare_{i}.png"
        cv2.imwrite(str(output_img_dir / name), glare_img)
        cv2.imwrite(str(output_mask_dir / name), bin_mask)

