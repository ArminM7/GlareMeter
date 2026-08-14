"""
Glanzflecken-Generator für SmartLens-Tests
------------------------------------------
Nimmt EIN festes Basisbild und legt 1-3 kreisrunde/elliptische Glanzflecke
darauf (weißer Kern 100% -> 0% am Rand, weicher Gradient).
Erzeugt daraus 5 Bilder, in denen die Flecke pro Schritt um +10% wachsen.

Was du anpassen musst:  nur BASISBILD_PFAD unten.
Ausgabe:                ordner "glanz_output" mit reflex_0.png ... reflex_4.png
"""

import os
import numpy as np
from PIL import Image

# ======================= EINSTELLUNGEN =======================
BASISBILD_PFAD = "D:/GitLab/GlareMeter0/original_Bilder/IMG_3134.JPG"   # <-- HIER deinen Pfad eintragen
AUSGABE_ORDNER = "glanz_output"

ANZAHL_BILDER   = 10        # wie viele Bilder in der Wachstumsreihe
WACHSTUM        = 0.20     # +10% Größe pro Bild
MIN_FLECKEN     = 1        # zufällig zwischen ...
MAX_FLECKEN     = 3        # ... so vielen Flecken

# Startgröße der Flecke als Anteil der Bild-Kurzseite (min/max Radius)
START_RADIUS_MIN = 0.10    # 4% der kurzen Bildseite
START_RADIUS_MAX = 0.20    # 9%

# Wie stark der Fleck maximal aufhellt (1.0 = reinweiß im Kern)
KERN_STAERKE = 1.5

# "Mitte des Bildes": Flecken-Zentren liegen in diesem zentralen Anteil
# 0.5 = Zentren nur im mittleren 50% des Bildes (nicht am Rand)
MITTEN_BEREICH = 0.5

ZUFALLS_SEED = None        # z.B. 42 für reproduzierbar, None = jedes Mal neu
# =============================================================


def erzeuge_fleck_maske(h, w, cx, cy, rx, ry):
    """
    Erzeugt eine Graustufen-Maske (0..1) für EINEN Fleck.
    Kern = 1.0 (voll), Rand = 0.0 (verschwindet), weicher Gradient dazwischen.
    Elliptisch über rx (Radius x) und ry (Radius y).
    """
    # Koordinatengitter
    ys, xs = np.mgrid[0:h, 0:w]
    # normierter Abstand vom Zentrum (elliptisch): 0 im Kern, 1 am Rand, >1 außerhalb
    dist = np.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)
    # Gradient: innen 1, am Rand 0. Alles außerhalb (dist>1) auf 0 clippen.
    maske = np.clip(1.0 - dist, 0.0, 1.0)
    # weicher machen (quadratisch fällt schöner ab als linear)
    maske = maske ** 1.5
    return maske


def main():
    if ZUFALLS_SEED is not None:
        np.random.seed(ZUFALLS_SEED)

    # --- Basisbild laden ---
    if not os.path.isfile(BASISBILD_PFAD):
        raise FileNotFoundError(
            f"Basisbild nicht gefunden: {BASISBILD_PFAD}\n"
            f"Bitte BASISBILD_PFAD oben im Skript anpassen."
        )
    basis = Image.open(BASISBILD_PFAD).convert("RGB")
    basis_np = np.asarray(basis).astype(np.float32)   # H x W x 3, Werte 0..255
    h, w = basis_np.shape[:2]
    kurzseite = min(h, w)

    # --- Flecken EINMAL festlegen (Position, Startgröße, Form) ---
    n = np.random.randint(MIN_FLECKEN, MAX_FLECKEN + 1)

    # zentraler Bereich, in dem die Zentren liegen dürfen
    rand_x = int(w * (1 - MITTEN_BEREICH) / 2)
    rand_y = int(h * (1 - MITTEN_BEREICH) / 2)

    flecken = []
    for _ in range(n):
        cx = np.random.randint(rand_x, w - rand_x)
        cy = np.random.randint(rand_y, h - rand_y)
        # Startradius
        r_base = np.random.uniform(START_RADIUS_MIN, START_RADIUS_MAX) * kurzseite
        # leichte Ellipsen-Verzerrung, damit nicht alle perfekt rund sind
        rx = r_base * np.random.uniform(0.85, 1.15)
        ry = r_base * np.random.uniform(0.85, 1.15)
        flecken.append({"cx": cx, "cy": cy, "rx": rx, "ry": ry})

    print(f"Bildgröße: {w}x{h}  |  {n} Fleck(en) erzeugt")

    # --- Ausgabeordner ---
    os.makedirs(AUSGABE_ORDNER, exist_ok=True)

    # --- Wachstumsreihe erzeugen ---
    for i in range(ANZAHL_BILDER):
        skala = (1.0 + WACHSTUM) ** i       # Bild 0 = 1.0, Bild 1 = 1.1, Bild 2 = 1.21, ...

        # Gesamt-Glanzmaske (Flecke überlagern sich additiv, dann clippen)
        glanz = np.zeros((h, w), dtype=np.float32)
        for f in flecken:
            m = erzeuge_fleck_maske(
                h, w, f["cx"], f["cy"],
                f["rx"] * skala, f["ry"] * skala
            )
            glanz = np.maximum(glanz, m)    # max = Flecke "verschmelzen" sauber statt zu übersättigen

        glanz = np.clip(glanz, 0.0, 1.0) * KERN_STAERKE

        # Glanz auf das Basisbild legen: Richtung Weiß (255) interpolieren
        # ergebnis = basis*(1-glanz) + 255*glanz   -> im Kern reinweiß, am Rand Originalbild
        glanz3 = glanz[:, :, None]          # H x W x 1 für Broadcasting
        ergebnis = basis_np * (1 - glanz3) + 255.0 * glanz3
        ergebnis = np.clip(ergebnis, 0, 255).astype(np.uint8)

        out_pfad = os.path.join(AUSGABE_ORDNER, f"reflex_{i}.png")
        Image.fromarray(ergebnis).save(out_pfad)
        # kurze Info: wie groß ist die "Glanzfläche" (Pixel mit >50% Glanz)
        flaeche = float((glanz > 0.5).mean() * 100)
        print(f"  reflex_{i}.png  |  Skala {skala:.2f}  |  Glanzfläche ~{flaeche:.1f}%")

    print(f"\nFertig. Bilder liegen in: {os.path.abspath(AUSGABE_ORDNER)}")


if __name__ == "__main__":
    main()