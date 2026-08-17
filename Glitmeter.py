import cv2
import numpy as np
from picamera2 import Picamera2

# ============================================================
#  EINSTELLUNGEN  (nur hier musst du drehen)
# ============================================================
V_MIN        = 249   # Helligkeit (0-255): ALLES darüber gilt als "hell"
S_MAX        = 30    # Sättigung  (0-255): ALLES darunter gilt als "farblos/weißlich"
MIN_FLAECHE  = 50  # Flecken kleiner als das (in Pixeln) ignorieren -> Rauschen
LINIEN_DICKE = 1     # Dicke der roten Umrandung
# ============================================================


def finde_reflexe(bild):
    """Nimmt ein BGR-Bild, gibt (ergebnis_bild, anzahl, gesamt_flaeche) zurück.
    Das ist exakt deine Offline-Logik von Schritt 2 bis 6."""

    # ---- Nach HSV umwandeln ----
    hsv = cv2.cvtColor(bild, cv2.COLOR_BGR2HSV)

    # ---- Maske: hell UND ungesättigt = Reflexion ----
    maske = cv2.inRange(hsv, (0, 0, V_MIN), (179, S_MAX, 255))

    # ---- Aufräumen ----
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN,  kernel)
    maske = cv2.morphologyEx(maske, cv2.MORPH_CLOSE, kernel)

    # ---- Flecken finden ----
    konturen, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ---- Vermessen + ROT umranden ----
    ergebnis = bild.copy()
    gesamt_flaeche = 0
    anzahl = 0

    for kontur in konturen:
        flaeche = cv2.contourArea(kontur)
        if flaeche < MIN_FLAECHE:
            continue                      # zu klein -> Rauschen, überspringen
        anzahl += 1
        gesamt_flaeche += flaeche

        # rote Linie um den Reflexionsbereich  (BGR: rot = (0, 0, 255))
        cv2.drawContours(ergebnis, [kontur], -1, (0, 0, 255), LINIEN_DICKE)

        # Flächenwert an den Fleck schreiben
        x, y, w, h = cv2.boundingRect(kontur)
        cv2.putText(ergebnis, f"{int(flaeche)} px", (x, max(y - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return ergebnis, anzahl, gesamt_flaeche


# ---- Kamera starten -------------------------------------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

print("Kamera läuft. Fenster fokussieren und 'q' zum Beenden drücken.")

# ---- Live-Schleife --------------------------------------------------
while True:
    # ein Bild von der Kamera holen (kommt als RGB-Array)
    frame = picam2.capture_array()

    # Picamera2 liefert RGB, OpenCV rechnet in BGR -> umdrehen
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # deine Erkennung drüberlaufen lassen
    ergebnis, anzahl, gesamt_flaeche = finde_reflexe(frame)

    # Gesamtwerte oben ins Bild schreiben
    bild_pixel = frame.shape[0] * frame.shape[1]
    anteil = 100 * gesamt_flaeche / bild_pixel
    text = f"Flecken: {anzahl}   Flaeche: {int(gesamt_flaeche)} px   ({anteil:.1f} %)"
    cv2.putText(ergebnis, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # anzeigen
    cv2.imshow("Reflexionen live (rot umrandet)", ergebnis)

    # 'q' beendet die Schleife
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---- Aufräumen ------------------------------------------------------
picam2.stop()
cv2.destroyAllWindows()