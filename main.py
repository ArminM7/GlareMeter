import cv2
import numpy as np

# ============================================================
#  EINSTELLUNGEN  (nur hier musst du drehen)
# ============================================================
BILD_PFAD    = "glanz_output/reflex_0.png"   # <-- Pfad zu deinem Foto
V_MIN        = 249   # Helligkeit (0-255): ALLES darüber gilt als "hell"
S_MAX        = 30    # Sättigung  (0-255): ALLES darunter gilt als "farblos/weißlich"
MIN_FLAECHE  = 5000   # Flecken kleiner als das (in Pixeln) ignorieren -> Rauschen
LINIEN_DICKE = 2     # Dicke der roten Umrandung
# ============================================================


# ---- 1. Bild laden --------------------------------------------------
bild = cv2.imread(BILD_PFAD)
if bild is None:
    raise FileNotFoundError(f"Bild nicht gefunden: {BILD_PFAD}")


# ---- 2. Nach HSV umwandeln ------------------------------------------
# HSV = Farbton / Sättigung / Helligkeit. Damit lässt sich "hell UND
# farblos" viel leichter beschreiben als in RGB.
hsv = cv2.cvtColor(bild, cv2.COLOR_BGR2HSV)


# ---- 3. Maske: hell UND ungesättigt = Reflexion ---------------------
# Regel: V >= V_MIN  (hell)   und   S <= S_MAX  (farblos/weißlich)
# H (Farbton) ist egal -> voller Bereich 0..179
maske = cv2.inRange(hsv, (0, 0, V_MIN), (179, S_MAX, 255))


# ---- 4. Aufräumen ---------------------------------------------------
# kleine Störpunkte entfernen (OPEN) und Löcher in Flecken füllen (CLOSE)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN,  kernel)
maske = cv2.morphologyEx(maske, cv2.MORPH_CLOSE, kernel)


# ---- 5. Flecken finden ----------------------------------------------
konturen, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# ---- 6. Vermessen + ROT umranden ------------------------------------
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

    # Flächenwert an den Fleck schreiben (optional)
    x, y, w, h = cv2.boundingRect(kontur)
    cv2.putText(ergebnis, f"{int(flaeche)} px", (x, max(y - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1)


# ---- 7. Ergebnis ausgeben -------------------------------------------
bild_pixel = bild.shape[0] * bild.shape[1]
print(f"Gefundene Reflexionsflecken: {anzahl}")
print(f"Gesamte Reflexionsflaeche:   {int(gesamt_flaeche)} Pixel")
print(f"Anteil am Bild:              {100 * gesamt_flaeche / bild_pixel:.2f} %")


# ---- 8. Speichern + Anzeigen ----------------------------------------
cv2.imwrite("ergebnis.jpg", ergebnis)   # Bild mit roter Umrandung
cv2.imwrite("maske.jpg", maske)         # reine Schwarz-Weiss-Maske
print("Gespeichert: ergebnis.jpg (rot umrandet), maske.jpg")

# Fenster anzeigen -- falls du headless laufen laesst, diese 4 Zeilen auskommentieren
#cv2.imshow("Reflexionen (rot umrandet)", ergebnis)
#cv2.imshow("Maske", maske)
cv2.waitKey(0)
cv2.destroyAllWindows()