import cv2
import numpy as np
import time
from picamera2 import Picamera2
import math
# -----------------------
# PARAMETRI
# -----------------------

# Dimensione del frame della camera
FRAME_WIDTH = 1200
FRAME_HEIGHT = 780

# Parametri per il blur dei frame
BLUR_KERNEL = (7, 7)

# Parametri HoughCircles per rilevamento cerchi
CIRCLE_MIN_DIST = 40       # distanza minima tra cerchi
CIRCLE_PARAM1 = 110        # soglia superiore per Canny interna
CIRCLE_PARAM2 = 28         # soglia per l'accettazione del cerchio
CIRCLE_MIN_RADIUS = 20
CIRCLE_MAX_RADIUS = 30

# Soglia per distinguere buco pieno da vuoto
FILL_THRESHOLD = 65

# Controllo frequenza di elaborazione
PROCESS_EVERY = 2
FRAME_DELAY = 0

# Refresh tray completo ogni N frame
TRAY_REFRESH = 30

# Numero massimo di punti per il tracking
MAX_TRACK_POINTS = 50

# -----------------------
# STATO GLOBALE
# -----------------------

tray_roi = None       # Regione del tray (x,y,w,h)
tray_locked = False   # Indica se il tray è stato rilevato e lockato

prev_gray = None      # Frame precedente in scala di grigi (per tracking)
track_points = None   # Punti tracciati con optical flow

frame_counter = 0     # Contatore frame totali

# -----------------------
# INIZIALIZZAZIONE CAMERA
# -----------------------

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"}
)
picam2.configure(config)
picam2.start()


#funzioni

def ordina_punti_matrice(lista_pos, tolleranza_y=10):
    """
    Ordina una lista di punti (x, y) da sinistra a destra e dall'alto al basso.

    Parametri:
        lista_pos: lista di tuple (x, y)
        tolleranza_y: due punti con y vicine entro questa soglia
                      vengono considerati nella stessa riga

    Ritorna:
        lista ordinata
    """
    if not lista_pos:
        return []

    # 1) ordino grossolanamente per y e poi x
    punti = sorted(lista_pos, key=lambda p: (p[1], p[0]))

    righe = []
    riga_corrente = [punti[0]]

    for p in punti[1:]:
        # confronto la y col primo punto della riga corrente
        if abs(p[1] - riga_corrente[0][1]) <= tolleranza_y:
            riga_corrente.append(p)
        else:
            righe.append(riga_corrente)
            riga_corrente = [p]

    righe.append(riga_corrente)

    # 2) ordino ogni riga da sinistra a destra
    for i in range(len(righe)):
        righe[i] = sorted(righe[i], key=lambda p: p[0])

    # 3) appiattisco le righe in una sola lista
    ordinati = []
    for riga in righe:
        ordinati.extend(riga)

    return ordinati

def calcola_rotazione(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    dx = x2 - x1
    dy = y2 - y1

    angolo_rad = math.atan2(dy, dx)
    angolo_deg = math.degrees(angolo_rad)

    return angolo_deg




# -----------------------
# LOOP PRINCIPALE
# -----------------------

try:
    while True:

        # Cattura frame dalla camera
        frame = picam2.capture_array()
        frame_counter += 1

        # Processa solo 1 frame ogni PROCESS_EVERY per risparmiare CPU
        if frame_counter % PROCESS_EVERY != 0:
            continue
        
        #bgr->rgb
        frame=cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

        # Copia frame per output visuale
        output = frame.copy()

        # Converti frame in scala di grigi
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -----------------------
        # DETECT TRAY
        # -----------------------

        if not tray_locked or frame_counter % TRAY_REFRESH == 0:

            # Blur per ridurre rumore
            blur = cv2.GaussianBlur(gray, BLUR_KERNEL, 1.5)
            eq = cv2.equalizeHist(blur)

            # Rilevamento cerchi nel frame completo
            circles = cv2.HoughCircles(
                eq,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=CIRCLE_MIN_DIST,
                param1=CIRCLE_PARAM1,
                param2=CIRCLE_PARAM2,
                minRadius=CIRCLE_MIN_RADIUS,
                maxRadius=CIRCLE_MAX_RADIUS
            )

            if circles is not None:
                circles = np.uint16(np.around(circles[0]))
                # Se ci sono abbastanza cerchi, rileviamo tray
                if len(circles) >= 10:
                    xs = [int(c[0]) for c in circles]
                    ys = [int(c[1]) for c in circles]
                    rs = [int(c[2]) for c in circles]

                    min_x = min(xs)
                    max_x = max(xs)
                    min_y = min(ys)
                    max_y = max(ys)

                    avg_r = int(np.mean(rs))
                    margin = avg_r * 3

                    x = max(min_x - margin, 0)
                    y = max(min_y - margin, 0)
                    w = (max_x - min_x) + margin * 2
                    h = (max_y - min_y) + margin * 2

                    tray_roi = (x, y, w, h)
                    tray_locked = True

                    # Inizializzazione tracking con punti di interesse
                    roi_img = gray[y:y+h, x:x+w]
                    track_points = cv2.goodFeaturesToTrack(
                        roi_img,
                        maxCorners=MAX_TRACK_POINTS,
                        qualityLevel=0.01,
                        minDistance=10
                    )
                    if track_points is not None:
                        track_points[:, 0, 0] += x
                        track_points[:, 0, 1] += y

                    prev_gray = gray.copy()

        # -----------------------
        # TRACKING
        # -----------------------

        if tray_locked and track_points is not None and prev_gray is not None:

            # Calcolo optical flow dei punti tracciati
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                track_points,
                None
            )

            if new_pts is not None:
                good_new = new_pts[status == 1]
                good_old = track_points[status == 1]

                # Se ci sono pochi punti, resetta tracking
                if len(good_new) >= 5:
                    movement = good_new - good_old
                    dx = np.mean(movement[:, 0])
                    dy = np.mean(movement[:, 1])

                    # Aggiorna posizione tray
                    x, y, w, h = tray_roi
                    x += int(dx)
                    y += int(dy)
                    tray_roi = (x, y, w, h)

                    track_points = good_new.reshape(-1, 1, 2)

        prev_gray = gray.copy()

        # -----------------------
        # PROCESS TRAY
        # -----------------------

        if tray_locked:
            x, y, w, h = tray_roi

            # Limita ROI ai bordi dell'immagine
            img_h, img_w = gray.shape
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_w - x)
            h = min(h, img_h - y)

            if w > 0 and h > 0:
                tray_img = gray[y:y+h, x:x+w]

                # Rilevamento cerchi nel tray
                blur = cv2.GaussianBlur(tray_img, BLUR_KERNEL, 1.5)
                eq = cv2.equalizeHist(blur)
                circles = cv2.HoughCircles(
                    eq,
                    cv2.HOUGH_GRADIENT,
                    dp=1.2,
                    minDist=CIRCLE_MIN_DIST,
                    param1=CIRCLE_PARAM1,
                    param2=CIRCLE_PARAM2,
                    minRadius=CIRCLE_MIN_RADIUS,
                    maxRadius=CIRCLE_MAX_RADIUS
                )
                if circles is not None:
                    circles = np.uint16(np.around(circles[0]))

                # Analisi dei fori
                results = []
                results2 = []
                filled = 0
                empty = 0
                if circles is not None:
                    for cx, cy, r in circles:
                        mask = np.zeros(tray_img.shape, np.uint8)
                        cv2.circle(mask, (cx, cy), int(r*0.7), 255, -1)
                        mean = cv2.mean(tray_img, mask=mask)[0]

                        if mean < FILL_THRESHOLD:
                            status = "EMPTY"
                            empty += 1
                        else:
                            status = "FILLED"
                            filled += 1
                        results.append((cx, cy, r, status))
                        results2.append((cx, cy))
                
                results2=ordina_punti_matrice(results2,20)
                if len(results2) > 6:
                    pendenza=calcola_rotazione(results2[1],results2[6])
                    print(f"questi sono i punti che utilizzo per calcolare la pendenza -> {results2[1]} e {results2[6]}")
                    print(f"beccati la pendenza {pendenza}")
                    cv2.putText(
                        output,
                        f"Pendenza:{pendenza}",
                        (900, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2
                    )
                else:
                    cv2.putText(
                        output,
                        f"Pendenza:{0}",
                        (900, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2
                    )

                

                # Disegna cerchi e stati sul frame
                for cx, cy, r, status in results:
                    x_draw = int(cx + x)
                    y_draw = int(cy + y)
                    color = (0, 255, 0) if status == "FILLED" else (0, 0, 255)
                    cv2.circle(output, (x_draw, y_draw), int(r), color, 2)
                    cv2.circle(output, (x_draw, y_draw), 3, (255, 255, 255), -1)

                # Disegna rettangolo del tray e contatori
                cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 3)
                cv2.putText(
                    output,
                    f"Filled:{filled} Empty:{empty}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )


                # Calcola il centro del tray
                center_x = x + w // 2
                center_y = y + h // 2

                # Disegna il centro come cerchio piccolo
                cv2.circle(output, (center_x, center_y), 5, (255, 255, 0), -1)

                # Disegna le diagonali del tray (croce dagli angoli al centro)
                cv2.line(output, (x, y), (x + w, y + h), (255, 255, 0), 2)   # diagonale alto-sinistra → basso-destra
                cv2.line(output, (x + w, y), (x, y + h), (255, 255, 0), 2)   # diagonale alto-destra → basso-sinistra

        # Mostra frame finale
        cv2.imshow("result", output)

        # Gestione input tastiera
        key = cv2.waitKey(1)
        if key == 27:  # ESC per uscire
            break
        if key == ord('r'):  # R per resettare tray
            tray_locked = False

        time.sleep(FRAME_DELAY)

finally:
    # Pulizia
    picam2.stop()
    cv2.destroyAllWindows()