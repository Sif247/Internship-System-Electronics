import cv2
import numpy as np
import time
from picamera2 import Picamera2
import math


#funziona! trova tutti i cerchi!
# -----------------------
# PARAMETRI
# -----------------------

FRAME_WIDTH = 1200
FRAME_HEIGHT = 780

GRID_SCALE_X = 0.75
GRID_SCALE_Y = 0.75

BLUR_KERNEL = (9, 9)

CIRCLE_MIN_DIST = 60
CIRCLE_PARAM1 = 90
CIRCLE_PARAM2 = 26
CIRCLE_MIN_RADIUS = 30
CIRCLE_MAX_RADIUS = 38

FILL_THRESHOLD = 80

PROCESS_EVERY = 2
FRAME_DELAY = 0

TRAY_SCALE_FACTOR = 1.15
TRAY_RADIUS_MARGIN = 2.0

ROWS = 8
COLS = 8

# -----------------------
# KPI TEST
# -----------------------

MAX_TEST_FRAMES = 100

EXPECTED_TRAYS = 1
EXPECTED_HOLES = 64
EXPECTED_EMPTY = 63
EXPECTED_FILLED = 1

frame_counter = 0
processed_frames = 0

correct_tray = 0
correct_holes = 0
correct_empty = 0
correct_filled = 0
correct_all = 0

tray_errors = []
holes_errors = []
empty_errors = []
filled_errors = []

processing_times_ms = []

# -----------------------
# FUNZIONI
# -----------------------

def ordina_punti_matrice(lista_pos):

    if not lista_pos:
        return []

    punti = sorted(lista_pos, key=lambda p: (p[1], p[0]))

    righe = []
    riga_corrente = [punti[0]]

    for p in punti[1:]:

        if abs(p[1] - riga_corrente[0][1]) <= 10:
            riga_corrente.append(p)
        else:
            righe.append(riga_corrente)
            riga_corrente = [p]

    righe.append(riga_corrente)

    for i in range(len(righe)):
        righe[i] = sorted(righe[i], key=lambda p: p[0])

    ordinati = []
    for r in righe:
        ordinati.extend(r)

    return ordinati


def trova_rotazione_generale(lista_punti):

    if len(lista_punti) < 2:
        return 0.0

    punti = np.array(lista_punti, dtype=np.float64)

    centro = np.mean(punti, axis=0)
    punti_centrati = punti - centro

    cov = np.cov(punti_centrati.T)

    autovalori, autovettori = np.linalg.eig(cov)

    indice_max = np.argmax(autovalori)

    dx, dy = autovettori[:, indice_max]

    angolo = math.degrees(math.atan2(dy, dx))

    while angolo > 90:
        angolo -= 180

    while angolo < -90:
        angolo += 180

    return angolo


def ordina_griglia_reale(points):

    pts = np.array(points, dtype=np.float32)

    ang = trova_rotazione_generale(points)
    rad = math.radians(-ang)

    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])

    rot = []

    for x, y in pts:

        xr = cx + (x - cx) * math.cos(rad) - (y - cy) * math.sin(rad)
        yr = cy + (x - cx) * math.sin(rad) + (y - cy) * math.cos(rad)

        rot.append((xr, yr))

    rot = np.array(rot)

    idx = np.argsort(rot[:, 1])
    rot = rot[idx]
    pts = pts[idx]

    rows = []
    current = [0]

    for i in range(1, len(rot)):

        if abs(rot[i][1] - rot[current[0]][1]) < 35:
            current.append(i)
        else:
            rows.append(current)
            current = [i]

    rows.append(current)

    ordered = []

    for r in rows:

        row = pts[r]
        row = sorted(row, key=lambda p: p[0])

        ordered.extend(row)

    return ordered


def trova_vicino(p, detected, tol=40):

    for d in detected:

        dist = np.hypot(p[0] - d[0], p[1] - d[1])

        if dist < tol:
            return d

    return None


def safe_div(a, b):
    return a / b if b != 0 else 0.0


#funzioni per tray

def centro_da_vertici(vertices, output):
    if vertices is None:
        return None
    xs = [p[0] for p in vertices]
    ys = [p[1] for p in vertices]

    cx = int(sum(xs) / len(xs))
    cy = int(sum(ys) / len(ys))

    cv2.circle(output, (cx, cy), 10, (255, 255, 255), -1)
    return cx, cy


def remove_shadows(gray):
    gray = gray.astype(np.float32) + 1.0

    blur = cv2.GaussianBlur(gray, (0, 0), 80)

    retinex = np.log(gray) - np.log(blur)

    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)

    return retinex.astype(np.uint8)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def detect_tray(frame, output):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (13, 13), 1)

    edges = cv2.Canny(blur, 60, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 27))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("canny", closed)

    if len(contours) == 0:
        return None

    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < 100000:
        return None

    # hull elimina i denti del bordo
    hull = cv2.convexHull(largest)

    # approssima poligono
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    # se ha più di 4 punti, riduci
    if len(approx) > 4:

        rect = cv2.minAreaRect(hull)

        box = cv2.boxPoints(rect)

        box = np.int32(box)

        # disegna il rettangolo
        cv2.drawContours(frame, [box], 0, (0, 255, 255), 2)

        # diagonale 1
        cv2.line(output,
                 tuple(box[0]),
                 tuple(box[2]),
                 (255, 0, 0),
                 2)

        # diagonale 2
        cv2.line(output,
                 tuple(box[1]),
                 tuple(box[3]),
                 (0, 255, 0),
                 2)

        return box

    else:

        pts = approx.reshape(4, 2)

        cv2.drawContours(frame, [pts], 0, (0, 255, 255), 2)

        cv2.line(output,
                 tuple(pts[0]),
                 tuple(pts[2]),
                 (255, 0, 0),
                 2)

        cv2.line(output,
                 tuple(pts[1]),
                 tuple(pts[3]),
                 (0, 255, 0),
                 2)

        return pts


# -----------------------
# CAMERA
# -----------------------

picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"}
)

picam2.configure(config)
picam2.start()

# -----------------------
# LOOP PRINCIPALE
# -----------------------

try:

    while processed_frames < MAX_TEST_FRAMES:

        frame = picam2.capture_array()
        frame_counter += 1

        if frame_counter % PROCESS_EVERY != 0:
            continue

        start_time = time.perf_counter()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = frame.copy()

#detect tray

        tray = detect_tray(frame, output)
        centro = centro_da_vertici(tray, output)

        tray_found = 0


        if tray is not None:
            tray_found = 1

            cv2.drawContours(output, [tray], 0, (0, 255, 255), 3)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        norm = remove_shadows(gray)
        blur = cv2.GaussianBlur(norm, BLUR_KERNEL, 1.5)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(blur)

        # -----------------------
        # DETECT CERCHI
        # -----------------------

        circles = cv2.HoughCircles(
            eq,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=CIRCLE_MIN_DIST,
            param1=CIRCLE_PARAM1,
            param2=CIRCLE_PARAM2,
            minRadius=CIRCLE_MIN_RADIUS,
            maxRadius=CIRCLE_MAX_RADIUS
        )


        filled = 0
        empty = 0
        holes_found = 0
        ang = 0.0

        if circles is not None:

            circles = np.uint16(np.around(circles[0]))
            avg_r = np.mean([c[2] for c in circles])



            # -----------------------
            # ANALISI RIEMPIMENTO
            # -----------------------

            detected_points = []

            for x, y, r in circles:
                detected_points.append((int(x), int(y)))

            points = [(int(c[0]), int(c[1])) for c in circles]
            grid = ordina_griglia_reale(points)

            for gx, gy in grid:

                gx = int(gx)
                gy = int(gy)

                vicino = trova_vicino((gx, gy), detected_points)

                if vicino is not None:

                    x, y = int(vicino[0]), int(vicino[1])

                    mask = np.zeros(gray.shape, np.uint8)
                    cv2.circle(mask, (x, y), int(avg_r * 0.7), 255, -1)

                    mean_val = cv2.mean(gray, mask=mask)[0]

                    if mean_val < FILL_THRESHOLD:
                        empty += 1
                        color = (0, 0, 255)
                        letter = "E"
                    else:
                        filled += 1
                        color = (0, 200, 0)
                        letter = "F"

                    draw_x, draw_y = x, y

                else:
                    empty += 1
                    color = (0, 0, 255)
                    letter = "E"
                    draw_x, draw_y = gx, gy

                cv2.circle(output, (draw_x, draw_y), int(avg_r), color, 2)
                cv2.circle(output, (draw_x, draw_y), 3, (255, 255, 255), -1)

                cv2.putText(
                    output,
                    letter,
                    (draw_x - 8, draw_y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                holes_found = filled + empty

                # -----------------------
                # ROTAZIONE
                # -----------------------

                if len(detected_points) > 6:
                    ang = trova_rotazione_generale(detected_points)

                    cv2.rectangle(output, (840, 10), (1185, 70), (0, 0, 0), -1)

                    cv2.putText(
                        output,
                        f"ROTATION: {round(ang, 2)} deg",
                        (860, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )

        # -----------------------
        # KPI SU QUESTO FRAME
        # -----------------------

        tray_ok = (tray_found == EXPECTED_TRAYS)
        holes_ok = (holes_found == EXPECTED_HOLES)
        empty_ok = (empty == EXPECTED_EMPTY)
        filled_ok = (filled == EXPECTED_FILLED)

        if tray_ok:
            correct_tray += 1
        if holes_ok:
            correct_holes += 1
        if empty_ok:
            correct_empty += 1
        if filled_ok:
            correct_filled += 1
        if tray_ok and holes_ok and empty_ok and filled_ok:
            correct_all += 1

        tray_errors.append(abs(tray_found - EXPECTED_TRAYS))
        holes_errors.append(abs(holes_found - EXPECTED_HOLES))
        empty_errors.append(abs(empty - EXPECTED_EMPTY))
        filled_errors.append(abs(filled - EXPECTED_FILLED))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        processing_times_ms.append(elapsed_ms)

        processed_frames += 1

        # -----------------------
        # UI
        # -----------------------

        cv2.rectangle(output, (20, 20), (430, 120), (0, 0, 0), -1)

        cv2.putText(
            output,
            f"FILLED: {filled}",
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            f"EMPTY: {empty}",
            (180, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.putText(
            output,
            f"HOLES: {holes_found}",
            (320, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"FRAME TEST: {processed_frames}/{MAX_TEST_FRAMES}",
            (30, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow("result", output)

        key = cv2.waitKey(1)

        if key == 27:
            break

        time.sleep(FRAME_DELAY)

finally:

    picam2.stop()
    cv2.destroyAllWindows()

# -----------------------
# RISULTATI FINALI
# -----------------------

tray_detection_accuracy = safe_div(correct_tray, processed_frames)
hole_count_accuracy = safe_div(correct_holes, processed_frames)
empty_count_accuracy = safe_div(correct_empty, processed_frames)
filled_count_accuracy = safe_div(correct_filled, processed_frames)
full_frame_accuracy = safe_div(correct_all, processed_frames)

mean_tray_error = np.mean(tray_errors) if tray_errors else 0.0
mean_holes_error = np.mean(holes_errors) if holes_errors else 0.0
mean_empty_error = np.mean(empty_errors) if empty_errors else 0.0
mean_filled_error = np.mean(filled_errors) if filled_errors else 0.0

avg_time_ms = np.mean(processing_times_ms) if processing_times_ms else 0.0
total_time_s = np.sum(processing_times_ms) / 1000.0 if processing_times_ms else 0.0
throughput_fps = safe_div(processed_frames, total_time_s)
throughput_fpm = throughput_fps * 60.0

print("\n========== KPI SU 100 FRAME ==========")
print(f"Frame processati           = {processed_frames}")
print(f"Tray detection accuracy    = {tray_detection_accuracy:.4f}")
print(f"Hole count accuracy        = {hole_count_accuracy:.4f}")
print(f"Empty count accuracy       = {empty_count_accuracy:.4f}")
print(f"Filled count accuracy      = {filled_count_accuracy:.4f}")
print(f"Full frame accuracy        = {full_frame_accuracy:.4f}")
print()
print(f"Mean tray error            = {mean_tray_error:.2f}")
print(f"Mean holes error           = {mean_holes_error:.2f}")
print(f"Mean empty error           = {mean_empty_error:.2f}")
print(f"Mean filled error          = {mean_filled_error:.2f}")
print()
print(f"Average processing time    = {avg_time_ms:.2f} ms")
print(f"Throughput                 = {throughput_fps:.2f} frame/s")
print(f"Throughput                 = {throughput_fpm:.2f} frame/min")
print("======================================")

# -----------------------
# SCHERMATA FINALE RISULTATI
# -----------------------

summary = np.zeros((700, 1200, 3), dtype=np.uint8)

lines = [
    "KPI RESULTS ON 100 FRAMES",
    f"Processed frames: {processed_frames}",
    f"Tray detection accuracy: {tray_detection_accuracy:.4f}",
    f"Hole count accuracy: {hole_count_accuracy:.4f}",
    f"Empty count accuracy: {empty_count_accuracy:.4f}",
    f"Filled count accuracy: {filled_count_accuracy:.4f}",
    f"Full frame accuracy: {full_frame_accuracy:.4f}",
    "",
    f"Mean tray error: {mean_tray_error:.2f}",
    f"Mean holes error: {mean_holes_error:.2f}",
    f"Mean empty error: {mean_empty_error:.2f}",
    f"Mean filled error: {mean_filled_error:.2f}",
    "",
    f"Average processing time: {avg_time_ms:.2f} ms",
    f"Throughput: {throughput_fps:.2f} frame/s",
    f"Throughput: {throughput_fpm:.2f} frame/min",
    "",
    "Press any key to close"
]

y = 50
for line in lines:
    cv2.putText(
        summary,
        line,
        (40, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )
    y += 40

cv2.imshow("KPI Summary", summary)
cv2.waitKey(0)
cv2.destroyAllWindows()