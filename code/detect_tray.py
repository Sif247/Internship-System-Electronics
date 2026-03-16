import cv2
import numpy as np
import time
from picamera2 import Picamera2
import math


#funziona!
#trova il tray e il suo centro!

# -----------------------
# PARAMETRI
# -----------------------

FRAME_WIDTH = 1200
FRAME_HEIGHT = 780

BLUR_KERNEL = (9, 9)

CIRCLE_MIN_DIST = 40
CIRCLE_PARAM1 = 90
CIRCLE_PARAM2 = 22
CIRCLE_MIN_RADIUS = 20
CIRCLE_MAX_RADIUS = 35

FILL_THRESHOLD = 100

PROCESS_EVERY = 2
FRAME_DELAY = 0

# dimensione tray rettificato
TRAY_WIDTH = 600
TRAY_HEIGHT = 400

frame_counter = 0


# -----------------------
# FUNZIONI
# -----------------------

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

    while True:

        frame = picam2.capture_array()
        frame_counter += 1

        if frame_counter % PROCESS_EVERY != 0:
            continue

        start_time = time.perf_counter()

        output = frame.copy()

        tray = detect_tray(frame, output)
        centro = centro_da_vertici(tray, output)

        tray_found = 0
        filled = 0
        empty = 0

        if tray is not None:

            tray_found = 1

            cv2.drawContours(output, [tray], 0, (0, 255, 255), 3)



        if cv2.waitKey(1) == 27:
            break

        time.sleep(FRAME_DELAY)

finally:

    picam2.stop()
    cv2.destroyAllWindows()