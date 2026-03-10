import cv2
import numpy as np
from picamera2 import Picamera2

# -----------------------
# Parametri configurabili
# -----------------------

BLUR_KERNEL = (9,9)
MORPH_KERNEL_SIZE = 15

CIRCLE_MIN_DIST = 40
CIRCLE_PARAM1 = 110
CIRCLE_PARAM2 = 30
CIRCLE_MIN_RADIUS = 20
CIRCLE_MAX_RADIUS = 30

FILL_THRESHOLD = 200   # soglia luminosità pieno/vuoto

# kernel fisso
KERNEL = np.ones((MORPH_KERNEL_SIZE,MORPH_KERNEL_SIZE), np.uint8)

# -----------------------
# Funzioni
# -----------------------

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
    return blur

def get_tray_mask(thresh):
    h,w = thresh.shape
    mask_ff = np.zeros((h+2,w+2), np.uint8)
    flood = thresh.copy()
    cv2.floodFill(flood, mask_ff, (0,0), 255)
    flood_inv = cv2.bitwise_not(flood)
    tray_mask = thresh | flood_inv
    tray_mask = cv2.morphologyEx(tray_mask, cv2.MORPH_CLOSE, KERNEL)
    return tray_mask

def detect_tray(thresh_img):
    contours, _ = cv2.findContours(
        thresh_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def crop_tray(gray_img, contour, margin_ratio=0.05):
    x, y, w, h = cv2.boundingRect(contour)
    margin = int(margin_ratio * max(w, h))
    x = max(0, x + margin)
    y = max(0, y + margin)
    w = min(gray_img.shape[1] - x, w - 2 * margin)
    h = min(gray_img.shape[0] - y, h - 2 * margin)
    tray = gray_img[y:y+h, x:x+w]
    return tray, x, y

def preprocess_circles(tray_gray):
    tray_blur = cv2.GaussianBlur(tray_gray, (7,7), 1.5)
    tray_eq = cv2.equalizeHist(tray_blur)
    return tray_eq

def detect_circles(tray_eq):
    circles = cv2.HoughCircles(
        tray_eq,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=CIRCLE_MIN_DIST,
        param1=CIRCLE_PARAM1,
        param2=CIRCLE_PARAM2,
        minRadius=CIRCLE_MIN_RADIUS,
        maxRadius=CIRCLE_MAX_RADIUS
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
    return circles

def analyze_holes(tray_gray, circles):
    results = []
    filled = 0
    empty = 0
    if circles is None:
        return results, filled, empty
    for cx,cy,r in circles[0,:]:
        mask = np.zeros(tray_gray.shape, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(r*0.7), 255, -1)
        mean_val = cv2.mean(tray_gray, mask=mask)[0]
        if mean_val < FILL_THRESHOLD:
            status = "EMPTY"
            empty += 1
        else:
            status = "FILLED"
            filled += 1
        results.append((cx,cy,r,status))
    return results, filled, empty

def draw_circles(output_img, results, offset_x, offset_y):
    for cx,cy,r,status in results:
        cx_global = cx + offset_x
        cy_global = cy + offset_y
        color = (0,255,0) if status=="FILLED" else (0,0,255)
        cv2.circle(output_img, (cx_global, cy_global), r, color, 3)
        cv2.circle(output_img, (cx_global, cy_global), 3, (255,255,255), -1)
        cv2.putText(output_img, status, (cx_global-30, cy_global),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# -----------------------
# Setup Picamera2
# -----------------------

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1280, 720), "format": "BGR888"})
picam2.configure(config)
picam2.start()

# -----------------------
# Loop principale
# -----------------------

try:
    while True:
        frame = picam2.capture_array()
        output = frame.copy()

        # preprocessing
        blur = preprocess(frame)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # tray detection
        tray_mask = get_tray_mask(thresh)
        tray_contour = detect_tray(tray_mask)
        if tray_contour is None:
            cv2.imshow("result", output)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # crop tray
        tray_gray, offset_x, offset_y = crop_tray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            tray_contour
        )

        # crea maschera del tray crop
        tray_mask_crop = np.zeros(tray_gray.shape, dtype=np.uint8)
        contour_shifted = tray_contour.copy()
        contour_shifted -= [offset_x, offset_y]
        cv2.drawContours(tray_mask_crop, [contour_shifted], -1, 255, -1)
        tray_gray_masked = cv2.bitwise_and(tray_gray, tray_gray, mask=tray_mask_crop)

        # cerchi detection
        tray_eq = preprocess_circles(tray_gray_masked)
        circles = detect_circles(tray_eq)

        # analisi fori
        results, filled, empty = analyze_holes(tray_gray_masked, circles)

        # filtraggio cerchi fuori tray
        results = [(cx,cy,r,status) for cx,cy,r,status in results
                   if cv2.pointPolygonTest(tray_contour, (cx+offset_x, cy+offset_y), False) >= 0]

        # disegno risultati
        draw_circles(output, results, offset_x, offset_y)
        cv2.putText(output, f"Filled: {filled}  Empty: {empty}", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        # display
        cv2.imshow("result", output)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC per uscire
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()