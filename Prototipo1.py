import cv2
import numpy as np

# --------------------------
# PARAMETRI (da calibrare)
# --------------------------

MIN_RADIUS = 30
MAX_RADIUS = 100
THRESHOLD_FILL = 125

# --------------------------
# Leggi immagine
# --------------------------

img = cv2.imread("OpenCV/Preprocessing/images/test/test1.jpeg")
output = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (11, 11), 0)
#erode, = cv.erode(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (9,9)), iterations=1)

# --------------------------
# Trova cerchi
# --------------------------

circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=100,
    param1=100,
    param2=30,
    minRadius=MIN_RADIUS,
    maxRadius=MAX_RADIUS
)

filled = 0
empty = 0

if circles is not None:
    circles = np.uint16(np.around(circles))

    for (x, y, r) in circles[0,:]:

        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r-10, 255, -1)

        mean_val = cv2.mean(gray, mask=mask)[0]
        print("Mean:", mean_val)

        if mean_val > THRESHOLD_FILL:
            status = "FILLED"
            color = (0,255,0)
            filled += 1
        else:
            status = "EMPTY"
            color = (0,0,255)
            empty += 1

        cv2.circle(output, (x, y), r, color, 3)
        cv2.putText(output, status, (x-30, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2)

# --------------------------
# Testo riassuntivo
# --------------------------

cv2.putText(output,
            f"Filled: {filled}  Empty: {empty}",
            (30,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2)

# --------------------------
# 🔥 RIDIMENSIONAMENTO SOLO PER DISPLAY
# --------------------------

max_width = 1000  # larghezza massima finestra

h, w = output.shape[:2]

if w > max_width:
    scale = max_width / (w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    display = cv2.resize(output, (new_w, new_h))
else:
    display = output

cv2.imshow("Result", display)
cv2.waitKey(0)
cv2.destroyAllWindows()