import cv2 as cv
import numpy as np
from collections import Counter
import math

# --------------------------
# PARAMETRI (da calibrare)
# --------------------------

MIN_RADIUS = 50
MAX_RADIUS = 70
THRESHOLD_FILL = 125

# --------------------------
# Leggi immagine
# --------------------------

def rescale(image, scale=0.05):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    return cv.resize(image, dim)

def mostfrequent(vettore):
    conteggio = Counter(vettore)
    return conteggio.most_common(1)[0][0]

def count_near_circles(centri, centro_riferimento, raggio):
    count = 0
    x0, y0 = centro_riferimento
    x0 = int(x0)
    y0 = int(y0)

    r2 = float(raggio) ** 2

    for x, y in centri:
        x = int(x)
        y = int(y)

        dx = x - x0
        dy = y - y0
        d2 = dx*dx + dy*dy

        if dx == 0 and dy == 0:
            continue

        if d2 <= r2:
            count += 1

    return count





#img = rescale(cv.imread("../Photo/cestello.jpg"))
img = cv.imread("../Photo/front_tray.jpeg")
output = img.copy()
blank2=np.zeros((700,700), np.uint8)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gaus = cv.GaussianBlur(gray, (11, 11), 0)
#gray= cv.medianBlur(gray, 5)
cv.imshow("blur", gaus)
#tresh=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
#cv.imshow("Thresh", tresh)
#erode, = cv.erode(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (9,9)), iterations=1)

# --------------------------
# Trova cerchi
# --------------------------

circles = cv.HoughCircles(
    gaus,
    cv.HOUGH_GRADIENT,
    dp=1,
    minDist=100,
    param1=100,
    param2=30,
    minRadius=MIN_RADIUS,
    maxRadius=MAX_RADIUS
)

filled = 0
empty = 0
raggi=[]
centri=[]

if circles is not None:
    circles = np.uint16(np.around(circles))

    for (x, y, r) in circles[0,:]:

        mask = np.zeros(gaus.shape, dtype=np.uint8)
        cv.circle(mask, (x, y), r-10, 255, -1)

        mean_val = cv.mean(gaus, mask=mask)[0]
        print("Mean:", mean_val)


        #riempio i vettori con posizioni e raggi dei cerchi trovati
        raggi.append(r)
        centri.append((x,y))


        if mean_val > THRESHOLD_FILL:
            status = "FILLED"
            color = (0,255,0)
            filled += 1
        else:
            status = "EMPTY"
            color = (0,0,255)
            empty += 1

        cv.circle(output, (x, y), r, color, 3)
        cv.putText(output, status, (x-30, y),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2)


print(circles[0,:])

#ricerca dei bordi
bordi=[]
cont=0
r_medio = mostfrequent(raggi)
raggio_vicinanza = 2.2 * r_medio

for x, y in centri:
    n_vicini = count_near_circles(centri, (x, y), 170)
    print((x, y), "->", n_vicini)

#2,4 per bordi su e giu 2,3 per destra e sinitra , 2,3,4 per tutti i buchi bordosi
    if n_vicini in (2,3,4):
        bordi.append((x, y))
        cont += 1
        cv.circle(output, (x, y), r_medio, (255, 0, 0), 3)
print("numero di cerchi bordosi trovati:", cont)


# --------------------------
# Testo riassuntivo
# --------------------------

cv.putText(output,
            f"Filled: {filled}  Empty: {empty}",
            (30,50),
            cv.FONT_HERSHEY_SIMPLEX,
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
    display = cv.resize(output, (new_w, new_h))
else:
    display = output



cv.imshow("Result", display)
cv.waitKey(0)
cv.destroyAllWindows()