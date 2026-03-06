import cv2
import numpy as np

img = cv2.imread("OpenCV/Preprocessing/images/test/test1.jpeg")
output = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(7,7),0)

# -----------------------
# threshold
# -----------------------

_,thresh = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# -----------------------
# floodfill per riempire buchi
# -----------------------

h,w = thresh.shape
mask = np.zeros((h+2,w+2),np.uint8)

flood = thresh.copy()
cv2.floodFill(flood,mask,(0,0),255)

flood_inv = cv2.bitwise_not(flood)
tray_mask = thresh | flood_inv

# -----------------------
# chiude bordo dentato
# -----------------------

kernel = np.ones((25,25),np.uint8)

tray_mask = cv2.morphologyEx(
    tray_mask,
    cv2.MORPH_CLOSE,
    kernel
)

# -----------------------
# trova contorno tray
# -----------------------

contours,_ = cv2.findContours(
    tray_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

largest = max(contours, key=cv2.contourArea)

rect = cv2.minAreaRect(largest)
box = cv2.boxPoints(rect)
box = np.intp(box)

cv2.drawContours(output,[box],0,(0,255,0),4)
blank = np.zeros_like(gray)
mask = cv2.rectangle(blank, (box[0][0], box[0][1]), (box[2][0], box[2][1]), 255, -1)
tray_mask = cv2.bitwise_and(img, img, mask=mask)


scale = 0.4

cv2.imshow("threshold", cv2.resize(thresh,None,fx=scale,fy=scale))
cv2.imshow("tray mask", cv2.resize(tray_mask,None,fx=scale,fy=scale))
cv2.imshow("result", cv2.resize(output,None,fx=scale,fy=scale))
cv2.imshow("mask    ", cv2.resize(tray_mask,None,fx=scale,fy=scale))

cv2.waitKey(0)
cv2.destroyAllWindows()