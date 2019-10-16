import cv2
import numpy as np

path = 'super/'
path2 = 'bg_super/'

img2 = cv2.imread('p.png')
img1 = cv2.imread(path2+'1.png')

img1 = cv2.resize(img1, (500,255))
rows, cols, channels = img2.shape


roi = img1[0:rows, 0:cols ]

img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#ret,mask = cv2.threshold(img2gray, 100,255,cv2.THRESH_BINARY)

ret, mask = cv2.threshold(img2gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
mask_inv = cv2.bitwise_not(mask)

img1_bg = cv2.bitwise_not(mask)



img1_bg = cv2.bitwise_and(roi,roi, mask=mask_inv)

img2_fg = cv2.bitwise_and(img2, img2, mask = mask)
dst = cv2.add(img1_bg, img2_fg)
img1[0:rows, 0:cols]  = dst

cv2.imshow('res', img1)



#dst = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)





#img = np.concatenate((img1,img2), axis = 0)

#cv2.imshow('prueba', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

