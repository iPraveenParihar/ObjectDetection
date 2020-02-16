import cv2 as cv
import numpy as np 

src_img = cv.imread('android.png')
src_img = cv.resize(src_img, (960,540))

gray_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)

blur_img = cv.bilateralFilter(gray_img, 11, 17, 17)

#blur_img = cv.GaussianBlur(gray_img, (17,17), cv.BORDER_DEFAULT)

canny_edges = cv.Canny(blur_img, 100,200)

#Morphological transformations
#To get kernel of desired size and shape
#shapes : MORPH_RECT, MORPH_ELLIPSE, MORPH_CROSS
kernel = cv.getStructuringElement(cv.MORPH_RECT, (7,7))

#Closing transformation , dilation followed by erosion
closing = cv.morphologyEx(canny_edges, cv.MORPH_CLOSE, kernel)

#cv.imshow("Closed", closing)
#cv.imshow("Smooting", np.hstack((gray_img,blur_img, canny_edges)))
#cv.waitKey(0)

file = 0
contours, heirarcy = cv.findContours(closing.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#To detect each object in the contour
for c in contours:
	perimeter = cv.arcLength(c, True)
	approx = cv.approxPolyDP(c, 0.02 * perimeter, True)
	#cv.drawContours(src_img, [approx], -1, (0,255,0), 2)
	x,y,w,h = cv.boundingRect(c)
	file += 1
	obj_detect = src_img[y:y+h, x:x+w]
	cv.imwrite(str(file)+'.jpg', obj_detect)

cv.drawContours(src_img, contours, -1, (0,255,0),2)
cv.imshow("detected", src_img)
cv.waitKey(0)
cv.destroyAllWindows()
