#import all needed modules, including custom created one
import cv2
import numpy as np
from stack import stackImages
import HandTrackingModule as htm

#create window with specified width/height
frame_width = 640
frame_height = 480
#use 0 to use the laptops webcam, would need to be changed if you were wanting to use a different connected video input
cap = cv2.VideoCapture(0)
#initalize our HandDetector
detector = htm.HandDetector()
cap.set(3, frame_width)
cap.set(4, frame_height)

#this is simply a do nothing function to create our trackbars, likely going to comment these out since I've found a pretty good mixing to use for our purposes
def empty(a):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 44, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 255, 255, empty)


#this finds the contours of any images in the frame
def getContours(img, img_contour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #draws contours for objects within a certain area, we will probably need to tweak this so it doesnt include whiteboard if
    # the whiteboards edges are in frames
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            cv2.drawContours(img_contour, cnt, -1, (255, 0, 255), 7)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            print(len(approx))


while True:
    success, img = cap.read()
    img_contour = img.copy()
    img_copy = img.copy()
    img_copy = detector.find_hands(img)
    landmark_list = detector.find_pos(img)
    if len(landmark_list) != 0:
        cv2.imshow("Result", img)
        print(landmark_list[4])
    else:
        img_blur = cv2.GaussianBlur(img, (7, 7), 1)
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
        thresehold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
        thresehold2 = cv2.getTrackbarPos("Threshold1", "Parameters")

        img_canny = cv2.Canny(img_gray, thresehold1, thresehold2)
        kernel = np.ones((5, 5))
        img_dil = cv2.dilate(img_canny, kernel, iterations=1)

        getContours(img_dil, img_contour)
        img_stack = stackImages(0.8, ([img, img_gray, img_canny],
                                    [img_dil, img_contour, img_contour]))

        cv2.imshow("Result", img_stack)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break