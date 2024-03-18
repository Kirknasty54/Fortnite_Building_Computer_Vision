#import all needed modules, including custom created one
import cv2
import numpy as np
import pynput.keyboard

from stack import stackImages
import HandTrackingModule as htm
from pynput.mouse import Button, Controller
from pynput.keyboard import Key, Controller

#create window with specified width/height
frame_width = 640
frame_height = 480
#use 0 to use the laptops webcam, would need to be changed if you were wanting to use a different connected video input
cap = cv2.VideoCapture(0)
#initalize our HandDetector
detector = htm.HandDetector()
mouse = pynput.mouse.Controller()
keyboard = pynput.keyboard.Controller()

cap.set(3, frame_width)
cap.set(4, frame_height)

#this is simply a do nothing function to create our trackbars, likely going to comment these out since I've found a pretty good mixing to use for our purposes
def empty(a):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 44, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 255, 255, empty)


#this finds the contours of any images in the frame, this function will only occur if a hand is not detected
def getContours(img_dil, img_contour, img):
    shapes = []
    contours, hierarchy = cv2.findContours(img_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #draws contours for objects within a certain area, we will probably need to tweak this so it doesnt include whiteboard if
    # the whiteboards edges are in frames
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            cv2.drawContours(img, cnt, -1, (255, 0, 255), 7)
            cv2.drawContours(img_dil, cnt, -1, (255, 0, 255), 7)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            x, y, w, h = cv2.boundingRect(cnt)
            #use aspect ratio to determine the type of rectangle detected, be it horizontal or vertical
            aspect_ratio = float(w) / h
            if(len(approx) == 3):
                print('triangle found, l will need to be pressed')
                keyboard.press('v')
                #keyboard.press('f')
                #keyboard.release('f')
                keyboard.release('v')
                #mouse.click(Button.left, 1)
                #mouse.release(Button.left)
            elif(len(approx) == 4):
                if aspect_ratio > 1.5:
                    print('horizontal rectangle found, k')
                    keyboard.press('x')
                    #keyboard.press('f')
                    #keyboard.release('f')
                    keyboard.release('x')
                    #mouse.click(Button.left, 1)
                    #mouse.release(Button.left)
                elif aspect_ratio < 0.5:
                    print('vertical rectangle found, j')
                    keyboard.press('z')
                    #keyboard.press('f')
                    #keyboard.release('f')
                    keyboard.release('z')
                    #mouse.click(Button.left, 1)
                    #mouse.release(Button.left)
            #print(len(approx))

while True:
    #read in the camera data and do some processing/filtering to get more accurate results for the corresponding task
    #the shape detection and hand detection algorithms use different kinds of filtering, may try and figure out a way to combine these to see if it can become
    #more efficient
    success, img = cap.read()
    img_contour = img.copy()
    img_copy = img.copy()
    img_copy = detector.find_hands(img)
    landmark_list = detector.find_pos(img)
    cv2.imshow("Result", img)

    img_blur = cv2.GaussianBlur(img, (7, 7), 1)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    thresehold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    thresehold2 = cv2.getTrackbarPos("Threshold1", "Parameters")

    img_canny = cv2.Canny(img_gray, thresehold1, thresehold2)
    kernel = np.ones((5, 5))
    img_dil = cv2.dilate(img_canny, kernel, iterations=1)

    #if no landmarks are detected, then shape detection will occur, otherwise we continue with hand recognition and drawing the hands to indicate
    #shape detection hasn't started yet
    if len(landmark_list) == 0:
        getContours(img_dil, img_contour, img)
    #this img_stack variable was used for getting a better view on the contour processing of the image, its not really needed in this context, but its a nice thing to have regardless
    #img_stack = stackImages(0.8, ([img, img_gray, img_canny],
    #                            [img_dil, img_contour, img_contour]))

    cv2.imshow("Result", img)
    #input q to terminate program and close windows
    if cv2.waitKey(1) & 0xff == ord("q"):
        break