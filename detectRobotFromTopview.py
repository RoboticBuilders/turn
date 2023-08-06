import cv2
import numpy as np
import imutils

'''
print("Starting capture")
camera = cv2.VideoCapture(0)
return_value, image = camera.read()
cv2.imwrite('opencv_captured_file.png', image)
print("return value: " + str(return_value))
del(camera)
'''

def detectContours(grayscale):
    contours, _ = cv2.findContours(grayscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #contours, _ = cv2.findContours(grayscale, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours = imutils.grab_contours(contours)
    #contours, _ = cv2.findContours(grayscale, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print("Number of Contours found = " + str(len(contours)))
    return contours
    
# returns the rectangle that covers the robot.
def findRobotRectangle(contours, canny_output, output_window_name):
     # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(contours)):        
        #print(str(cv2.contourArea(contours[i])))
        # Ignore all contours that are small, the robot is quite large.
        if (cv2.contourArea(contours[i]) > 1000):
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)

    # We expect that after all this processing we are left with just the robot.
    assert(len(hull_list) == 1)
            
    # Convert the hull that we found into min bounding rectangles.
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    color = (255, 255 , 255)
    for i in range(len(hull_list)):
        #cv2.drawContours(drawing, hull_list, i, color)

        # Min rectangles cover the robot the rigth way
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(drawing,[box],0,(0,0,255),2)
    
        
    cv2.namedWindow(output_window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(output_window_name, drawing)
    return rect

def convertImageToGrayScale(image):
    # converting image into grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    '''
    # setting threshold of gray image
    _, threshold_img_ostu = cv2.threshold(gray,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    showImage("GrayScale OSTU", threshold_img_ostu)

    _, threshold_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    showImage("GrayScale Binary", threshold_img)
    '''

    blur = cv2.GaussianBlur(gray,(17,17),0)
    _,threshold_img_blur_ostu = cv2.threshold(blur,170,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #showImage("GrayScale OSTU BLUR", threshold_img_blur_ostu)
    
    threshold_img = threshold_img_blur_ostu
    
    # Use a kernel to erode and dilate the picture so we can capture the contours better.
    kernel = np.ones((50,50),np.uint8)
    threshold_img = cv2.erode(threshold_img, kernel, iterations=2)
    threshold_img = cv2.dilate(threshold_img, kernel, iterations=2)
    #threshold_img = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,21,10)
    #showImage("GrayScale", threshold_img)
    return threshold_img

# readimage and convert into gray scale.
def readImage(filename):
    # reading image
    img = cv2.imread(filename)
    if img is None:
        print("Could not load the image")
        return None
    #showImage("Original", img)
    return img

def showImage(windowName, image):
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.imshow(windowName, image)

def captureVideo():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
    cam.release()

    cv2.destroyAllWindows()

# This method finds all the bounding boxes around the various robot parts 
# It also determines the bounding box around the entire robot by finding the
# minx,miny, maxx, maxy points.
#
# Returns
# BoundingBoxes, RobotBoundingBox
#
# RobotBounding box is an array of [minx, miny, maxx, maxy]
def detectBoundingBoxes(contours):
    # Approximate contours to polygons + get bounding rects and circles
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)

    minX = 0
    minY = 0
    maxX = 0
    maxY = 0
    for i, c in enumerate(contours):
        boundRect[i] = cv2.boundingRect(c)
        if i == 0:
            minX = boundRect[i][0]
            minY = boundRect[i][1]
            maxX = boundRect[i][0]
            maxY = boundRect[i][1]

        if boundRect[i][0] < minX:
            minX = boundRect[i][0]
        if boundRect[i][1] < minY:
            minY = boundRect[i][1]

        if boundRect[i][0] > maxX:
            maxX = boundRect[i][0]
        if boundRect[i][1] > maxY:
            maxY = boundRect[i][1]

        #contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        #boundRect[i] = cv2.boundingRect(contours_poly[i])
        #centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    robotBoundingBox = [minX, minY, maxX, maxY]
    return boundRect, robotBoundingBox

# Draws all the boundingBoxes, it also takes as input a robotBoundingbox
# The robotBoundingBox is represented as [minX,minY, maxX, maxY]
def drawBoundingBoxes(boundingBoxes, robotBoundingBox, canny_output):
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    color = (0,0,255)
    for boundingBox in boundingBoxes:
        #cv2.rectangle(drawing, boundingBox, color, 10)
        if (boundingBox[2] > 50 and boundingBox[3] > 50):
            cv2.rectangle(drawing, 
                        (int(boundingBox[0]), int(boundingBox[1])),
                        (int(boundingBox[0]+boundingBox[2]), int(boundingBox[1]+boundingBox[3])), 
                        color, 20)

    # Now draw the robot bounding box around the robot.
    cv2.rectangle(drawing, (robotBoundingBox[0], robotBoundingBox[1]), (robotBoundingBox[2], robotBoundingBox[3]), (255,0,0), 20)        
    cv2.namedWindow("BoundingBoxes", cv2.WINDOW_NORMAL)
    cv2.imshow('BoundingBoxes', drawing)

#image = readImage("winning_robot_pos30.jpg")
#image = readImage("winning_robot_neg30.jpg")
#image = readImage("winning_robot_at_angle.jpg")
#image = readImage("winning_robot.jpg")
image = readImage("robot_with_pointer_180.jpg")


grayImage = convertImageToGrayScale(image)
# Find Canny edges
cannyImage = cv2.Canny(grayImage, 30, 200)
#showImage("Canny", cannyImage)
contours = detectContours(cannyImage)
cv2.drawContours(image, contours, -1, (0, 255, 0), 20)
#showImage("Contours", image)

robotRectangle = findRobotRectangle(contours, cannyImage, "FirstConvexHull")
print("Width: " + str(robotRectangle[1][0]))
print("Height: " + str(robotRectangle[1][1]))
print("Angle of the robot: " + str(robotRectangle[2]))


#boundingBoxes, robotBoundingBox = detectBoundingBoxes(contours)
#drawBoundingBoxes(boundingBoxes, robotBoundingBox, cannyImage)

cv2.waitKey(0)
cv2.destroyAllWindows()





