import cv2
import numpy as np
import math

def findCenterOfBlackGear(filename):
    image = readImage(filename)

    # It converts the BGR color space of image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Threshold of green in HSV space
    lower_blue = np.array([55, 60, 60])
    upper_blue = np.array([70, 255, 255])

    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions
    result = cv2.bitwise_and(image, image, mask = mask)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    #showImage("gray", gray)

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(image, contours, -1, (0, 255, 0), 20)
    #showImage("Contours", image)

    contour = contours[0]
    approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    #cv2.drawContours(image, [approx], 0, (0, 0, 0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    
    return x,y
 
    #cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    #cv2.imshow('frame', image)

    #cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    #cv2.imshow('mask', mask)

    #cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    #cv2.imshow('result', result)

def findLineEquation(startX, startY, endX, endY):
    if (endX == startX):
        A = 1
        B = 0
        C = -endX
    else:
        slope = (endY - startY) / (endX - startX)
        c = startY - (slope * startX)
        # In the form Ax+By+C = 0
        B = 1
        A = -1* slope
        C = -1 * c

    return A,B,C

# A,B,C is the hessian form of the line. 
# X, Y is the point. 
# Retuns the distance of Point from the line.
def findDistanceOfPointFromLine(A,B,C, X, Y):
    distance = abs((A * X)+ (B * Y) + C) / math.sqrt(A **2 + B ** 2)
    return distance

def findDistanceBetweenLineAndPoint(startX, startY, endX, endY, pointX, pointY):
    A,B,C = findLineEquation(startX,startY, endX, endY)
    distance = findDistanceOfPointFromLine(A, B, C, pointX, pointY)
    return distance

def findAngleOfLine(startX, startY, endX, endY):
    angle = 90
    if abs(startX - endX) > 0:
            angle = math.degrees(math.atan((startY - endY)/(startX - endX)))
    return angle

def capturePicture(filename):
    #print("Starting capture")
    camera = cv2.VideoCapture(cv2.CAP_DSHOW)
    #camera = cv2.VideoCapture(1)
    while camera.isOpened() == False:
        # Do nothing.
        pass
    return_value, image = camera.read()
    cv2.imwrite(filename, image)
    #print("captured picture in:" + filename)
    del(camera)

def detectContours(grayscale):
    contours, _ = cv2.findContours(grayscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #contours, _ = cv2.findContours(grayscale, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours = imutils.grab_contours(contours)
    #contours, _ = cv2.findContours(grayscale, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #print("Number of Contours found = " + str(len(contours)))
    return contours
    
# returns the rectangle that covers the robot.
# Also returns the Angle of the robot rectangle.
# This method first finds the robot rectangle, then it finds the longer edge of the 
# robot. This is assumed to the front or the back of the robot.
#
# Then the code searches for a green circle near the front of the robot and
# finds the edge of the rectangle that is closest to that green circle. 
# That is considered the front of the robot and the angle is returned 
# based on that edge being the front.
#
# The co-ordinate plane that this method uses is the robot facing the camera
# is zero, and the angle increase clockwise from 0-359.
def findRobotRectangle(contours, canny_output, output_window_name, filename):
     # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(contours)):        
        #print(str(cv2.contourArea(contours[i])))
        # Ignore all contours that are small, the robot is quite large.
        if (cv2.contourArea(contours[i]) > 1000):
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)

    # We expect that after all this processing we are left with just the robot.
    #assert(len(hull_list) == 1)

    if (len(hull_list) != 1):
        return None
    # Convert the hull that we found into min bounding rectangles.
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    color = (255, 255 , 255)
    hull = hull_list[0]
    #cv2.drawContours(drawing, hull_list, i, color)

    # Min rectangles cover the robot the rigth way
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    #Find the rectangle edges and find the longest edge.
    # First find distance between point 0 and point 1
    distance0and1 = math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1]-box[1][1]) ** 2)
    distance1and2 = math.sqrt((box[1][0] - box[2][0]) ** 2 + (box[1][1]-box[2][1]) ** 2)

    # Find the longer edge. This edge either the front of the rear of the robot.
    angleOfTheFront = 0
    longerEdgeIs0and1 = False
    if distance0and1 > distance1and2:
        longerEdgeIs0and1 = True

    #cv2.drawContours(drawing,[box],0,(0,0,255),2)    
    #cv2.namedWindow(output_window_name, cv2.WINDOW_NORMAL)
    #cv2.imshow(output_window_name, drawing)
    #cv2.waitKey(0)

    # Find the green circle. Determine which edge the green circle is closest to

    # Since the longer edge is between 0and1, the green circle should be between 0 and 1
    # or 2 and 3. First find which one the circle is closest to.
    # This would be the front of the robot, Once we know the front, then we can decide
    # on the angle or 180+angle.
    greenCircleX, greenCircleY = findCenterOfBlackGear(filename)
    angle = 0
    if longerEdgeIs0and1 == True:
        distanceofGreenCircleFrom0and1 = findDistanceBetweenLineAndPoint(box[0][0],box[0][1], box[1][0], box[1][1], greenCircleX, greenCircleY)
        distanceofGreenCircleFrom2and3 = findDistanceBetweenLineAndPoint(box[2][0],box[2][1], box[3][0], box[3][1], greenCircleX, greenCircleY)

        angle = findAngleOfLine(box[0][0], box[0][1], box[1][0],box[1][1])
        # This means that the 2-3 edge is the front of the robot.
        if distanceofGreenCircleFrom0and1 < distanceofGreenCircleFrom2and3:
            # Now determine     
            angle = angle + 180
    else:
        distanceofGreenCircleFrom1and2 = findDistanceBetweenLineAndPoint(box[1][0],box[1][1], box[2][0], box[2][1], greenCircleX, greenCircleY)
        distanceofGreenCircleFrom3and0 = findDistanceBetweenLineAndPoint(box[3][0],box[3][1], box[0][0], box[0][1], greenCircleX, greenCircleY)

        angle = findAngleOfLine(box[1][0], box[1][1], box[2][0],box[2][1])
        # This means that the 3-0 edge is the front of the robot.
        if distanceofGreenCircleFrom1and2 < distanceofGreenCircleFrom3and0:
            angle = angle + 180

    if angle < 0:
        angle = angle +360

    return rect, angle

def convertImageToGrayScale(image):
    # converting image into grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    '''
    # setting threshold of gray image
    _, threshold_img_ostu = cv2.threshold(gray,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    showImage("GrayScale OSTU", threshold_img_ostu)
    '''

    '''
    _, threshold_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    showImage("GrayScale Binary", threshold_img)
    '''
    
    blur = cv2.GaussianBlur(gray,(17,17),0)
    _,threshold_img_blur_ostu = cv2.threshold(blur,170,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #showImage("GrayScale OSTU BLUR", threshold_img_blur_ostu)
    threshold_img = threshold_img_blur_ostu
    
    # Use a kernel to erode and dilate the picture so we can capture the contours better.
    kernel = np.ones((10,10),np.uint8)
    threshold_img = cv2.erode(threshold_img, kernel, iterations=5)
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

# Takes a file with the picture of the robot taken from the top,
# against a background of white.
# returns the width, height and angle of the robot.
def detectAngleOfRobotUsingImage(filename):
    #capturePicture()
    #image = readImage("winning_robot_pos30.jpg")
    #image = readImage("winning_robot_neg30.jpg")
    #image = readImage("winning_robot_at_angle.jpg")
    #image = readImage("winning_robot.jpg")
    #image = readImage("robot_with_pointer_180.jpg")
    image = readImage(filename)

    grayImage = convertImageToGrayScale(image)
    # Find Canny edges
    cannyImage = cv2.Canny(grayImage, 30, 200)
    #showImage("Canny", cannyImage)
    contours = detectContours(cannyImage)
    #cv2.drawContours(image, contours, -1, (0, 255, 0), 20)
    #showImage("Contours", image)

    robotRectangle, angleOfTheFront = findRobotRectangle(contours, cannyImage, "FirstConvexHull", filename)
    #print("Width: " + str(robotRectangle[1][0]))
    #print("Height: " + str(robotRectangle[1][1]))
    #print("Angle of the robot: " + str(robotRectangle[2]))

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    width = 0
    height = 0
    angle = 0
    foundRobot = False
    if robotRectangle != None:
        width = robotRectangle[1][0]
        height = robotRectangle[1][1]
        angle = robotRectangle[2]
        foundRobot = True

    #boundingBoxes, robotBoundingBox = detectBoundingBoxes(contours)
    #drawBoundingBoxes(boundingBoxes, robotBoundingBox, cannyImage)

    return foundRobot, width, height, angle, angleOfTheFront






