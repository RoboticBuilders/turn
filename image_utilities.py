import cv2
import numpy as np
import math
import time

def findCenterOfGreenCircle(filename, showImages=False):
    image = readImage(filename)

    # It converts the BGR color space of image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Threshold of green in HSV space
    lower_green = np.array([55, 60, 60])
    upper_green = np.array([70, 255, 255])

    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-green regions
    result = cv2.bitwise_and(image, image, mask = mask)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    #showImage("gray", gray)

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if (showImages == True):
        cv2.drawContours(image, contours, -1, (0, 255, 0), 20)
        showImage("Step 5: Detect GreenCircle Contours", image)

    # Approximate the contour using the Approximate Polygon method.
    contour = contours[0]
    approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    #cv2.drawContours(image, [approx], 0, (0, 0, 0), 5)
    
    # Return the center of the approximated polygon.
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    
    return x,y
 
    #cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    #cv2.imshow('frame', image)

    #cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    #cv2.imshow('mask', mask)
    
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
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #camera = cv2.VideoCapture(cv2.CAP_DSHOW)
    while camera.isOpened() == False:
        # Do nothing.
        pass
    time.sleep(1)
    _, image = camera.read()
    cv2.imwrite(filename, image)
    #print("captured picture in:" + filename)
    del(camera)

def detectContours(grayscale):
    contours, _ = cv2.findContours(grayscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
def findRobotRectangle(contours, canny_output, output_window_name, filename, showImages=False,
                       showLastImage=False):
     # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(contours)):        
        #print(str(cv2.contourArea(contours[i])))
        # Ignore all contours that are small, the robot is quite large.
        # We know Marvin is atleast 5000 pixels, so we ignore everything smaller.
        if (cv2.contourArea(contours[i]) > 5000):
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)

    # We expect that after all this processing we are left with just the robot.
    # If we dont find such a convex hull or we find more than one, then something really went
    # wrong and we do not process further, we declare this image processing run a 
    # failure and just return none, to indicate we could not find the robot rectangle
    if (len(hull_list) != 1):
        return None
    
    # Convert the hull that we found into min bounding rectangles.
    # This is used for Debugging.
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    hull = hull_list[0]

    # Min rectangles cover the robot the rigth way
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    if (showImages == True or showLastImage == True):
        cv2.drawContours(drawing,[box],0,(0,0,255),5)    
        #for i in range(4):
        #    cv2.putText(drawing, str(box[i][0]) + str(",") + str(box[i][1]), 
        #                org=(box[i][0],box[i][1]), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        #                thickness=1, fontScale=1, color=(255,255,255))
        cv2.namedWindow(output_window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(output_window_name, drawing)
        cv2.waitKey(0)

    # Find the green circle. Determine which edge the green circle is closest to

    # Since the longer edge is between 0and1, the green circle should be between 0 and 1
    # or 2 and 3. First find which one the circle is closest to.
    # This would be the front of the robot, Once we know the front, then we can decide
    # on the angle or 180+angle.
    greenCircleX, greenCircleY = findCenterOfGreenCircle(filename, showImages=False)
    angle = 0
    
    # Dictionary of key=distance, value=which edge. 
    # Edge0 = Edge between box[0] and box[1]
    # Edge1 = Edge between box[1] and box[2] etc.
    # We do this because it is easy to sort the dictionary and find which
    # edge we should use to find the angle of the robot.
    distancesFromGreenCircle = {}
    distanceofGreenCircleFrom0and1 = findDistanceBetweenLineAndPoint(box[0][0],box[0][1], 
                                                                     box[1][0], box[1][1], 
                                                                     greenCircleX, greenCircleY)
    distancesFromGreenCircle[distanceofGreenCircleFrom0and1] = "Edge0"
    distanceofGreenCircleFrom1and2 = findDistanceBetweenLineAndPoint(box[1][0],box[1][1], 
                                                                     box[2][0], box[2][1], 
                                                                     greenCircleX, greenCircleY)
    distancesFromGreenCircle[distanceofGreenCircleFrom1and2] = "Edge1"
    distanceofGreenCircleFrom2and3 = findDistanceBetweenLineAndPoint(box[2][0],box[2][1], 
                                                                     box[3][0], box[3][1], 
                                                                     greenCircleX, greenCircleY)
    distancesFromGreenCircle[distanceofGreenCircleFrom2and3] = "Edge2"
    distanceofGreenCircleFrom3and0 = findDistanceBetweenLineAndPoint(box[3][0],box[3][1], 
                                                                     box[0][0], box[0][1], 
                                                                     greenCircleX, greenCircleY)
    distancesFromGreenCircle[distanceofGreenCircleFrom3and0] = "Edge3"

    # Distances sorted, to find which edge is closest to the green circle.
    # Once we have that edge, we can then calculate the angle of the edge.
    sortedDistancesFromGreenCircle = dict(sorted(distancesFromGreenCircle.items()))

    # Get the first key in the dictionary. This is the edge that is closest to the green circle.
    closestEdge = None
    for key in sortedDistancesFromGreenCircle.values():
        closestEdge = key
        break

    #closestEdge = distancesFromGreenCircle.items()[0]
    if closestEdge == "Edge0":
        angle = findAngleOfLine(box[0][0], box[0][1], box[1][0],box[1][1])
    elif closestEdge == "Edge1":
        angle = findAngleOfLine(box[1][0], box[1][1], box[2][0],box[2][1])
    elif closestEdge == "Edge2":
        angle = findAngleOfLine(box[3][0], box[3][1], box[2][0],box[2][1])
    elif closestEdge == "Edge3":
        angle = findAngleOfLine(box[0][0], box[0][1], box[3][0],box[3][1])

    if (showImages == True or showLastImage == True):
        #cv2.drawContours(drawing,[box],0,(0,0,255),5)    
        cv2.putText(drawing, "ANGLE = " + str(angle), org=(120,120), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    thickness=2, fontScale=1, color=(0,0,255))
        #cv2.namedWindow("Final output angle", cv2.WINDOW_NORMAL)
        cv2.imshow(output_window_name, drawing)
        cv2.waitKey(0)
    else:
        print("angle = " + str(angle))

    return rect, angle


def convertImageToGrayScale(image, showImages):
    # converting image into grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if showImages == True:
        showImage("Step 0a: GrayScale", gray)

    # Smoothen the image using gaussian blue. 
    # We apply this to remove any noise in the white background, at the same time 
    # keeping the robot obvious and detectable.
    # The kernel needs to be a square with odd width and height
    blur = cv2.GaussianBlur(gray,(17,17),0)
    if showImages == True:
        showImage("Step 0b: Gaussian Blur", blur)

    # We then apply the threshod to the gaussian blurred image.
    # thresh = 170 and maxValue = 255. We use THRESH_BINARY_THRESH_OTSU Algorithm which
    # we found to be best after trial and error. 
    # OTSU uses a histogram of the gray scale image to find the best point to separate
    # Binary just compares with a given value.
    _,threshold_img_blur_ostu = cv2.threshold(blur,170,255,cv2.THRESH_OTSU)
    
    threshold_img = threshold_img_blur_ostu
    if showImages == True:
        showImage("Step 0c: Thresholded image", threshold_img)
     
    # Use a 10x10 kernel to erode and dilate the picture so we can capture the contours better.
    kernel = np.ones((10,10),np.uint8)
    threshold_img = cv2.erode(threshold_img, kernel, iterations=5)
    threshold_img = cv2.dilate(threshold_img, kernel, iterations=2)
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

# Takes a file with the picture of the robot taken from the top,
# against a background of white.
# returns the width, height and angle of the robot.
def detectAngleOfRobotUsingImage(filename, showImages=False, showLastImage=False):
    image = readImage(filename)
    if (showImages == True):
        showImage("Step 0: Input Image", image)
        cv2.waitKey(0)

    grayImage = convertImageToGrayScale(image, showImages=False)
    if (showImages == True):
        showImage("Step 1: GrayScale and Thresholded Image", grayImage)
        cv2.waitKey(0)

    # Find Canny edges. These values found after some trial and error.
    cannyImage = cv2.Canny(grayImage, 30, 200)
    if (showImages == True):
        showImage("Step 2: Canny on GrayScale Image", cannyImage)
        cv2.waitKey(0)

    # Next detect the contours.
    contours = detectContours(cannyImage)
    if (showImages == True):
        cv2.drawContours(image, contours, -1, (0, 255, 0), 20)
        showImage("Step 3: Contours on the CannyImage", image)
        cv2.waitKey(0)

    robotRectangle, angleOfTheFront = findRobotRectangle(contours, cannyImage, "Step 4: MinimumRectangleAroundConvexHull", 
                                                         filename, showImages=showImages,
                                                         showLastImage=showLastImage)
    #print("Width: " + str(robotRectangle[1][0]))
    #print("Height: " + str(robotRectangle[1][1]))
    #print("Angle of the robot: " + str(robotRectangle[2]))

    foundRobot = False
    if robotRectangle != None:
        foundRobot = True

    if showImages == True:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return foundRobot, angleOfTheFront
