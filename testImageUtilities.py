from turn_utilities import *
from image_utilities import *

startAnglePictureFilename = "startangle_robot_picture.png"
    
#while True:
#print("Capturing start picture")
startPicture = capturePicture(startAnglePictureFilename)
#CircleCenterY = findCenterOfBlackGear(startAnglePictureFilename)

#print("Dectecting Start angle")
_, width, height, startActualAngle, angleOfTheFront = detectAngleOfRobotUsingImage(startAnglePictureFilename)
print("start angle: " + str(startActualAngle) + "   Angle of the Front: " + str(angleOfTheFront))
#input("press a key")