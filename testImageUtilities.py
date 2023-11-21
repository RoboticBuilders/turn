from turn_utilities import *
from image_utilities import *

startAnglePictureFilename = "winning_robot_with_green_circle.jpg"
    
#while True:
#print("Capturing start picture")
#startPicture = capturePicture(startAnglePictureFilename)
#CircleCenterY = findCenterOfBlackGear(startAnglePictureFilename)

print("Dectecting Start angle")
_, width, height, _, angleOfTheFront = detectAngleOfRobotUsingImage(startAnglePictureFilename, showImages=True)
print("Detected Angle of the Front: " + str(angleOfTheFront))
print("press a key")
cv2.waitKey(0)
cv2.destroyAllWindows()
