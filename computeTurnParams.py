from turn_utilities import *
from image_utilities import *

# 1. Takes a picture of the robot first
# 2. Computes the angle that the robot is at.
# 3. Creates and run the program for the anglet that we want to turn.
# 4. Captures the voltage and the gyro readings of the program
# 5. Takes the end picture
# 6. computes the end actual angle.
# 7. returns all of this as output.
def runOneTest(angleToTest, speedToTest):
    startAnglePictureFilename = "startangle_robot_picture.png"
    endAnglePictureFilename = "endangle_robot_picture.png"

    print("Capturing start picture")
    startPicture = capturePicture(startAnglePictureFilename)

    print("Dectecting Start angle")
    width, height, startActualAngle = detectAngleOfRobotUsingImage(startAnglePictureFilename)
    print("start angle: " + str(startActualAngle))

    programFilename = "testProgram.py"
    print("Writing a turn program and saving to: " + programFilename)
    createTurnProgram(targetAngle=angleToTest, speed=speedToTest, filename=programFilename)

    print("Pushing " + programFilename + " to the hub")
    output = uploadProgramToHub(filename=programFilename)
    voltage, startAngle, endAngle = extractParamsFromProgramOutput(output)
    print("Extracted params from run: voltage:" + str(voltage) + " startAngle:" +str(startAngle) + " endAngle:" + str(endAngle))

    print("Capturing end picture")
    endPicture = capturePicture(endAnglePictureFilename)

    print("Dectecting end angle")
    width, height, endActualAngle = detectAngleOfRobotUsingImage(startAnglePictureFilename)
    print("end angle: " + str(endActualAngle))

    outputValues = {}
    outputValues["angle"] = angleToTest
    outputValues["speed"] = speedToTest
    outputValues["voltage"] = voltage
    outputValues["startAngle"] = startAngle
    outputValues["endAngle"] = endAngle
    outputValues["startActualAngle"] = startActualAngle
    outputValues["endActualAngle"] = endActualAngle

    return outputValues

output = runOneTest(angleToTest=90, speedToTest=30)
print(output)
