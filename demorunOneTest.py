from turn_utilities import *
from image_utilities import *
import random
import datetime
import csv

# 1. Takes a picture of the robot first
# 2. Computes the angle that the robot is at.
# 3. Creates and run the program for the angle that we want to turn.
# 4. Captures the voltage and the gyro readings of the program
# 5. Takes the end picture
# 6. computes the end actual angle.
# 7. returns all of this as output.
def runOneTest(angleToTest, speedToTest, turn_deceleration, correction):
    startAnglePictureFilename = "startangle_robot_picture.png"
    endAnglePictureFilename = "endangle_robot_picture.png"

    print("Capturing start picture")
    startPicture = capturePicture(startAnglePictureFilename)


    print("Dectecting Start angle")
    startFoundRobot, width, height, angle, startActualAngle = detectAngleOfRobotUsingImage("demo_start_picture.jpg", 
                                                                                           showImages=True)
    #startFoundRobot, width, height, angle, startActualAngle = detectAngleOfRobotUsingImage(startAnglePictureFilename, 
    #                                                                                       showImages=False)
    print("start angle: " + str(startActualAngle))

    programFilename = "testProgram.py"
    print("Writing a turn program and saving to: " + programFilename)
    createTurnProgram(targetAngle=angleToTest, speed=speedToTest, turn_deceleration=turn_deceleration, 
                      correction=correction, filename=programFilename)

    print("Pushing " + programFilename + " to the hub")
    output = uploadProgramToHub(filename=programFilename)
    voltage, startAngle, endAngle = extractParamsFromProgramOutput(output)
    print("Extracted params from run: voltage:" + str(voltage) + " startAngle:" +str(startAngle) + " endAngle:" + str(endAngle))

    print("Capturing end picture")
    endPicture = capturePicture(endAnglePictureFilename)

    print("Dectecting end angle")
    endFoundRobot, width, height, angle, endActualAngle = detectAngleOfRobotUsingImage("demo_end_picture.jpg",
                                                                                       showLastImage = True)
    #endFoundRobot, width, height, angle, endActualAngle = detectAngleOfRobotUsingImage(endAnglePictureFilename,
    #                                                                                   showLastImage = False)
    print("end angle: " + str(endActualAngle))

    outputValues = {}
    outputValues["StartFoundRobot"] = startFoundRobot
    outputValues["EndFoundRobot"] = endFoundRobot
    outputValues["RequestedAngle"] = angleToTest
    outputValues["RequestedSpeed"] = speedToTest
    outputValues["Correction"] = correction
    outputValues["Deceleration"] = turn_deceleration
    outputValues["voltage"] = voltage
    outputValues["GyroStartAngle"] = startAngle
    outputValues["GyroEndAngle"] = endAngle
    outputValues["ActualStartAngle"] = startActualAngle
    outputValues["ActualEndAngle"] = endActualAngle

    absoluteError = abs(abs(float(outputValues["GyroEndAngle"])) - abs(float(outputValues["GyroStartAngle"])))
    outputValues["GyroError"] = absoluteError
    absoluteActualError = abs(abs(float(outputValues["ActualEndAngle"])) - abs(float(outputValues["ActualStartAngle"])))
    outputValues["ActualError"] = absoluteActualError

    return outputValues

def doOneExperiment():
    f = open("demo.csv", "a")
    writer = csv.DictWriter(f, ["DateTime", "Experiment", "Iteration", "RequestedAngle", "RequestedSpeed", "voltage", 
                                "Correction", "Deceleration",
                                "GyroStartAngle", "GyroEndAngle", 
                                "ActualStartAngle", "ActualEndAngle",
                                "GyroError", "ActualError", "StartFoundRobot", "EndFoundRobot"])
    #writer.writeheader()

    # Use some standard values that we want to test to create one datapoint.
    output = runOneTest(angleToTest=90, speedToTest=800, turn_deceleration=500, correction=0)
    output["Experiment"] = 1
    output["Iteration"] = 1

    # If the start of end image detection did not work, then dont write the output to the file.
    # increase the counter of the number of times this is happening.
    if output["StartFoundRobot"] == False or output["EndFoundRobot"] == False:
        print("------------------Ending iteration :  BAD -------------------------------------")
    else:
        print("------------------Ending iteration :  GOOD -------------------------------------")
        output["DateTime"] = datetime.datetime.now()
        writer.writerow(output)  
        
    f.flush()    
    f.close()

# This runs one experiment
doOneExperiment()
