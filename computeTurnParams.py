from turn_utilities import *
from image_utilities import *
import random
import datetime
import csv

def finish():
    for i in range(1,20):
        print("\n")
    input("Press Enter to start experiment...")

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
    capturePicture(startAnglePictureFilename)

    print("Dectecting Start angle")
    startFoundRobot, width, height, angle, startActualAngle = detectAngleOfRobotUsingImage(startAnglePictureFilename, 
                                                                                           showImages=True)
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
    capturePicture(endAnglePictureFilename)

    print("Dectecting end angle")
    endFoundRobot, width, height, angle, endActualAngle = detectAngleOfRobotUsingImage(endAnglePictureFilename)
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

'''
print("Printing output of experiment")
print("Requested turn angle: " + str(output["RequestedAngle"]))
print("Requested turn speed: " + str(output["RequestedSpeed"]))
print("Battery Voltage: " + str(output["voltage"]))
print("Start angle as measured by Robot Gyro: " + str(output["GyroStartAngle"]))
print("End angle as measured by Robot Gyro: " + str(output["GyroEndAngle"]))
print("Real Start Angle (from image processing): " + str(output["ActualStartAngle"]))
print("Real End angle (from image processing): " + str(output["ActualEndAngle"]))
'''


'''
print("Angle turned according to gyro: " + str(absoluteError))
print("Actual angle turned by Robot (from image processing): " + str(absoluteActualError))
'''

# Method tries different values of turn, speed, deceleration, iteration
# and records the output.
def tryDifferentAngles(numExperiments):
    turnDecelerationToTry = [200, 300, 400, 500]
    correctionsToTry = [-0.02, -0.01, 0, 0.01, 0.02, 0.03]
    anglesToTry = [30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 0]
    speedsToTry = [200, 300, 400, 500, 600, 700, 800]
    iterationsToTry = [1, 2, 4]

    f = open("turn_experiment_output.csv", "a")
    #f = open("test.csv", "a")
    writer = csv.DictWriter(f, ["DateTime", "Experiment", "Iteration", "RequestedAngle", "RequestedSpeed", "voltage", 
                                "Correction", "Deceleration",
                                "GyroStartAngle", "GyroEndAngle", 
                                "ActualStartAngle", "ActualEndAngle",
                                "GyroError", "ActualError", "StartFoundRobot", "EndFoundRobot"])
    #writer.writeheader()
    
    for expt in range(numExperiments):
        targetAngle = random.choice(anglesToTry)
        speed = random.choice(speedsToTry)
        iterations = random.choice(iterationsToTry)
        correction = random.choice(correctionsToTry)
        deceleration = random.choice(turnDecelerationToTry)

        counterOfBadExperiments = 0
        for iter in range(iterations):
            print("------------------Starting experiment: " + str(expt) + " iteration : " + str(iter) + "-------------------------------------")
            output = runOneTest(angleToTest=targetAngle, speedToTest=speed, turn_deceleration=deceleration, correction=correction)
            output["Experiment"] = expt
            output["Iteration"] = iter

            # If the start of end image detection did not work, then dont write the output to the file.
            # increase the counter of the number of times this is happening.
            if output["StartFoundRobot"] == False or output["EndFoundRobot"] == False:
                counterOfBadExperiments = counterOfBadExperiments + 1
                print("------------------Ending iteration :  BAD -------------------------------------")
            else:
                print("------------------Ending iteration :  GOOD -------------------------------------")
                output["DateTime"] = datetime.datetime.now()
                writer.writerow(output)  
                
            f.flush()    
            
    f.close()


def doOneExperiment():
    f = open("test_one_experiment.csv", "a")
    writer = csv.DictWriter(f, ["DateTime", "Experiment", "Iteration", "RequestedAngle", "RequestedSpeed", "voltage", 
                                "Correction", "Deceleration",
                                "GyroStartAngle", "GyroEndAngle", 
                                "ActualStartAngle", "ActualEndAngle",
                                "GyroError", "ActualError", "StartFoundRobot", "EndFoundRobot"])
    #writer.writeheader()

    # Use some standard values that we want to test to create one datapoint.
    output = runOneTest(angleToTest=90, speedToTest=300, turn_deceleration=300, correction=0)
    output["Experiment"] = 1
    output["Iteration"] = iter

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

# Use this method to actually capture all the data.
tryDifferentAngles(5)

