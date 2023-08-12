import subprocess


# These are utility functions that are used by the computeTurnParameters
# program.

# This function uses PybricksDev to push the program to the brick
# filename: The name of the file to push. Should be in the same directory as the program.
# returns the output of the program, all prints in the program are captured.
def uploadProgramToHub(filename):
    process = subprocess.run(["pybricksdev", "run", "ble", filename], capture_output=True, ) 
    return str(process.stdout)

# Creates a simple program to turn the robot using the libraries
# It passes in the speed and turn angles.
def createTurnProgram(targetAngle, speed, filename="testProgram.py"):
    file = open(filename, "w")
    file.write("from utilities import *\n")
    file.write("hub = PrimeHub()\n")
    file.write("print(\"Voltage=\" + str(hub.battery.voltage()))\n")
    file.write("print(\"Start Angle=\" + str(getyawangle()))\n")
    file.write("turnToAngle(targetAngle=" + str(targetAngle) + ", speed=" + str(speed) + ")\n")
    file.write("print(\"End Angle=\" + str(getyawangle()))\n")
    file.close()

# Extract voltage from the output of the program
def extractVoltageFromProgramOutput(output):
    voltage = 0
    stringToSearch="Voltage="
    startIndex = output.find(stringToSearch)
    # Voltage is always a 4 digit number.
    voltageStr = output[startIndex+len(stringToSearch):startIndex+len(stringToSearch) + 4]
    voltage = int(voltageStr)
    return voltage

# Extract Start angle from the output of the program
def extractAngleFromProgramOutput(stringToSearchFor, output):
    startAngle = None
    stringToSearch=stringToSearchFor
    startIndex = output.find(stringToSearch) + len(stringToSearch)
    if (startIndex == -1):
        return None

    startAngleStr = ''
    for i in range(startIndex, len(output)):
        if ((output[i] >= '0' and output[i] <= '9') or output[i] == '.'):
            startAngleStr = startAngleStr + output[i]
        else:
            break

    startAngle = float(startAngleStr)
    #print("Angle = " + str(startAngle))
    return startAngle
 

# Extracts voltage, start angle and end angle from the program output.
def extractParamsFromProgramOutput(output):
    voltage = extractVoltageFromProgramOutput(output)
    startAngle = extractAngleFromProgramOutput("Start Angle=", output)
    endAngle = extractAngleFromProgramOutput("End Angle=", output)
    return voltage, startAngle, endAngle
