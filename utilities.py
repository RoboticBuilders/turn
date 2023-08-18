from pybricks.hubs import PrimeHub
from pybricks.pupdevices import Motor, ColorSensor, UltrasonicSensor, ForceSensor
from pybricks.parameters import Button, Color, Direction, Port, Side, Stop
from pybricks.robotics import GyroDriveBase
from pybricks.tools import wait, StopWatch

hub = PrimeHub()

left_motor = Motor(Port.B,Direction.COUNTERCLOCKWISE)
right_motor = Motor(Port.F)

color_sensor = ColorSensor(Port.C)
bucket_arm = Motor(Port.E)
whale_arm = Motor(Port.A)
whacker = Motor(Port.D)
drive_base = GyroDriveBase(left_motor, right_motor, wheel_diameter=88, axle_track=112)

def getHeadingValue():
    return str(hub.imu.heading())

def printDriveBaseValues():
    straight_speed, straight_acceleration, turn_rate, turn_acceleration = drive_base.settings()
    print("straight_speed: " + str(straight_speed))
    print("straight_accelration: " + str(straight_acceleration))
    print("turn_rate: " + str(turn_rate))
    print("turn_acceleration: " + str(turn_acceleration))

def waitForRobotReady():
    while True:
        if (hub.imu.ready() == True):
            break

    hub.speaker.beep()

# Convert angle to zero to 359 space.
# negative angles are also converted into zero to 359 space.
def _convertAngleTo360(angle):
    negative = False
    if angle < 0:
        angle = abs(angle)
        negative = True

    degreesLessThan360 = angle
    if angle >= 360:
        degreesLessThan360 = angle % 360
    
    if negative == True:
        degreesLessThan360 = 360 - degreesLessThan360

    return degreesLessThan360

# targetAngle should be in 0-359 space.
def turnToAngle(targetAngle, turn_rate=300, turn_acceleration=500, turn_deceleration=300, correction=0.01):
    """
    Turns the robot to the specified absolute angle.
    It calculates if the right or the left turn is the closest
    way to get to the target angle. Can handle negative gyro readings.
    The input should however be in the 0-359 space.
    """
    # setup the speed to turn.
    straight_speed, straight_acceleration, _, _ = drive_base.settings()
    drive_base.settings(straight_speed=straight_speed,straight_acceleration=straight_acceleration,
                        turn_rate=turn_rate,turn_acceleration=(turn_acceleration, turn_deceleration))

    direction = None
    currentAngle = hub.imu.heading()
    currentAngle = _convertAngleTo360(currentAngle)
    
    if targetAngle >= currentAngle:
        rightTurnDegrees = targetAngle - currentAngle
        leftTurnDegrees = 360 - targetAngle + currentAngle
    else: 
        leftTurnDegrees = currentAngle - targetAngle
        rightTurnDegrees = 360 - currentAngle + targetAngle

    # Figure out the degrees to turn using the correction and the 
    # shortest turning side. Either left or Right.
    degreesToTurn = 0  
    if (rightTurnDegrees < leftTurnDegrees):
        direction = "Right"
        degreesToTurn = rightTurnDegrees - (correction * rightTurnDegrees)
    else:
        direction = "Left"
        degreesToTurn = (leftTurnDegrees - (correction * rightTurnDegrees) )* -1

    # Use the gyro drive base to turn.
    drive_base.turn(degreesToTurn)


def drive( distance_in_CM):
    drive_base.straight(distance_in_CM * 10)