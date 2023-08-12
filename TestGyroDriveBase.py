from pybricks.hubs import PrimeHub
from pybricks.pupdevices import Motor, ColorSensor, UltrasonicSensor, ForceSensor
from pybricks.parameters import Button, Color, Direction, Port, Side, Stop
from pybricks.robotics import GyroDriveBase
from pybricks.tools import wait, StopWatch
from pybricks.tools import wait


hub = PrimeHub()
# Initialize both motors. In this example, the motor on the
# left must turn counterclockwise to make the robot go forward.
left_motor = Motor(Port.B, Direction.COUNTERCLOCKWISE)
right_motor = Motor(Port.F)

#left_motor.run(50)
#wait(2000)
#left_motor.stop()
# Initialize the drive base. In this example, the wheel diameter is 56mm.
# The distance between the two wheel-ground contact points is 112mm.
drive_base = GyroDriveBase(left_motor, right_motor, wheel_diameter=78, axle_track=100)
drive_base.settings(straight_speed = 100)

# Drive forward by 500mm (half a meter).
#drive_base.straight(500)

# Turn around clockwise by 180 degrees.
drive_base.turn(90)
drive_base.turn(90)
drive_base.turn(90)
drive_base.turn(90)

# Drive forward again to get back to the start.
#drive_base.straight(500)

# Turn around counterclockwise.
#drive_base.turn(-180)