from utilities import *
hub = PrimeHub()
print("Voltage=" + str(hub.battery.voltage()))
print("Start Angle=" + str(getHeadingValue()))
turnToAngle(targetAngle=180, turn_rate=300, turn_deceleration=500, correction=-0.01)
print("End Angle=" + str(getHeadingValue()))
