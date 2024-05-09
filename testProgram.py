from utilities import *
hub = PrimeHub()
print("Voltage=" + str(hub.battery.voltage()))
print("Start Angle=" + str(getHeadingValue()))
turnToAngle(targetAngle=90, turn_rate=800, turn_deceleration=500, correction=0)
print("End Angle=" + str(getHeadingValue()))
