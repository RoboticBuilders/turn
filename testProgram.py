from utilities import *
hub = PrimeHub()
print("Voltage=" + str(hub.battery.voltage()))
print("Start Angle=" + str(getyawangle()))
turnToAngle(targetAngle=90, speed=40)
print("End Angle=" + str(getyawangle()))
