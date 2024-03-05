
from hub import port

MotorB = port.B.motor
MotorC = port.C.motor
S1d = port.F.device
S2d = port.D.device

#Mode 4 RREFL - Raw Reflect Light
S1d.mode(4) 
S2d.mode(4)

#Define Sensor 1,2 raw min and max value
S1_in_min = 265
S1_in_max = 1024
S2_in_min = 240
S2_in_max = 1024

#Function - return normalized reflect light value
def S1():
    return (S1d.get()[0] - S1_in_min) * 100 / (S1_in_max - S1_in_min)

def S2():
    return (S2d.get()[0] - S2_in_min) * 100/ (S2_in_max - S2_in_min)

#Constant define 
Kp = 0.3
Kd = 7.5
Base = 90
err = 0
err_old = 0
pwrB = 0
pwrC = 0
corr = 0

while True:
    err = S1() - S2()
    corr = err * Kp + Kd * (err - err_old)
    pwrB = int(-Base - corr) # motor B is invert
    pwrC = int(Base - corr)
    MotorB.pwm(pwrB)
    MotorC.pwm(pwrC)
    err_old = err