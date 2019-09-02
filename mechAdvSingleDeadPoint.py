import numpy as np
import math
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import sys
import os

r1 = .94
r2 = .88
r3 = 1.44
r4 = 1.42
theta1 = 18.61*np.pi/180
theta2 = 68*np.pi/180


def theta4(x):
        out = [-r2*math.sin(theta2)-r3*math.sin(x[0])+r1*math.sin(theta1)+r4*math.sin(x[1])]
        out.append(-r2*math.cos(theta2)-r3*math.cos(x[0])+r1*math.cos(theta1)+r4*math.cos(x[1]))
        return out

total= np.array([[0,0,0]])
MATotal = np.array([[0]])

# startInputAngle = 53.95*np.pi/180
# endInputAngle = 174.57*np.pi/180
# startInputAngle = 53*np.pi/180
# endInputAngle = 174*np.pi/180
startInputAngle = 53*np.pi/180
endInputAngle = (360-40)*np.pi/180
stepAngle = 1*np.pi/180
angleRange = np.arange(startInputAngle, endInputAngle, stepAngle)

x3init = .1
x4init = .1
for i in range(len(angleRange)):
    theta2 = angleRange[i]
    x = fsolve(theta4,[x3init, x4init])
    total = np.append(total, [[theta2, x[0], x[1]]], axis = 0)
    MA = stepAngle/(x[1]-total[i,2])
    MATotal = np.append(MATotal, [[MA]], axis = 0)
    print(MA)
    x3init = x[0]
    x4init = x[1]
    # MATotal = np.append(MATotal, [[]]
# print(total[:,1])

total = total*180/np.pi

figWidth = 5
figHeight = 3
plt.figure(1, figsize=(figWidth, figHeight))
# plt.title("vNorm")
plt.grid(True)
plt.xlabel("Input Angle [deg]")
plt.ylabel("Angle [deg]")
# plt.xticks(np.arange(0, simTime+.1, 1)) 
# plt.yticks(np.arange(0, 22, 2)) 
# plt.ylim([0,22])
inputAngle, = plt.plot(total[2:,0], total[2:,0], 'k', label = "inputAngle")
outputAngle, = plt.plot(total[2:,0], total[2:,2], 'b', label = "outputAngle")
plt.legend([inputAngle, outputAngle], ["Input Angle", "Output Angle"])
plt.tight_layout()
plt.savefig('../../simFigs/'+(os.path.basename(__file__)[:-3])+'_inputOutputAngles.pdf')


plt.figure(2, figsize=(figWidth, figHeight))
# plt.title("vNorm")
plt.grid(True)
plt.xlabel("Input Angle [deg]")
plt.ylabel("Mechanical Advantage")
# plt.xticks(np.arange(0, simTime+.1, 1)) 
# plt.yticks(np.arange(0, 22, 2)) 
# plt.ylim([0,22])
inputAngle, = plt.plot(total[2:,0], MATotal[2:], 'k')
plt.tight_layout()
plt.savefig('../../simFigs/'+(os.path.basename(__file__)[:-3])+'_MA.pdf')

plt.show()

# wing2, = plt.plot(MATotal[2:], '-b', label = "wing0")
