import numpy as np
import math
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


r1 = 10
r2 = 10
r3 = 5
r4 = 9
theta1 = 0
theta2 = 68*np.pi/180


def theta4(x):
        out = [-r2*math.sin(theta2)-r3*math.sin(x[0])+r1*math.sin(theta1)+r4*math.sin(x[1])]
        out.append(-r2*math.cos(theta2)-r3*math.cos(x[0])+r1*math.cos(theta1)+r4*math.cos(x[1]))
        return out

total= np.array([[0,0,0]])
MATotal = np.array([[0]])

for i in range(40, 68):
    theta2 = i*np.pi/180
    x = fsolve(theta4,[10*np.pi/180, 50*np.pi/180])*180/np.pi
    total = np.append(total, [[theta2*180/np.pi, x[0], x[1]]], axis = 0)
    MA = (1)/(x[1]-total[i-40,2])
    print(MA)
    # MATotal = np.append(MATotal, [[]]
# print(total[:,1])

figWidth = 5
figHeight = 5
plt.figure(1, figsize=(figWidth, figHeight))
# plt.title("vNorm")
plt.grid(True)
plt.xlabel("Time[s]")
plt.ylabel("Velocity [m/s]")
# plt.xticks(np.arange(0, simTime+.1, 1)) 
# plt.yticks(np.arange(0, 22, 2)) 
# plt.ylim([0,22])
wing0, = plt.plot(total[:,0], '*k', label = "wing0")
wing1, = plt.plot(total[:,2], '*b', label = "wing0")
# MA, = 
plt.show()
# plt.savefig('../../simFigs/'+(os.path.basename(__file__)[:-3])+'vNorm.pdf')