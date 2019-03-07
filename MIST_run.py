import numpy as np
import pybullet as p
import time
import pybullet_data

# Initalization Code
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")


# Load MIST-UAV
robotStartPos = [0,0,1]
robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
# robotId = p.loadURDF("MIST.urdf",robotStartPos, robotStartOrientation)
robotId = p.loadURDF("MIST.urdf",robotStartPos, robotStartOrientation)


# for i in range (0,1):
#     print(p.getLinkInfo(robotId, linkIndex=i))

# hingeVelocity = 500 # Multi-rotor
# hingeVelocity = 500 # Fixed-wing
hingePosition = 1.57


p.setJointMotorControl2(robotId,
                    jointIndex=0,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=hingePosition,
                    force=1000)
p.setJointMotorControl2(robotId,
                    jointIndex=1,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=hingePosition,
                    force=1000)
p.setJointMotorControl2(robotId,
                    jointIndex=2,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=hingePosition,
                    force=1000)

p.setJointMotorControl2(robotId,
                    jointIndex=3,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=0,
                    force=1000)


def cycleEverything(i, simTime):
    hingePosition = 1.57*np.sin((i*300/simTime))
    # hingePosition = np.linspace(0, 1.57, )

    for hingeNum in range(0,3):
        p.setJointMotorControl2(robotId,
                    jointIndex=hingeNum,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=hingePosition,
                    force=1000)
    
    for ctrlSurfNum in range(3,7):
        p.setJointMotorControl2(robotId,
                    jointIndex=ctrlSurfNum,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=-(hingePosition+ctrlSurfNum/4.)*3,
                    force=1000)
    print("hingeposition", hingePosition)

simTime = 10000
simDelay = 1./1000.

for i in range (simTime): #Time to run simulation
    p.stepSimulation()
    time.sleep(simDelay)
    # cycleEverything(i, simTime)

    # p.applyExternalForce(robotId, 
    #                 0, 
    #                 [0,0, 300],
    #                 [0,-.28,0],
    #                 1)

    # p.applyExternalForce(robotId, 
    #                 1, 
    #                 [0,0, 300],
    #                 [0,-.28,0],
    #                 1)

    # p.applyExternalForce(robotId, 
    #                 2, 
    #                 [0,0, 300],
    #                 [0,-.28,0],
    #                 1)
    
    # p.applyExternalForce(robotId, 
    #                 3, 
    #                 [0,0, 300],
    #                 [0,-.28,0],
    #                 1)


    # p.applyExternalForce(robotId, 
    #                 -1, 
    #                 [0,0, 2000],
    #                 [0,.85,0],
    #                 1)
                    
    
    # print("sin:", np.sin(i))


