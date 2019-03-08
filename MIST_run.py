import numpy as np
import pybullet as p
import time
import pybullet_data

# Initalization Code
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.81)
planeId = p.loadURDF("plane.urdf")


# Load MIST-UAV
robotStartPos = [0,0,1]
robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
# robotId = p.loadURDF("MIST.urdf",robotStartPos, robotStartOrientation)
robotId = p.loadURDF("MIST.urdf",robotStartPos, robotStartOrientation)


hingeIds = [0, 1, 2]
ctrlSurfIds = [3, 5, 7, 9]
propIds = [4, 6, 8, 10]


# for i in range (0,1):
#     print(p.getLinkInfo(robotId, linkIndex=i))

def setHingePosition(hingePosition):
    hingeForce = 100
    p.setJointMotorControl2(robotId,
                        jointIndex=0,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=hingePosition,
                        force=hingeForce)
    p.setJointMotorControl2(robotId,
                        jointIndex=1,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=hingePosition,
                        force=hingeForce)
    p.setJointMotorControl2(robotId,
                        jointIndex=2,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=hingePosition,
                        force=hingeForce)



def cycleEverything(i, simTime):
    hingePosition = 1.57*np.sin((i*300/simTime))

    for hingeNum in hingeIds:
        p.setJointMotorControl2(bodyUniqueId=robotId,
                                jointIndex=hingeNum,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=hingePosition,
                                force=1000)
    
    for ctrlSurfNum in ctrlSurfIds:
        p.setJointMotorControl2(bodyUniqueId=robotId,
                                jointIndex=ctrlSurfNum,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=-(hingePosition),
                                force=1000)

    for propNum in propIds:
        p.setJointMotorControl2(bodyUniqueId=robotId,
                                jointIndex=propNum,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=100,
                                force=1000)
    

    print("hingeposition", hingePosition)



def quadAttitudeControl():
    p.applyExternalForce(robotId, 
                    0, 
                    [0,0, 300],
                    [0,-.28,0],
                    1)

    p.applyExternalForce(robotId, 
                    1, 
                    [0,0, 300],
                    [0,-.28,0],
                    1)

    p.applyExternalForce(robotId, 
                    2, 
                    [0,0, 300],
                    [0,-.28,0],
                    1)
    
    p.applyExternalForce(robotId, 
                    3, 
                    [0,0, 300],
                    [0,-.28,0],
                    1)

    # p.applyExternalForce(robotId, 
    #                 -1, 
    #                 [0,0, 2000],
    #                 [0,.85,0],
    #                 1)

def tailsitterAttitudeControl():
    # print("link0 state:",p.getLinkState(robotId, 0))
    a, b, c, d, e, f= p.getLinkState(robotId, 0)

    print("Orientation:", f) 
    p.applyExternalForce(robotId, 
                    0, 
                    [0,0, 300],
                    [0,-.28,0],
                    1)

    p.applyExternalForce(robotId, 
                    1, 
                    [0,0, 300],
                    [0,-.28,0],
                    1)

    p.applyExternalForce(robotId, 
                    2, 
                    [0,0, 300],
                    [0,-.28,0],
                    1)
    
    p.applyExternalForce(robotId, 
                    3, 
                    [0,0, 300],
                    [0,-.28,0],
                    1)


###################     RUN SIMULATION     #####################

# tailsitterAttitudeControl()

print("numjoints: ", p.getNumJoints(robotId))
simTime = 10000
simDelay = 1./1000.



for i in range (simTime): #Time to run simulation
    p.stepSimulation()
    time.sleep(simDelay)

    setHingePosition(1.57)
    # cycleEverything(i, simTime)
    
    


