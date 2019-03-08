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


###################     Helper functions   #####################
# Action Vector
def applyAction(actionVector, robotId):
    m0, m1, m2, m3, c0, c1, c2, c3, h0, h1, h2 = actionVector

    # Thrust for each Motor
    p.applyExternalForce(robotId, 0, [0,0, m0], [0,-.28,0], 1) #Apply m0 force[N] on link0, w.r.t. local frame
    p.applyExternalForce(robotId, 1, [0,0, m1], [0,-.28,0], 1) #Apply m1 force[N] on link1, w.r.t. local frame
    p.applyExternalForce(robotId, 2, [0,0, m2], [0,-.28,0], 1) #Apply m2 force[N] on link2, w.r.t. local frame
    p.applyExternalForce(robotId, 3, [0,0, m3], [0,-.28,0], 1) #Apply m3 force[N] on link3, w.r.t. local frame

    # Visual of propeller spinning (not critical)
    p.setJointMotorControl2(robotId, propIds[0], p.VELOCITY_CONTROL, targetVelocity=m0*10, force=1000) 
    p.setJointMotorControl2(robotId, propIds[1], p.VELOCITY_CONTROL, targetVelocity=m1*10, force=1000)
    p.setJointMotorControl2(robotId, propIds[2], p.VELOCITY_CONTROL, targetVelocity=m2*10, force=1000)
    p.setJointMotorControl2(robotId, propIds[3], p.VELOCITY_CONTROL, targetVelocity=m3*10, force=1000)
    
    # Control surface deflection [rads]
    p.setJointMotorControl2(robotId, ctrlSurfIds[0], p.POSITION_CONTROL, targetPosition=c0, force=1000)
    p.setJointMotorControl2(robotId, ctrlSurfIds[1], p.POSITION_CONTROL, targetPosition=c1, force=1000)
    p.setJointMotorControl2(robotId, ctrlSurfIds[2], p.POSITION_CONTROL, targetPosition=c2, force=1000)
    p.setJointMotorControl2(robotId, ctrlSurfIds[3], p.POSITION_CONTROL, targetPosition=c3, force=1000)
    
    # Hinge angle [rads]
    p.setJointMotorControl2(robotId, hingeIds[0], p.POSITION_CONTROL, targetPosition=h0, force=1000)
    p.setJointMotorControl2(robotId, hingeIds[1], p.POSITION_CONTROL, targetPosition=h1, force=1000)
    p.setJointMotorControl2(robotId, hingeIds[2], p.POSITION_CONTROL, targetPosition=h2, force=1000)

# State Vector
def getUAVState(robotId):
    a, b, c, d, e, f = p.getLinkState(robotId, 0)
    position = e # x,y,z
    orientation = f #Quaternion
    return position, orientation 




###################     RUN SIMULATION     #####################

# tailsitterAttitudeControl()

print("numjoints: ", p.getNumJoints(robotId))
simTime = 10000
simDelay = 1./1000.


for i in range (simTime): #Time to run simulation
    p.stepSimulation()
    time.sleep(simDelay)

    # applyAction([500, 500, 500, 500, .3, .1, -.1, -.3, .2, .2, .2], robotId) #Example applyAction
    print(getUAVState(robotId))
    


