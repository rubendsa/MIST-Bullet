import pybullet as p
import time
import pybullet_data

import numpy as np

# Various helper functions for physics calculations
def computeCenterOfMass(robotId):
    allLinkPositions=[]    #TODO: Refactor this.

    allLinkPositions.append((p.getBasePositionAndOrientation(robotId))[0])
    for i in range(0, 3):
        allLinkPositions.append((p.getLinkState(robotId, i, 1))[0])

    centerOfMass = np.sum(allLinkPositions, axis = 0)/4 #Average x, y, z, of all 4 link CoMs 
    centerOfMass[2] = centerOfMass[2] -.01 # Z intertial offset used in the urdf file

    return centerOfMass


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

def getUAVState(robotId):
    a, b, c, d, e, f, g, h = p.getLinkState(robotId, 0, 1)
    position = e # x,y,z
    orientation = f #Quaternion
    velocity = g 
    angular_velocity = h
    return position, orientation, velocity, angular_velocity 


def step():
    p.stepSimulation()
    # time.sleep(0.001)

def set_to_pos_and_q(pos, q):
    p.resetBasePositionAndOrientation(robotId, pos, q)

def visualizeThrottle(m0, m1, m2, m3):
    p.addUserDebugLine([0,0,0], [0, 0, m0/10], [1.0,1.0,1.0], parentObjectUniqueId = 1, parentLinkIndex = -1, lifeTime = .1)
    p.addUserDebugLine([0,0,0], [0, 0, m1/10], [1.0,0.0,0.0], parentObjectUniqueId = 1, parentLinkIndex = 0, lifeTime = .1)
    p.addUserDebugLine([0,0,0], [0, 0, m2/10], [0.0,1.0,0.0], parentObjectUniqueId = 1, parentLinkIndex = 1, lifeTime = .1)
    p.addUserDebugLine([0,0,0], [0, 0, m3/10], [0.0,0.0,1.0], parentObjectUniqueId = 1, parentLinkIndex = 2, lifeTime = .1)

def visualizeLinkFrame(link):
    p.addUserDebugLine([0,0,0], [10, 0, 0], [1.0,0.0,0.0], parentObjectUniqueId = 1, parentLinkIndex = link, lifeTime = .1)
    p.addUserDebugLine([0,0,0], [0, 10, 0], [0.0,1.0,0.0], parentObjectUniqueId = 1, parentLinkIndex = link, lifeTime = .1)
    p.addUserDebugLine([0,0,0], [0, 0, 10], [0.0,0.0,1.0], parentObjectUniqueId = 1, parentLinkIndex = link, lifeTime = .1)


def visualizeCenterOfMass():
    p.addUserDebugLine([0,0,0], computeCenterOfMass(), [1.0,1.0,1.0], lifeTime = .05)

