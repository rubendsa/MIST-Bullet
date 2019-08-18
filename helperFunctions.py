import pybullet as p
import time
import pybullet_data

import numpy as np
from numpy import loadtxt
import helperFunctions as hf
import math


# Various helper functions for physics calculations

def rotY(angle):
    mat = np.array([[math.cos(angle), 0, math.sin(angle)],
                [0            , 1,        0       ], 
                [-math.sin(angle), 0,math.cos(angle)]])
    return mat

def readAeroData(fileName):
    lines = loadtxt(fileName, unpack=False, skiprows=11)
    alphaRTable = lines[:,0]*3.1415/180
    cLTable = lines[:,1]
    cDTable = lines[:,2]
    cMTable = lines[:,4]
    return alphaRTable, cLTable, cDTable, cMTable

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


def visualizeZoom(robotId, i, startIter, duration, startZoom, endZoom, yaw, pitch):
    rateZoom = (endZoom-startZoom)
    
    if i>(startIter+duration):
        zoom = startZoom + rateZoom * (1)    
    else:
        zoom = startZoom + rateZoom * math.sin(1.57*(i-startIter)/duration)
   
    p.resetDebugVisualizerCamera(zoom, yaw, pitch, hf.computeCenterOfMass(robotId)) # Camera position (distance, yaw, pitch, focuspoint)

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 

def rotationMatrixToEulerAngles(R) : # Copied from 
 
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

def eulerAnglesToRotationMatrix(theta) :
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
         
         
                     
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R


# class actuatorStored(self):
#     def motors(self):
#         self.w1, w2, w3, w0 = w
#             e1, e2, e3, e0 = e