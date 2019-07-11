import pybullet as p
import time
import pybullet_data

import numpy as np
from numpy import loadtxt

import math


# Wing Dynamics

def readAeroData():
    lines = loadtxt("T1_Re0.100_M0.00_N9.0.txt", unpack=False, skiprows=11)
    alphaRTable = lines[:,0]*3.1415/180
    cLTable = lines[:,1]
    cDTable = lines[:,2]
    cMTable = lines[:,4]
    return alphaRTable, cLTable, cDTable, cMTable

def calcFreeStreamVelocity(robotId, wingId):
    if wingId == -1:
        # vA = p.getBaseVelocity(robotId)[0]
        # listMat = p.getMatrixFromQuaternion(np.array(p.getBasePositionAndOrientation(robotId)[1]))
        vA = np.array(p.getLinkState(robotId, wingId+1, 1)[6]) # Calculate Air-relative velocity vector Va
        listMat = p.getMatrixFromQuaternion(p.getLinkState(robotId, wingId+1, 1)[1])
    else:
        vA = np.array(p.getLinkState(robotId, wingId, 1)[6]) # Calculate Air-relative velocity vector Va
        listMat = p.getMatrixFromQuaternion(p.getLinkState(robotId, wingId, 1)[1])

    rotWtoB = np.array([[listMat[0], listMat[1], listMat[2]],
                        [listMat[3], listMat[4], listMat[5]],
                        [listMat[6], listMat[7], listMat[8]]])
    vABody = np.linalg.inv(rotWtoB) @ vA

    vNorm = (vABody.T @ vABody)**(1/2) # Magnitude of vehicle velocity
    alphar = math.atan2(vABody[0],abs(vABody[2])) # Angle of attack in radians. Axis rotated 90 about y. Z is the old X, X is the old Z 
    betar = math.asin(vABody[1]/vNorm) # Side-slip angle - DOUBLE CHECK COORDINATE FRAME N.E.D. VS N.W.U.

    return vA, vABody, vNorm, alphar, betar

def wingDynamics(robotId, wingId):
    sArea = 0.634 # area in m^2 from Xflr5
    rho = 1.225 # kg/m^3 International Standard Atmosphere air density
    
    vA, vABody, vNorm, alphar, betar = calcFreeStreamVelocity(robotId, wingId)
    print("alphar", alphar)
    alphaRTable, cLTable, cDTable, cMTable = readAeroData()
    # print("alphar:", alphar)
    cL = np.interp(alphar, alphaRTable, cLTable)
    cD = np.interp(alphar, alphaRTable, cDTable)
    cM = np.interp(alphar, alphaRTable, cMTable)
    # print("cL", cL, "cD", cD, "cM", cM)
    # cL = .9
    # print("vNorm", vNorm)
    FL = 2*cL * (1/2) * sArea * rho * (vNorm**2)
    FD = cD * (1/2) * sArea * rho * (vNorm**2)
    M = 2*cM * (1/2) * sArea * rho * (vNorm**2)


    # print("FL:", FL, "alphar", alphar, "cL:", cL)
    p.applyExternalForce(robotId, wingId, [FL, 0, 0], [0,0,-.1], 1)
    p.applyExternalTorque(robotId, wingId, [0, M, 0], 1)
    # Debug for wing force
    # p.addUserDebugLine([0,0,0], [-FL/10, 0, 0], [1.0,1.0,1.0], parentObjectUniqueId = 1, parentLinkIndex = wingId, lifeTime = .1)
