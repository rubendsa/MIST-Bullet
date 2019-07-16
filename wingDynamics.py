import pybullet as p
import time
import pybullet_data

import numpy as np
import math



import helperFunctions as hf
# Wing Dynamics



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
    print("vABody", vABody)
    vNorm = (vABody.T @ vABody)**(1/2) # Magnitude of vehicle velocity
    alphar = math.atan2(vABody[0],abs(vABody[2])) # Angle of attack in radians. Axis rotated 90 about y. Z is the old X, X is the old Z 
    betar = math.asin(vABody[1]/vNorm) # Side-slip angle - DOUBLE CHECK COORDINATE FRAME N.E.D. VS N.W.U.

    return vA, vABody, vNorm, alphar, betar

def mh49Aero(robotId, wingId):
    alphaRTable, cLTable, cDTable, cMTable = hf.readAeroData("T1_Re0.100_M0.00_N9.0.txt") # Read aero data from file
    # print("alphar:", alphar)
    
    cL = np.interp(alphar, alphaRTable, cLTable)
    cD = np.interp(alphar, alphaRTable, cDTable)
    cM = np.interp(alphar, alphaRTable, cMTable)
    

def simpleAero(robotId, wingId):
    vA, vABody, vNorm, alphar, betar = calcFreeStreamVelocity(robotId, wingId)

    ###### Approach 1
    # cLAlpha = .0875 * 180/3.1415 # slope of CL/alpha
    # alphaStall = 25 * 3.1415/180 # stall angle 
    # cLAlphaStall = -.025 * 180/3.1415
    # if alphar > alphaStall or alphar < -1 * alphaStall:
    #     cL = cLAlphaStall * alphar + 4
    # else:    
    #     cL = cLAlpha * alphar -.02
    # cD = 0
    # cM = 0

    ###### Approach 2
    cLTable = [-1, -1, -1.10414493484302,	-0.883315947874415,	-0.662486960905811,	-0.441657973937207,	-0.220828986968604,	0,	0.220828986968604,	0.441657973937207,	0.662486960905811,	0.883315947874415,	1.10414493484302,	1.21170652584920,	1.28853623371075,	1.33463405842769,	1.35000000000000,	1.33463405842769,	1.28853623371075,	1.21170652584920,	1.10414493484302,	0.965851460692216,	0.796826103396791,	0.730000000000000,	0.730000000000000,	0.740000000000000,	0.760000000000000,	0.780000000000000,	0.972831540445641,	1.21091619655235,	1.42218430518409,	1.58862675882105,	1.69489278019590,	1.72936655653370,	1.68499212712194,	1.55979005609270,	1.35703040737636,	1.08504994405072,	0.756725704861244,	0.388640515367069,	-3.02492158249372e-06] 
    cDTable = [.1, .1, 0.0813550435243688, 0.0613436742181042, 0.0457792758687873, 0.0346618484764181, 0.0279913920409966, 0.0257679065625227, 0.0279913920409966, 0.0346618484764181, 0.0457792758687873, 0.0613436742181042, 0.0705452253508198,	0.0811270091534428,	0.0932960605264592,	0.107290469605428,	0.118019516565971,	0.129821468222568,	0.142803615044825,	0.157083976549307,	0.172792374204238,	0.190071611624662,	0.209078772787128,	0.229986650065841,	0.252985315072425,	0.278283846579668,	0.306112231237634,	0.339165775094203,	0.561664842825890,	0.847893121418590,	1.19335505107119,	1.58862782829305,	2.01989609406167,	2.46979356322061,	2.91849499945190,	3.34498481591431,	3.72841647598934,	4.04947060702661,	4.29161974980414,	4.44221394754024,	4.49331350328450]
    cMTable = [.16, .16, 0.146681149175663,	0.117619086803684,	0.0883741535184620,	0.0589921826682437,	0.0295189087697684,	0,	-0.0295189087697684,	-0.0589921826682437,	-0.0883741535184620,	-0.117619086803684,	-0.146431183983874,	-0.160453916164061,	-0.170421615607425,	-0.176385633035138,	-0.178235051816793,	-0.176145295187666,	-0.170182791654346,	-0.160422460252220,	-0.146947792845151,	-0.129850946412764,	-0.109232846970114,	-0.0983095622731028,	-0.0884786060457925,	-0.0796307454412133,	-0.0716676708970920,	-0.0754664135070912,	-0.122278914880603,	-0.194012618843902,	-0.253668919760856,	-0.306974466289298,	-0.360280006538578,	-0.411965877257355,	-0.460461628075459,	-0.504293738884495,	-0.542130392038978,	-0.572821938996161,	-0.595435831835118,	-0.609284958277656,	-0.613948519265042]
    alphaTable = 3.14/180*np.array([-90, -11, -10, 	-8,	-6,	-4,	-2,	0,	2,	4,	6,	8,	10,	11,	12,	13,	14,	15,	16,	17,	18,	19,	20,	21,	22,	23,	24,	25,	30,	35,	40,	45,	50,	55,	60,	65,	70	,75	,80,	85,	90])
    
    cL = np.interp(alphar, alphaTable, cLTable)
    cD = np.interp(alphar, alphaTable, cDTable)
    cM = np.interp(alphar, alphaTable, cMTable)


    return cL, cD, cM


def wingDynamics(robotId, wingId):
    sArea = 0.634 # area in m^2 from Xflr5
    rho = 1.225 # kg/m^3 International Standard Atmosphere air density
    
    vA, vABody, vNorm, alphar, betar = calcFreeStreamVelocity(robotId, wingId)

    cL, cD, cM = simpleAero(robotId, wingId)

    FL = 1 * cL * (1/2) * sArea * rho * (vNorm**2)
    # FL = -40
    FD = 1 * cD * (1/2) * sArea * rho * (vNorm**2)
    M =  1 * cM * (1/2) * sArea * rho * (vNorm**2)
    # M = vNorm * 1/alphar

    # print("FL:", FL, "alphar", alphar, "cL:", cL)
    
    # p.applyExternalForce(robotId, 3, [-FL, 0, 0], [0,0,0], 1)
    # p.applyExternalForce(robotId, 0, [-FL, 0, 0], [0,0,0], 1)
    # p.applyExternalForce(robotId, 1, [-FL, 0, 0], [0,0,0], 1)
    # p.applyExternalForce(robotId, 2, [-FL, 0, 0], [0,0,0], 1)
    p.applyExternalForce(robotId, wingId, [-FL, 0, -FD], [0,0,-.02], 1)
    if wingId == -1:
        p.applyExternalTorque(robotId, wingId, [0, -M, 0], 2) # # BUG: for the base_link, p.LINK_FRAME=1 is inverted with p.WORLD_FRAME=2. Hence, for LINK_FRAME, we have to use 2. https://github.com/bulletphysics/bullet3/issues/1949 
    else:
        p.applyExternalTorque(robotId, wingId, [0, -M, 0], 1)
    # Debug for wing force
    # p.addUserDebugLine([0,0,-.02], [-FL/10, 0, -.02], [1.0,1.0,1.0], parentObjectUniqueId = 1, parentLinkIndex = wingId, lifeTime = .1)
    # print("FL", FL, "alphar", alphar, "vNorm", vNorm, "M", M)