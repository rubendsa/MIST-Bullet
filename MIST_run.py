import numpy as np
from numpy import linalg as LA
import pybullet as p
import time
import pybullet_data
import matplotlib as plt
import math


########## START UNCOMMENT FOR WINDOWS ###########
# from ctypes import windll #new
# timeBeginPeriod = windll.winmm.timeBeginPeriod #new
# timeBeginPeriod(1) #new
########## END UNCOMMENT FOR WINDOWS ############

# Initalization Code
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
# physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.81)
planeId = p.loadURDF("plane.urdf")



# Load MIST-UAV
robotStartPos = [0,0,1]
robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
# robotId = p.loadURDF("MIST.urdf",robotStartPos, robotStartOrientation)
robotId = p.loadURDF("MIST.urdf",robotStartPos, robotStartOrientation)
# print(robotId)

hingeIds = [0, 1, 2]
ctrlSurfIds = [9,  7, 5, 3]
propIds = [10, 8, 6, 4]


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
    

    # print("hingeposition", hingePosition)



def quadAttitudeControl(robotId, robotDesiredPoseWorld):

    # Set Gains and Parameters TODO: Move this out
    # K_position = np.eye(3) * np.array([[30, 30, 100]]) # gain for x, y, z components of error vector
    # K_velocity = np.eye(3) * np.array([[20, 20, 60]])

    K_position = np.eye(3) * np.array([[20, 20, 20]]) # gain for x, y, z components of error vector
    K_velocity = np.eye(3) * np.array([[10, 10, 10]])

    # K_rotation = np.eye(3) * np.array([[50, 50, 5]])
    # K_angularVelocity = np.eye(3) * np.array([[40, 40, 5]])
    K_rotation = np.eye(3) * np.array([[3000, 3000, 3000]])
    K_angularVelocity = np.eye(3) * np.array([[300, 300, 300]])
    # K_rotation = np.eye(3) * np.array([[200, 200, 200]])
    # K_angularVelocity = np.eye(3) * np.array([[100, 100, 100]])

    Kf = 1
    Km = .1

    # Kf = 8.54858e-06
    # Km = kf * .06
    L = .28

    mass = 4 #Mass in [kg]
    gravity = 9.81

    # Get pose
    a, b, c, d, e, f, g, h = p.getLinkState(robotId, 0, 1) #getLinkState() has a bug for getting the parent link index (-1). Use 0 for now

    positionW = computeCenterOfMass()
    orientationW = b
    positionB = e
    orientationB = f
    velocityW = g
    angularVelocityW = np.array([h])
    # Get World to Body Rotation Matrix from Quaternion (3x3 for x,y,z)
    # print(p.getMatrixFromQuaternion(orientationW))
    listBtoW = p.getMatrixFromQuaternion(orientationW)
    rotBtoW = np.array([[listBtoW[0], listBtoW[1], listBtoW[2]],
                        [listBtoW[3], listBtoW[4], listBtoW[5]],
                        [listBtoW[6], listBtoW[7], listBtoW[8]]])

    des_positionW, des_orientationW, des_velocityW, des_angular_velocityW = robotDesiredPoseWorld
    des_yaw = 0
    

    # Compute position and velocity error
    error_position = np.array([positionW]) - np.array([des_positionW])
    error_velocity = np.array([velocityW]) - np.array([des_velocityW])

    # print(type(K_position))
    # print(type(error_position))
    # print(type(K_velocity))
    # print(type(error_velocity))
    # print(type(np.array([[0,0, mass * g]]).T))
    # print((error_position.T))
    # Compute Force in world frame
    des_F = -K_position @ error_position.T - K_velocity @ error_velocity.T + np.array([[0,0, mass * gravity]]).T #
    # Compute u1 -> Force in world frame projected into the body z-axis
    zB = rotBtoW @ np.array([[0,0,1]]).T
    # print("zB", zB)
    # print("des_F", des_F)

    u1 = des_F.T @ zB

    des_zB = (des_F / LA.norm(des_F)).T

    des_xC = np.array([np.cos(des_yaw), np.sin(des_yaw), 0]).T
    # des_xC = np.clip(des_xC, 0, .2)

    # print(des_zB)
    # print(des_xC)
    # print(np.cross(des_zB.T, des_xC.T))
    des_yB = (np.cross(des_zB, des_xC) / LA.norm(np.cross(des_zB, des_xC)))
    # print(LA.norm(np.cross(des_zB.T, des_xC.T)))
    # print((np.cross(des_zB.T, des_xC.T) / LA.norm(np.cross(des_zB.T, des_xC.T))).T)
    des_xB = np.cross(des_yB, des_zB)

    # print("des_xB.T", des_xB.T)
    # print("des_xB", des_xB)
    # print("des_yB", des_yB)
    # print("des_zB", des_zB)
    # print("des_R", des_R)

    des_R = np.concatenate((des_xB, des_yB, des_zB), axis = 0).T

    desRoll = -math.asin(des_R[2,0])
    desPitch = math.atan2(des_R[2,1], des_R[2,1])
    desYaw = math.atan2(des_R[1,0], des_R[0,0])
    # desAnglesCliped = np.clip([desRoll, desPitch, desYaw], -.2, .2)
    desAnglesCliped = [desRoll, desPitch, desYaw]

    deslistBtoW = p.getMatrixFromQuaternion(p.getQuaternionFromEuler(desAnglesCliped))
    desrotBtoW = np.array([[deslistBtoW[0], deslistBtoW[1], deslistBtoW[2]],
                        [deslistBtoW[3], deslistBtoW[4], deslistBtoW[5]],
                        [deslistBtoW[6], deslistBtoW[7], deslistBtoW[8]]])
    # des_R = desrotBtoW

    # deslistBtoW = p.getMatrixFromQuaternion(des_orientationW)
    # desrotBtoW = np.array([[deslistBtoW[0], deslistBtoW[1], deslistBtoW[2]],
    #                     [deslistBtoW[3], deslistBtoW[4], deslistBtoW[5]],
    #                     [deslistBtoW[6], deslistBtoW[7], deslistBtoW[8]]])
    # des_R = desrotBtoW


    eR_mat = .5 * (des_R.T @ rotBtoW - rotBtoW.T @ des_R)
    # print("eR_mat:", eR_mat)

    eR = np.array([[eR_mat[2,1], eR_mat[0,2], eR_mat[1,0]]])
    # print("eR", eR)
    # print("er_mat", eR_mat)
    # print("er", eR)
    # print(angularVelocityW)
    # print(des_angular_velocityW)

    eW = (LA.inv(rotBtoW) @ angularVelocityW.T - np.array([des_angular_velocityW]).T).T # Was this supposed to be in the Body frame?
    # print("eW", eW)


    # print("-K_rotation @ eR.T", -K_rotation @ eR.T)
    # print("- K_angularVelocity @ eW.T", - K_angularVelocity @ eW.T)
    u24 = -K_rotation @ eR.T - K_angularVelocity @ eW.T
    
    u1 = np.clip(u1, 0.0, 100.0)
    u24 = np.clip(u24, -20.0, 20.0)
    
    # print("u1:", u1)
    # print("des_F", des_F)
    # print("zB", zB)
    # print("eR", eR)
    # print("k_Rot", -K_rotation @ eR.T)

    # print("u24:", u24)
    u = np.concatenate((u1, u24))
    # print("u:", u)

    # print("u:", u)

    geo = np.array([[Kf, Kf, Kf, Kf],
                    [0, Kf*L, 0, -Kf*L],
                    [-Kf*L, 0, Kf*L, 0],
                    [Km, -Km, Km, -Km]])
    
    # Compute angular velocities
    # print("LA.inv(geo):", LA.inv(geo))
    # print("u:", u)

    w2 = LA.inv(geo) @ u
    # w2 = np.clip(w2,0, None)
    w = w2
    # print("w2", w2)
    # w = np.sqrt(w2)
    # print("w", w)

    return w



###################     Helper functions   #####################
# Action Vector
def applyAction(actionVector, robotId=robotId):
    w0, w1, w2, w3, c0, c1, c2, c3, h0, h1, h2 = actionVector

    Kf = 1 # TODO: Put this in an object. 
    Km = .1

    # Kf = 8.54858e-06
    # Km = Kf * .06

    Fm0 = Kf * w0 
    Fm1 = Kf * w1  
    Fm2 = Kf * w2 
    Fm3 = Kf * w3
    Mm0 = Km * w0
    Mm1 = Km * w1
    Mm2 = Km * w2
    Mm3 = Km * w3

    # Thrust for each Motor
    p.applyExternalForce(robotId, -1, [0,0, Fm0], [0,0,0], 1) #Apply m0 force[N] on link0, w.r.t. local frame
    p.applyExternalForce(robotId, 0, [0,0, Fm1], [0,0,0], 1) #Apply m1 force[N] on link1, w.r.t. local frame
    p.applyExternalForce(robotId, 1, [0,0, Fm2], [0,0,0], 1) #Apply m2 force[N] on link2, w.r.t. local frame
    p.applyExternalForce(robotId, 2, [0,0, Fm3], [0,0,0], 1) #Apply m3 force[N] on link3, w.r.t. local frame

    # Torque for each Motor
    # p.applyExternalTorque(robotId, -1, [0,0, Mm0], 1) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 
    # p.applyExternalTorque(robotId, 0, [0,0, -Mm1], 1) 
    # p.applyExternalTorque(robotId, 1, [0,0, Mm2], 1) 
    # p.applyExternalTorque(robotId, 2, [0,0, -Mm3], 1) 
    p.applyExternalTorque(robotId, -1, [0,0, -Mm0], 1) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 
    p.applyExternalTorque(robotId, 0, [0,0, Mm1], 1) 
    p.applyExternalTorque(robotId, 1, [0,0, -Mm2], 1) 
    p.applyExternalTorque(robotId, 2, [0,0, Mm3], 1) 


    # Visual of propeller spinning (not critical)
    p.setJointMotorControl2(robotId, propIds[0], p.VELOCITY_CONTROL, targetVelocity=w0*100, force=1000) 
    p.setJointMotorControl2(robotId, propIds[1], p.VELOCITY_CONTROL, targetVelocity=-w1*100, force=1000)
    p.setJointMotorControl2(robotId, propIds[2], p.VELOCITY_CONTROL, targetVelocity=w2*100, force=1000)
    p.setJointMotorControl2(robotId, propIds[3], p.VELOCITY_CONTROL, targetVelocity=-w3*100, force=1000)
    
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
def getUAVState(robotId=robotId):
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


def computeCenterOfMass():
    allLinkPositions=[]    #TODO: Refactor this.


    allLinkPositions.append((p.getBasePositionAndOrientation(robotId))[0])
    for i in range(0, 3):
        # a[i], b[i], c[i], d[i], e[i], f[i], g[i], h[i] = p.getLinkState(robotId, 0, 1)
        allLinkPositions.append((p.getLinkState(robotId, i, 1))[0])
        # print("a", a)
        # b.append(a[4])
    
    centerOfMass = np.sum(allLinkPositions, axis = 0)/4 #Average x, y, z, of all 4 link CoMs 
    # print(allLinkPositions[0])
    # print(allLinkPositions[1])
    # print(allLinkPositions[2])
    # print(allLinkPositions[3])

    # print("centerOfMass:", centerOfMass)
    return centerOfMass




###################     RUN SIMULATION     #####################

# tailsitterAttitudeControl()
if __name__ == "__main__":
    simTime = 1000000
    simDelay = 0.001
    p.resetDebugVisualizerCamera(20, 70, -20, [0,0,0]) # Camera position (distance, yaw, pitch, focuspoint)
    # p.resetBasePositionAndOrientation(robotId, [0,0,10], [.5,0,0,.5]) # Staring position of robot
    p.resetBasePositionAndOrientation(robotId, [0,0,5], [0,0,0,1]) # Staring position of robot

    for i in range (simTime): #Time to run simulation
        p.stepSimulation()
        time.sleep(simDelay)

        # applyAction([0, 0, 0, 0, .9, .9, .9, .9, 1.57, 1.57, 1.57], robotId) #Example applyAction
        # applyAction([0, 0, 0, 0, .3, .1, -.1, -.3, 0, 0, 0], robotId) #Example applyAction


        ##### Testing attitude and position controller:
        des_positionW = [0,0,5]
        des_orientationW = p.getQuaternionFromEuler([0,0,0]) # [roll, pitch, yaw]
        des_velocityW = [0,0,0]
        des_angular_velocityW = [0,0,0]

        if i>100:
            des_positionW = [0,0,5]
        
        if i>1000:
            des_positionW = [0,-.03*(i-1000),5]

        robotDesiredPoseWorld = des_positionW, des_orientationW, des_velocityW, des_angular_velocityW 
        print("des_position", des_positionW)
        w1, w2, w3, w0 = quadAttitudeControl(robotId, robotDesiredPoseWorld) # starts with w1 instead of w0 to match the motor geometry of the UAV in the paper.  
        applyAction([w0, w1, w2, w3, -1, -1, -1, -1, 1.57, 1.57, 1.57], robotId)
        # applyAction([0, 0, 0, 100, -1, -1, -1, -1, 1.57, 1.57, 1.57], robotId)

        computeCenterOfMass()
        # visualizeCenterOfMass()
        # visualizeLinkFrame(0)
        # visualizeThrottle(w0, w1, w2, w3)


