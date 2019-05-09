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



def quadAttitudeControl(robotId, robotDesiredPoseWorld, hingeAngle, frameState):

    # Set Gains and Parameters TODO: Move this out
    # K_position = np.eye(3) * np.array([[30, 30, 100]]) # gain for x, y, z components of error vector
    # K_velocity = np.eye(3) * np.array([[20, 20, 60]])

    K_position = np.eye(3) * np.array([[6, 6, 20]]) # gain for x, y, z components of error vector
    K_velocity = np.eye(3) * np.array([[2, 2, 10]])

    # K_rotation = np.eye(3) * np.array([[50, 50, 5]])
    # K_angularVelocity = np.eye(3) * np.array([[40, 40, 5]])
    K_rotation = np.eye(3) * np.array([[30, 30, 30]])
    K_angularVelocity = np.eye(3) * np.array([[3, 3, 3]])
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
    # positionW = a
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

    des_F = -K_position @ error_position.T - K_velocity @ error_velocity.T + np.array([[0,0, mass * gravity]]).T #
    # Compute u1 -> Force in world frame projected into the body z-axis
    zB = rotBtoW @ np.array([[0,0,1]]).T
    # print("zB", zB)
    # print("des_F", des_F)

    u1 = des_F.T @ zB

    # RUN ATTITUDE AND POSITON CONTROLLER: 
    des_zB = (des_F / LA.norm(des_F)).T
    des_xC = np.array([np.cos(des_yaw), np.sin(des_yaw), 0]).T
    des_yB = (np.cross(des_zB, des_xC) / LA.norm(np.cross(des_zB, des_xC)))
    des_xB = np.cross(des_yB, des_zB)
    des_R = np.concatenate((des_xB, des_yB, des_zB), axis = 0).T

    # RUN ONLY ATTITUDE CONTROLLER: 
    # deslistBtoW = p.getMatrixFromQuaternion(des_orientationW)
    # desrotBtoW = np.array([[deslistBtoW[0], deslistBtoW[1], deslistBtoW[2]],
    #                     [deslistBtoW[3], deslistBtoW[4], deslistBtoW[5]],
    #                     [deslistBtoW[6], deslistBtoW[7], deslistBtoW[8]]])
    # des_R = desrotBtoW


    eR_mat = .5 * (des_R.T @ rotBtoW - rotBtoW.T @ des_R)


    eR = np.array([[eR_mat[2,1], eR_mat[0,2], eR_mat[1,0]]])

    eW = (LA.inv(rotBtoW) @ angularVelocityW.T - np.array([des_angular_velocityW]).T).T # Was this supposed to be in the Body frame?

    u24 = -K_rotation @ eR.T - K_angularVelocity @ eW.T
    
    u1 = np.clip(u1, 0.0, 2000.0)
    u24 = np.clip(u24, -40.0, 40.0)
    
    u = np.concatenate((u1, u24))

    geo = np.array([[Kf, Kf, Kf, Kf],
                    [0, Kf*L, 0, -Kf*L],
                    [-Kf*L, 0, Kf*L, 0],
                    [Km, -Km, Km, -Km]])
    # Below is the inverse of "geo" computed using the matlab symbolic package
    # geoTailSitterRaw = np.array([
    #     [ 1/(4*Kf),           0, -1/(2*Kf*L),  1/(4*Km)]
    #     [ 1/(4*Kf),  1/(2*Kf*L),           0, -1/(4*Km)]
    #     [ 1/(4*Kf),           0,  1/(2*Kf*L),  1/(4*Km)]
    #     [ 1/(4*Kf), -1/(2*Kf*L),           0, -1/(4*Km)]])
    
    # Motor and elevon mixing when in a tailsitter state:
    # Motors down the rows, mixing across the columns [Throttle, roll, pitch, yaw]
    # geoTailSitter has zeros for the pitch and yaw because when in a tailsitter state, only the control surfaces are used for pitch and yaw.
    geoTailSitter = np.array([
        [ 1/(4*Kf), -1/(2*Kf*L),           0,   0],  
        [ 1/(4*Kf),  1/(2*Kf*L),           0,   0],
        [ 1/(4*Kf),  1/(2*Kf*L),           0,   0],
        [ 1/(4*Kf), -1/(2*Kf*L),           0,   0]])

    # geoTailSitterCtrlSurf: used for computing elevon angles "e" from the general actuation effort "u" 
    geoTailSitterCtrlSurf = np.array([
        [ 0,    0,    1/(2*Kf*L),     1/(4*Km)],
        [ 0,    0,    1/(2*Kf*L),     -1/(4*Km)],
        [ 0,    0,    1/(2*Kf*L),     -1/(4*Km)],
        [ 0,    0,    1/(2*Kf*L),     1/(4*Km)]])
        

    w2 = LA.inv(geo) @ u
    w2 = np.clip(w2,0, None)
    w = w2
    
    e = 0,0,0,0

    # if in a tailsitter state, recompute motor velocity and elevon angles as per geoTailSitter and geoTailSitterCtrlSurf geometry.
    if frameState == 0:
        tempStep = step
        w2 = geoTailSitter @ u
        w2 = np.clip(w2,0, None)
        w = w2
        e = geoTailSitterCtrlSurf @ u
        e = np.clip(e, -.5, .5)

    if frameState == 1:
        w2 = LA.inv(geo) @ u
        w2 = np.clip(w2,0, None)
        w = w2
        e = 0,0,0,0

    


    return w,e



###################     Helper functions   #####################
# Action Vector
def applyAction(actionVector, robotId=robotId):
    w0, w1, w2, w3, e0, e1, e2, e3, h0, h1, h2 = actionVector

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
    p.applyExternalTorque(robotId, -1, [0,0, -Mm0], 2) # BUG: for the base_link, p.LINK_FRAME=1 is inverted with p.WORLD_FRAME=2. Hence, for LINK_FRAME, we have to use 2. https://github.com/bulletphysics/bullet3/issues/1949 
    p.applyExternalTorque(robotId, 0, [0,0, Mm1], 1) 
    p.applyExternalTorque(robotId, 1, [0,0, -Mm2], 1) 
    p.applyExternalTorque(robotId, 2, [0,0, Mm3], 1) 

    # Torque for each Elevon
    p.applyExternalTorque(robotId, -1, [0,5*e0,0], 2) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 
    p.applyExternalTorque(robotId, 0, [0,5*e1,0], 1) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 
    p.applyExternalTorque(robotId, 1, [0,5*e2,0], 1) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 
    p.applyExternalTorque(robotId, 2, [0,5*e3,0], 1) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 
    # The difference in elevons induces a torque in the z axis for tailsitter. 
    p.applyExternalTorque(robotId, -1, [0,0,10*(e0+e1-e2-e3)], 2) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 



    # Visual of propeller spinning (not critical)
    p.setJointMotorControl2(robotId, propIds[0], p.VELOCITY_CONTROL, targetVelocity=w0*100, force=1000) 
    p.setJointMotorControl2(robotId, propIds[1], p.VELOCITY_CONTROL, targetVelocity=-w1*100, force=1000)
    p.setJointMotorControl2(robotId, propIds[2], p.VELOCITY_CONTROL, targetVelocity=w2*100, force=1000)
    p.setJointMotorControl2(robotId, propIds[3], p.VELOCITY_CONTROL, targetVelocity=-w3*100, force=1000)
    
    # Control surface deflection [rads]
    p.setJointMotorControl2(robotId, ctrlSurfIds[0], p.POSITION_CONTROL, targetPosition=2*e0, force=1000)
    p.setJointMotorControl2(robotId, ctrlSurfIds[1], p.POSITION_CONTROL, targetPosition=2*e1, force=1000)
    p.setJointMotorControl2(robotId, ctrlSurfIds[2], p.POSITION_CONTROL, targetPosition=2*e2, force=1000)
    p.setJointMotorControl2(robotId, ctrlSurfIds[3], p.POSITION_CONTROL, targetPosition=2*e3, force=1000)
    
    # Hinge angle [rads]
    p.setJointMotorControl2(robotId, hingeIds[0], p.POSITION_CONTROL, targetPosition=h0, maxVelocity=8, force=100)
    p.setJointMotorControl2(robotId, hingeIds[1], p.POSITION_CONTROL, targetPosition=h1, maxVelocity=8, force=100)
    p.setJointMotorControl2(robotId, hingeIds[2], p.POSITION_CONTROL, targetPosition=h2, maxVelocity=8, force=100)

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
    centerOfMass[2] = centerOfMass[2] -.01 # Z intertial offset used in the urdf file

    # print("centerOfMass:", centerOfMass)
    return centerOfMass


def wingAero():
    # Based on control input (elevon angle and air vector), apply a force and moment on the wing section local frame. 
    
    # p.applyExternalForce(robotId, -1, [0,0,10], [0,0,0], 1) #Apply m0 force[N] on link0, w.r.t. local frame
    # p.applyExternalForce(robotId, 0, [0,0,0], [0,0,0], 1) #Apply m0 force[N] on link0, w.r.t. local frame
    # p.applyExternalForce(robotId, 1, [0,0,0], [0,0,0], 1) #Apply m0 force[N] on link0, w.r.t. local frame
    # p.applyExternalForce(robotId, 2, [0,0,0], [0,0,0], 1) #Apply m0 force[N] on link0, w.r.t. local frame
    
    p.applyExternalTorque(robotId, -1, [0,200,0], 2) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 
    p.applyExternalTorque(robotId, 0, [0,200,0], 1) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 
    p.applyExternalTorque(robotId, 1, [0,200,0], 1) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 
    p.applyExternalTorque(robotId, 2, [0,200,0], 1) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 


###################     RUN SIMULATION     #####################

# tailsitterAttitudeControl()
if __name__ == "__main__":
    simTime = 1000000
    simDelay = .00001
    p.resetDebugVisualizerCamera(20, 70, -20, [0,0,0]) # Camera position (distance, yaw, pitch, focuspoint)
    # p.resetDebugVisualizerCamera(20, 70, -20, computeCenterOfMass()) # Camera position (distance, yaw, pitch, focuspoint)
    p.resetBasePositionAndOrientation(robotId, [-10,0,10], [0,0,0,1]) # Staring position of robot
    
    # p.resetBasePositionAndOrientation(robotId, [0,0,10], [.5,0,0,.5]) # Staring position of robot
    # p.resetBaseVelocity(robotId, [0,2,0], [2,0,0])

    for i in range (simTime): #Time to run simulation
        p.stepSimulation()
        time.sleep(simDelay)

        applyAction([0, 0, 0, 0, .1, .1, .1, .1, 1.57, 1.57, 1.57], robotId) #Example applyAction
        # applyAction([0, 0, 0, 0, .3, .1, -.1, -.3, 0, 0, 0], robotId) #Example applyAction


        ##### Testing attitude and position controller:
        des_positionW = [0,0,10]
        des_orientationW = p.getQuaternionFromEuler([0,0,0]) # [roll, pitch, yaw]
        des_velocityW = [0,0,0]
        des_angular_velocityW = [0,0,0]
        
        step = i
        if i>100:
            des_positionW = [10,10,10]

            hingeAngle = 0
            frameState = 0
            
            robotDesiredPoseWorld = des_positionW, des_orientationW, des_velocityW, des_angular_velocityW 
            w, e = quadAttitudeControl(robotId, robotDesiredPoseWorld, hingeAngle, frameState) # starts with w1 instead of w0 to match the motor geometry of the UAV in the paper.  
            w1, w2, w3, w0 = w
            e1, e2, e3, e0 = e

            
            if step in range(1000, 1100):
                hingeAngle = 1.57
                frameState = 0
                w, e = quadAttitudeControl(robotId, robotDesiredPoseWorld, hingeAngle, frameState) # starts with w1 instead of w0 to match the motor geometry of the UAV in the paper.  
                w1, w2, w3, w0 = w
                e1, e2, e3, e0 = [0,0,0,0]
            if step in range(1100, 2000):
                hingeAngle = 1.57
                frameState = 1
                w, e = quadAttitudeControl(robotId, robotDesiredPoseWorld, hingeAngle, frameState) # starts with w1 instead of w0 to match the motor geometry of the UAV in the paper.  
                w1, w2, w3, w0 = w
                e1, e2, e3, e0 = e
            if step in range(2000,2100):
                hingeAngle = 0
                frameState = 0
                w, e = quadAttitudeControl(robotId, robotDesiredPoseWorld, hingeAngle, frameState) # starts with w1 instead of w0 to match the motor geometry of the UAV in the paper.  
                w1, w2, w3, w0 = w
                e1, e2, e3, e0 = [0,0,0,0]
            if step in range(2100, 3000):
                hingeAngle = 0
                frameState = 0
                w, e = quadAttitudeControl(robotId, robotDesiredPoseWorld, hingeAngle, frameState) # starts with w1 instead of w0 to match the motor geometry of the UAV in the paper.  
                w1, w2, w3, w0 = w
                e1, e2, e3, e0 = e

            if step in range(3000, 3100):
                hingeAngle = 1.57
                frameState = 0
                w, e = quadAttitudeControl(robotId, robotDesiredPoseWorld, hingeAngle, frameState) # starts with w1 instead of w0 to match the motor geometry of the UAV in the paper.  
                w1, w2, w3, w0 = w
                e1, e2, e3, e0 = [0,0,0,0]
            if step in range(3100, 4000):
                hingeAngle = 1.57
                frameState = 1
                w, e = quadAttitudeControl(robotId, robotDesiredPoseWorld, hingeAngle, frameState) # starts with w1 instead of w0 to match the motor geometry of the UAV in the paper.  
                w1, w2, w3, w0 = w
                e1, e2, e3, e0 = e
           
            
            
            applyAction([w0, w1, w2, w3, e0, e1, e2, e3, hingeAngle, hingeAngle, hingeAngle], robotId)
            # p.applyExternalForce(robotId, 1, [0,1,0], [0,0,0], 2) #Apply m0 force[N] on link0, w.r.t. local frame
            # print("linkframe", p.WORLD_FRAME)

            # wingAero()

            # computeCenterOfMass()

            # visualizeCenterOfMass()
            # visualizeLinkFrame(0)
            # visualizeThrottle(w0, w1, w2, w3)
        p.resetDebugVisualizerCamera(8, 0, -20, computeCenterOfMass()) # Camera position (distance, yaw, pitch, focuspoint)
        # p.addUserDebugLine([0,0,0], (p.getLinkState(robotId, 1, 1))[0], [1.0,1.0,1.0], lifeTime = .05)
        # p.addUserDebugLine([0,0,0], [-.0, 0, .0], [1.0,0.0,0.0], parentObjectUniqueId = 1, parentLinkIndex = 0, lifeTime = .1)


