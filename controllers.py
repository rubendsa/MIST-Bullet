import pybullet as p
import time
import pybullet_data

import numpy as np
from numpy import linalg as LA
import math

import helperFunctions as hf


from numba import jit



# Controllers for the UAV
# @jit(nopython=True)
def quadAttitudeControl(robotId, step, robotDesiredPoseWorld, frameState, ctrlMode):

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

    K_position = 1 * K_position
    K_velocity = 3 * K_velocity
    K_rotation = 3 * K_rotation
    K_angularVelocity = 3 * K_angularVelocity

    # K_position = 1 * K_position
    # K_velocity = 1 * K_velocity
    # K_rotation = 1 * K_rotation
    # K_angularVelocity = 1 * K_angularVelocity

    # Kf = 1
    # Km = .1
    # Kf = 2.02E-7
    # Km = 2.02E-8
    # Kf = 2.0268E-7 #710 KV
    # Km = 2.0268E-8
    Kf = 2.0661E-7 # 770 KV
    Km = 2.0661E-8


    # Kf = 8.54858e-06
    # Km = kf * .06
    L = .28

    mass = 3.68 #Mass in [kg]
    gravity = 9.81

    # Get pose
    a, b, c, d, e, f, g, h = p.getLinkState(robotId, 0, 1) #getLinkState() has a bug for getting the parent link index (-1). Use 0 for now

    positionW = hf.computeCenterOfMass(robotId)
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

    # rotFixedWing = np.array([[math.cos(1.57), 0, -math.sin(1.57)],
    #                         [0, 1, 0],
    #                         [math.sin(1.57), 0, math.cos(1.57)]])
    
    # rotBtoW = rotBtoW * np.linalg.inv(rotFixedWing)

    des_positionW, des_orientationW, des_velocityW, des_angular_velocityW, des_yawW = robotDesiredPoseWorld
    # des_yaw = 0
    

    # Compute position and velocity error
    error_position = np.array([positionW]) - np.array([des_positionW])
    error_velocity = np.array([velocityW]) - np.array([des_velocityW])

    des_F = -K_position @ error_position.T - K_velocity @ error_velocity.T + np.array([[0,0, mass * gravity]]).T #
    # Compute u1 -> Force in world frame projected into the body z-axis
    zB = rotBtoW @ np.array([[0,0,1]]).T
    
    # print("zB", zB)
    # print("des_F", des_F)

    u1 = des_F.T @ zB
    if ctrlMode == "attitude" and frameState == "fixedwing":
        xB = rotBtoW @ np.array([[1,0,0]]).T
        u1 = des_F.T @ xB

    # RUN ATTITUDE AND POSITON CONTROLLER: 
    des_zB = (des_F / LA.norm(des_F)).T
    des_xC = np.array([np.cos(des_yawW), np.sin(des_yawW), 0]).T
    des_yB = (np.cross(des_zB, des_xC) / LA.norm(np.cross(des_zB, des_xC)))
    des_xB = np.cross(des_yB, des_zB)
    des_R = np.concatenate((des_xB, des_yB, des_zB), axis = 0).T

    # Apply roll and pitch limits 
    tiltMax = 15 # Max tilt in degrees
    tiltMaxR = tiltMax*3.1415/180
    roll, pitch, yaw = hf.rotationMatrixToEulerAngles(des_R.T)
    if abs(roll) > tiltMaxR:
        roll = np.sign(roll)*tiltMaxR
    if abs(pitch) > tiltMaxR:
        pitch = np.sign(pitch)*tiltMaxR
    des_R = hf.eulerAnglesToRotationMatrix([roll, pitch, yaw]).T


    if ctrlMode == "attitude":
        # RUN ONLY ATTITUDE CONTROLLER: 
        deslistBtoW = p.getMatrixFromQuaternion(des_orientationW)
        desrotBtoW = np.array([[deslistBtoW[0], deslistBtoW[1], deslistBtoW[2]],
                            [deslistBtoW[3], deslistBtoW[4], deslistBtoW[5]],
                            [deslistBtoW[6], deslistBtoW[7], deslistBtoW[8]]])
        des_R = desrotBtoW

    # rotFixedWing = np.array([[math.cos(1.57), 0, -math.sin(1.57)],
    #                     [0, 1, 0],
    #                     [math.sin(1.57), 0, math.cos(1.57)]])
    
    # des_R = des_R * np.linalg.inv(rotFixedWing)


    eR_mat = .5 * (des_R.T @ rotBtoW - rotBtoW.T @ des_R)


    eR = np.array([[eR_mat[2,1], eR_mat[0,2], eR_mat[1,0]]])

    eW = (LA.inv(rotBtoW) @ angularVelocityW.T - np.array([des_angular_velocityW]).T).T # Was this supposed to be in the Body frame?

    u24 = -K_rotation @ eR.T - K_angularVelocity @ eW.T
    
    u1 = np.clip(u1, 0.0, 100.0)
    u24 = np.clip(u24, -120.0, 120.0)
    
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
        
    # w2Limit = 63202500 #7950RPM
    w2Limit = 77440000 #8800RPM
    # w2Limit = 147440000
    w2 = LA.inv(geo) @ u
    w2 = np.clip(w2,0, w2Limit) #8800 peak RPM -> w2 is angularvelocity^2 -> 8800^(2) =~ 77440000
    w = w2
    
    e = 0,0,0,0

    # if in a tailsitter state, recompute motor velocity and elevon angles as per geoTailSitter and geoTailSitterCtrlSurf geometry.
    if frameState == "fixedwing":
        tempStep = step
        w2 = geoTailSitter @ u
        w2 = np.clip(w2,0, w2Limit)
        w = w2
        e = geoTailSitterCtrlSurf @ u
        eNorm = e / w2Limit
        e = eNorm #Normalize the output because it was designed for propulsion system (omega^2, not elevon deflection)
        # e = np.clip(e, -.5, .5)

    if frameState == "quadrotor":
        w2 = LA.inv(geo) @ u
        w2 = np.clip(w2, 0, w2Limit)
        w = w2
        e = np.array([[0],[0],[0],[0]])

    
    # print("w:", w)

    return w,e

