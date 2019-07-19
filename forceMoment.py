import wingDynamics as wd

import pybullet as p
import time
import pybullet_data

import wingDynamics as wd
import helperFunctions as hf
from numba import jit



# Apply forces and moments

# @jit(nopython = True)
def applyAction(actionVector, robotId, hingeIds, ctrlSurfIds, propIds):
    # p.connect(p.GUI)
    
    w0, w1, w2, w3, e0, e1, e2, e3, h0, h1, h2 = actionVector

    # Kf = 1 # TODO: Put this in an object. 
    # Km = .1

    Kf = 2.02E-7
    Km = 2.02E-8 # Roughly an order of magnitude less than kf. 

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
    vA, vABody, vNorm, alphar, betar = wd.calcFreeStreamVelocity(robotId, 1)
    eM_0 = 1*(.1*(e0 * Fm0) + (e0 * vNorm))
    eM_1 = 1*(.1*(e1 * Fm1) + (e1 * vNorm))
    eM_2 = 1*(.1*(e2 * Fm2) + (e2 * vNorm))
    eM_3 = 1*(.1*(e3 * Fm3) + (e3 * vNorm))
    # eM_0 = 30*e0
    # eM_1 = 30*e1
    # eM_2 = 30*e2
    # eM_3 = 30*e3
    # print("eM_0", eM_0, "vNorm", vNorm)
    p.applyExternalTorque(robotId, -1, [0,eM_0,0], 2) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 
    p.applyExternalTorque(robotId, 0, [0,eM_1,0], 1) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 
    p.applyExternalTorque(robotId, 1, [0,eM_2,0], 1) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 
    p.applyExternalTorque(robotId, 2, [0,eM_3,0], 1) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 
    # The difference in elevons induces a torque in the z axis for tailsitter. 
    p.applyExternalTorque(robotId, -1, [0,0,(eM_0+eM_1-eM_2-eM_3)], 2) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 
    
    # p.applyExternalTorque(robotId, -1, [0,30*e0,0], 2) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 
    # p.applyExternalTorque(robotId, 0, [0,30*e1,0], 1) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 
    # p.applyExternalTorque(robotId, 1, [0,30*e2,0], 1) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 
    # p.applyExternalTorque(robotId, 2, [0,30*e3,0], 1) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 
    # # The difference in elevons induces a torque in the z axis for tailsitter. 
    # p.applyExternalTorque(robotId, -1, [0,0,300*(e0+e1-e2-e3)], 2) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 

    # p.addUserDebugLine([0,0,0], [e0, 0, 0], [1.0,1.0,1.0], parentObjectUniqueId = 1, parentLinkIndex = ctrlSurfIds[0], lifeTime = .1)
    # p.addUserDebugLine([0,0,0], [e1, 0, 0], [1.0,1.0,1.0], parentObjectUniqueId = 1, parentLinkIndex = ctrlSurfIds[1], lifeTime = .1)
    # p.addUserDebugLine([0,0,0], [e2, 0, 0], [1.0,1.0,1.0], parentObjectUniqueId = 1, parentLinkIndex = ctrlSurfIds[2], lifeTime = .1)
    # p.addUserDebugLine([0,0,0], [e3, 0, 0], [1.0,1.0,1.0], parentObjectUniqueId = 1, parentLinkIndex = ctrlSurfIds[3], lifeTime = .1)



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
    p.setJointMotorControl2(robotId, hingeIds[0], p.POSITION_CONTROL, targetPosition=h0, maxVelocity=8, force=10000)
    p.setJointMotorControl2(robotId, hingeIds[1], p.POSITION_CONTROL, targetPosition=h1, maxVelocity=8, force=10000)
    p.setJointMotorControl2(robotId, hingeIds[2], p.POSITION_CONTROL, targetPosition=h2, maxVelocity=8, force=10000)

    # AERODYNAMICS
    wd.wingDynamics(robotId, -1) # Apply lift, drag, and moments on each wing section according to their pose and velocity. 
    wd.wingDynamics(robotId, 0)
    wd.wingDynamics(robotId, 1)
    wd.wingDynamics(robotId, 2)

    # Visualize Forces
    # hf.visualizeThrottle(w0*Kf, w1*Kf, w2*Kf, w3*Kf)
 

