import numpy as np
from numpy import linalg as LA
import pybullet as p
import time
import pybullet_data


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
print(robotId)

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
    

    print("hingeposition", hingePosition)



def quadAttitudeControl(robotId, robotDesiredPoseWorld):

    # Set Gains and Parameters TODO: Move this out
    K_position = np.eye(3) * np.array([[20, 20, 20]])
    # print(K_position) 
    K_velocity = np.eye(3) * np.array([[20, 20, 20]])
    K_rotation = np.eye(3) * np.array([[20, 20, 20]])
    K_angularVelocity = np.eye(3) * np.array([[20, 20, 20]])

    kf = 2
    km = 2
    L = .5

    mass = 4 #Mass in [kg]
    gravity = 9.81

    # Get pose
    a, b, c, d, e, f, g, h = p.getLinkState(robotId, 0, 1) #getLinkState() has a bug for getting the parent link index (-1). Use 0 for now
    positionW = a
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
    print("zB", zB)
    print("des_F", des_F)

    u1 = des_F.T @ zB

    des_zB = des_F / LA.norm(des_F)

    des_xC = np.array([[np.cos(des_yaw), np.sin(des_yaw), 0]]).T

    # print(des_zB)
    # print(des_xC)
    # print(np.cross(des_zB.T, des_xC.T))
    des_yB = (np.cross(des_zB.T, des_xC.T) / LA.norm(np.cross(des_zB.T, des_xC.T))).T
    # print(LA.norm(np.cross(des_zB.T, des_xC.T)))
    # print((np.cross(des_zB.T, des_xC.T) / LA.norm(np.cross(des_zB.T, des_xC.T))).T)
    des_xB = np.cross(des_yB.T, des_zB.T)

    print("des_xB.T", des_xB.T)
    print(des_yB)
    print(des_zB)
    des_R = np.concatenate((des_xB.T, des_yB, des_zB), axis = 1)

    # print(des_R.T)
    # print(rotBtoW)

    print("des_R.T", des_R.T)
    eR_mat = .5 * (des_R.T @ rotBtoW - rotBtoW.T @ des_R)

    eR = np.array([[eR_mat[2,1], eR_mat[0,2], eR_mat[1,0]]])

    print("er_mat", eR_mat)
    print("er", eR)
    # print(angularVelocityW)
    # print(des_angular_velocityW)

    eW = angularVelocityW - np.array([des_angular_velocityW])

    print("-K_rotation @ eR.T", -K_rotation @ eR.T)
    print("- K_angularVelocity @ eW.T", - K_angularVelocity @ eW.T)
    u24 = -K_rotation @ eR.T - K_angularVelocity @ eW.T

    # print("u1:", u1)
    # print("des_F", des_F)
    # print("zB", zB)
    print("eR", eR)
    print("k_Rot", -K_rotation @ eR.T)

    print("u24:", u24)
    u = np.concatenate((u1, u24))

    print("u:", u)

    geo = np.array([[kf, kf, kf, kf],
                    [0, kf*L, 0, -kf*L],
                    [-kf*L, 0, kf*L, 0],
                    [km, -km, km, -km]])
    
    # Compute angular velocities
    print("LA.inv(geo):", LA.inv(geo))
    print("u:", u)

    w = LA.inv(geo) @ u

    return w

###################     Helper functions   #####################
# Action Vector
def applyAction(actionVector, robotId=robotId):
    m0, m1, m2, m3, c0, c1, c2, c3, h0, h1, h2 = actionVector

    # Thrust for each Motor
    p.applyExternalForce(robotId, -1, [0,0, m0], [0,0,0], 1) #Apply m0 force[N] on link0, w.r.t. local frame
    p.applyExternalForce(robotId, 0, [0,0, m1], [0,0,0], 1) #Apply m1 force[N] on link1, w.r.t. local frame
    p.applyExternalForce(robotId, 1, [0,0, m2], [0,0,0], 1) #Apply m2 force[N] on link2, w.r.t. local frame
    p.applyExternalForce(robotId, 2, [0,0, m3], [0,0,0], 1) #Apply m3 force[N] on link3, w.r.t. local frame

    # Torque for each Motor
    p.applyExternalTorque(robotId, -1, [0,0,m0/4], 1) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 
    p.applyExternalTorque(robotId, 0, [0,0, -m1/4], 1) 
    p.applyExternalTorque(robotId, 1, [0,0, m2/4], 1) 
    p.applyExternalTorque(robotId, 2, [0,0, -m3/4], 1) 


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
def getUAVState(robotId=robotId):
    a, b, c, d, e, f, g, h = p.getLinkState(robotId, -1, 1)
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

###################     RUN SIMULATION     #####################

# tailsitterAttitudeControl()
if __name__ == "__main__":
    print("numjoints: ", p.getNumJoints(robotId))
    simTime = 1000000
    simDelay = 0.001

    # p.addUserDebugLine([0,0,0], [0, 0, 1.0], [1.0,1.0,1.0], parentObjectUniqueId = 1, parentLinkIndex = -1)
    # p.addUserDebugLine([0,0,0], [0, 0, 1.0], [1.0,0.0,0.0], parentObjectUniqueId = 1, parentLinkIndex = 0)
    # p.addUserDebugLine([0,0,0], [0, 0, 1.0], [0.0,1.0,0.0], parentObjectUniqueId = 1, parentLinkIndex = 1)
    # p.addUserDebugLine([0,0,0], [0, 0, 1.0], [0.0,0.0,1.0], parentObjectUniqueId = 1, parentLinkIndex = 2)

    for i in range (simTime): #Time to run simulation
        p.stepSimulation()
        time.sleep(simDelay)

        # applyAction([0, 0, 0, 0, .9, .9, .9, .9, 1.57, 1.57, 1.57], robotId) #Example applyAction
        # applyAction([0, 0, 0, 0, .3, .1, -.1, -.3, 0, 0, 0], robotId) #Example applyAction

        # if i>500:
        #     applyAction([0, 0, 0, 0, -1, -1, -1, -1, 1.2, 1.2, 1.2], robotId) #Example applyAction
            # applyAction([00, 00, 000, 000, 0, 0, 0, 0, 0, 0, .9], robotId) #Example applyAction

        ##### Testing attitude and position controller:
        des_positionW = [0,0,10]
        des_orientationW = [0, 0, 0, 1] # [x, y, z, w] quaternion
        des_velocityW = [0,0,0]
        des_angular_velocityW = [0,0,0]

        robotDesiredPoseWorld = des_positionW, des_orientationW, des_velocityW, des_angular_velocityW 

        m2, m3, m0, m1 = quadAttitudeControl(robotId, robotDesiredPoseWorld) 

        applyAction([m0, m1, m2, m3, -1, -1, -1, -1, 1.57, 1.57, 1.57], robotId)

         
        # print("linkid", p.getLinkState(robotId, 0, 1))


