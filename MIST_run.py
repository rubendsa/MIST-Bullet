import numpy as np
import pybullet as p
import time
import pybullet_data
from ctypes import windll #new

timeBeginPeriod = windll.winmm.timeBeginPeriod #new
timeBeginPeriod(1) #new

# Initalization Code
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
# physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.81)
# planeId = p.loadURDF("plane.urdf")



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

###################     RUN SIMULATION     #####################

# tailsitterAttitudeControl()
if __name__ == "__main__":
    print("numjoints: ", p.getNumJoints(robotId))
    simTime = 1000000
    simDelay = 0.003

    p.addUserDebugLine([0,0,0], [0, 0, 1.0], [1.0,1.0,1.0], parentObjectUniqueId = 1, parentLinkIndex = -1)
    p.addUserDebugLine([0,0,0], [0, 0, 1.0], [1.0,0.0,0.0], parentObjectUniqueId = 1, parentLinkIndex = 0)
    p.addUserDebugLine([0,0,0], [0, 0, 1.0], [0.0,1.0,0.0], parentObjectUniqueId = 1, parentLinkIndex = 1)
    p.addUserDebugLine([0,0,0], [0, 0, 1.0], [0.0,0.0,1.0], parentObjectUniqueId = 1, parentLinkIndex = 2)

    for i in range (simTime): #Time to run simulation
        p.stepSimulation()
        time.sleep(simDelay)

        # applyAction([0, 0, 0, 0, .9, .9, .9, .9, 1.57, 1.57, 1.57], robotId) #Example applyAction
        applyAction([0, 0, 0, 0, .3, .1, -.1, -.3, 0, 0, 0], robotId) #Example applyAction

        if i>500:
            applyAction([300, 250, 300, 250, 0, 0, 0, 0, 1.57, 1.57, 1.57], robotId) #Example applyAction
            # applyAction([00, 00, 000, 000, 0, 0, 0, 0, 0, 0, .9], robotId) #Example applyAction

    
    # print(getUAVState(robotId))
    


