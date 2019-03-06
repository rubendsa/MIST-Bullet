import numpy as np
import pybullet as p
import time
import pybullet_data

# Initalization Code
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")


# Load MIST-UAV
robotStartPos = [0,0,1]
robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
# robotId = p.loadURDF("MIST.urdf",robotStartPos, robotStartOrientation)
robotId = p.loadURDF("MIST.urdf",robotStartPos, robotStartOrientation)


# for i in range (0,1):
#     print(p.getLinkInfo(robotId, linkIndex=i))

# hingeVelocity = 500 # Multi-rotor
hingeVelocity = -500 # Fixed-wing


for i in range (10000): #Time to run simulation
    p.stepSimulation()
    time.sleep(1./100.)
    p.applyExternalForce(robotId, 
                    0, 
                    [0,0, 100],
                    [0,.25,0],
                    1)
    p.setJointMotorControl2(robotId,
                        jointIndex=0,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=hingeVelocity,
                        force=500)
    p.setJointMotorControl2(robotId,
                        jointIndex=1,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=hingeVelocity,
                        force=500)
    p.setJointMotorControl2(robotId,
                        jointIndex=2,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=hingeVelocity,
                        force=500)
