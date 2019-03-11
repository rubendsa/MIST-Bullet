# # Py Bullet Example:http://alexanderfabisch.github.io/pybullet.html

# import pybullet
# import pybullet_data
# import os
# os.system("git clone https://github.com/ros-industrial/kuka_experimental.git")


# pybullet.connect(pybullet.GUI)
# # To connect without GUI: pybullet.connect(pybullet.DIRECT)

# # Reset simulation
# # pybullet.resetSimulation()

# robot = pybullet.loadURDF("kuka_experimental/kuka_kr210_support/urdf/kr210l150.urdf")

# # position, orientation = pybullet.getBasePositionAndOrientation(robot)
# # orientation

# # pybullet.getNumJoints(robot)
# # pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
import numpy as np
import pybullet as p
import time
import pybullet_data


physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
# boxId = p.loadURDF("~/Quadrotor/quadrotor.urdf",cubeStartPos, cubeStartOrientation)
boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)

maxForce = 500
for i in range (0,15):
    print(p.getJointInfo(boxId, jointIndex=i))



# p.setJointMotorControl2(boxId,
#                         jointIndex=2,
#                         controlMode=p.VELOCITY_CONTROL,
#                         targetVelocity=500,
#                         force=maxForce)

# p.setJointMotorControl2(boxId,
#                         jointIndex=3,
#                         controlMode=p.VELOCITY_CONTROL,
#                         targetVelocity=500,
#                         force=maxForce)

# p.setJointMotorControl2(boxId,
#                         jointIndex=6,
#                         controlMode=p.VELOCITY_CONTROL,
#                         targetVelocity=500,
#                         force=maxForce)

# p.setJointMotorControl2(boxId,
#                         jointIndex=7,
#                         controlMode=p.VELOCITY_CONTROL,
#                         targetVelocity=500,
#                         force=maxForce)

# p.resetDebugVisualizerCamera(cameraDistance = 10.,
#                              cameraYaw = -45.,
#                              cameraPitch = -45.,
#                              cameraTargetPosition = [-10., -10., 10.])

# p.applyExternalForce(boxId, 
#                     -1, 
#                     [40000,0, 0],
#                     [0,0,0],
#                     1)

for i in range (10000): #Time to run simulation
    p.stepSimulation()
    # time.sleep(0.0001)
    print(i)
    p.applyExternalForce(boxId, 
                    -1, 
                    [0,0, 700],
                    [0,0,0],
                    1)


    # p.setRealTimeSimulation(1)
    # print("num of joints", p.getBasePositionAndOrientation(boxId))
    # print("boxId", boxId)


cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()
