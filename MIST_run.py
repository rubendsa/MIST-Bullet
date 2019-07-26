import numpy as np
from numpy import linalg as LA
from numpy import loadtxt
import pybullet as p
import time
import pybullet_data
import matplotlib.pyplot as plt
import math

# Functions for Transformer UAV dynamics and control
import wingDynamics
import forceMoment as fm
import controllers as ctrl
import record
import helperFunctions as hf



########## START UNCOMMENT FOR WINDOWS ###########
# from ctypes import windll #new
# timeBeginPeriod = windll.winmm.timeBeginPeriod #new
# timeBeginPeriod(1) #new
########## END UNCOMMENT FOR WINDOWS ############

# Initalization Code
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # Removes the GUI text boxes
# physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.81)
planeId = p.loadURDF("plane.urdf")
# p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "dropTest_vel5_test.mp4")	

# Load MIST-UAV
robotStartPos = [0,0,1]
robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
robotId = p.loadURDF("MIST.urdf",robotStartPos, robotStartOrientation)

hingeIds = [0, 1, 2]
ctrlSurfIds = [9,  7, 5, 3]
propIds = [10, 8, 6, 4]


###################     RUN SIMULATION     #####################

if __name__ == "__main__":
    simTime = 10000
    simDelay = .01
    p.setTimeStep(.001)
    p.resetBasePositionAndOrientation(robotId, [-20,0,40], p.getQuaternionFromEuler((3.1415/180)*np.array([0,0,0]))) # Staring position of robot
    p.resetBaseVelocity(robotId, [0,0,0], [0,0,0])

    # p.resetBasePositionAndOrientation(robotId, [0,0,10], [.5,0,0,.5]) # Staring position of robot
    # p.resetBaseVelocity(robotId, [0,2,0], [2,0,0])
    # recordedElevonAngles = [[0.] * int(simTime), [0.] * int(simTime), [0.] * int(simTime), [0.] * int(simTime)]
    # recordedHinge = [[0.] * int(simTime), [0.] * int(simTime), [0.] * int(simTime), [0.] * int(simTime)]
    # recordedTestHinge = [[0.] * int(simTime), [0.] * int(simTime), [0.] * int(simTime)]
    # recordedvA = [[0.] * int(simTime), [0.] * int(simTime), [0.] * int(simTime)]
    # recordedvABody = [[0.] * int(simTime), [0.] * int(simTime), [0.] * int(simTime)]
    # recorded_alphaR = [[0.] * int(simTime)]

    eList = np.array([[None, None, None, None]])
    wList = np.array([[None, None, None, None]])
    wdList0 = np.array([[None, None, [None, None, None], None]])
    wdList1 = np.array([[None, None, [None, None, None], None]])
    wdList2 = np.array([[None, None, [None, None, None], None]])
    wdList3 = np.array([[None, None, [None, None, None], None]])


    for i in range (simTime): #Time to run simulation
        p.stepSimulation()
        # time.sleep(simDelay)

        # fm.applyAction([0, 0, 0, 0, .1, .1, .1, .1, 1.57, 1.57, 1.57], robotId, hingeIds, ctrlSurfIds, propIds) #Example applyAction
        # fm.applyAction([0, 0, 0, 0, .1, .1, .1, .1, 0, 0, 0], robotId, hingeIds, ctrlSurfIds, propIds) #Example applyAction
        # fm.applyAction([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], robotId, hingeIds, ctrlSurfIds, propIds) #Example applyAction
        
        step = i

        if step in range(0, 5000):
            hingeAngle = 0
            frameState = "fixedwing"
                
            des_yawW = 0
            des_positionW = [0,10,40] 
            des_orientationW = p.getQuaternionFromEuler([0,-1.4,3.1415]) # [roll, pitch, yaw]
            des_velocityW = [0,0,0]
            des_angular_velocityW = [0,0,0]
            
            robotDesiredPoseWorld = des_positionW, des_orientationW, des_velocityW, des_angular_velocityW, des_yawW 
            w, e = ctrl.quadAttitudeControl(robotId, step, robotDesiredPoseWorld, frameState, ctrlMode = "attitude") # starts with w1 instead of w0 to match the motor geometry of the UAV in the paper.  
            w1, w2, w3, w0 = w
            e1, e2, e3, e0 = e


        # if step in range(1000, 2000):
        #     hingeAngle = 0
        #     frameState = "fixedwing"
            
        #     des_yawW = 0
        #     des_orientationW = p.getQuaternionFromEuler([0,-1.4,3.1415]) # [roll, pitch, yaw]
        #     des_positionW = [20,10,10]
        #     des_velocityW = [0,0,0]
        #     des_angular_velocityW = [0,0,0]
            
        #     robotDesiredPoseWorld = des_positionW, des_orientationW, des_velocityW, des_angular_velocityW, des_yawW 
        #     w, e = ctrl.quadAttitudeControl(robotId, step, robotDesiredPoseWorld, frameState, ctrlMode = "attitude") # starts with w1 instead of w0 to match the motor geometry of the UAV in the paper.  
        #     w1, w2, w3, w0 = w
        #     e1, e2, e3, e0 = e

        if step in range(5000, 10000):
            hingeAngle = 1.57
            frameState = "quadrotor"
            
            des_yawW = 0
            des_orientationW = p.getQuaternionFromEuler([0,0,3.14]) # [roll, pitch, yaw]
            des_positionW = [0,10,50]
            des_velocityW = [0,0,0]
            des_angular_velocityW = [0,0,0]

            robotDesiredPoseWorld = des_positionW, des_orientationW, des_velocityW, des_angular_velocityW, des_yawW 
            w, e = ctrl.quadAttitudeControl(robotId, step, robotDesiredPoseWorld, frameState, ctrlMode = "position") # starts with w1 instead of w0 to match the motor geometry of the UAV in the paper.  
            w1, w2, w3, w0 = w
            e1, e2, e3, e0 = e

            
            
            

        
        
        

        
        

        
        # vA = np.array(p.getLinkState(robotId, 0, 1)[6]) # Calculate Air-relative velocity vector Va
        # # print("vA:", vA)
        # listMat = p.getMatrixFromQuaternion(p.getLinkState(robotId, 0, 1)[1])
        # rotWtoB = np.array([[listMat[0], listMat[1], listMat[2]],
        #             [listMat[3], listMat[4], listMat[5]],
        #             [listMat[6], listMat[7], listMat[8]]])
        # vABody = np.linalg.inv(rotWtoB) @ vA

        # vNorm = (vABody.T @ vABody)**(1/2) # Magnitude of vehicle velocity
        # alphaR = math.atan2(vABody[0],abs(vABody[2]))

        # recordedvA[0][i] = vA[0]
        # recordedvA[1][i] = vA[1]
        # recordedvA[2][i] = vA[2]
        # recordedvABody[0][i] = vABody[0]
        # recordedvABody[1][i] = vABody[1]
        # recordedvABody[2][i] = vABody[2]
        # recordedTestHinge[0][i] = p.getEulerFromQuaternion(p.getLinkState(robotId, ctrlSurfIds[0])[1])[0]
        # recordedTestHinge[1][i] = p.getEulerFromQuaternion(p.getLinkState(robotId, ctrlSurfIds[0])[1])[1]
        # recordedTestHinge[2][i] = p.getEulerFromQuaternion(p.getLinkState(robotId, ctrlSurfIds[0])[1])[2]
        # recorded_alphaR[0][i] = alphaR * 180/3.1415
        
        # print("euler angles:",p.getEulerFromQuaternion(p.getLinkState(robotId, ctrlSurfIds[0])[1]))
        # p.applyExternalForce(robotId, 1, [0,1,0], [0,0,0], 2) #Apply m0 force[N] on link0, w.r.t. local frame
        # print("linkframe", p.WORLD_FRAME)

        

        
        wing0, wing1, wing2, wing3 = fm.applyAction([w0, w1, w2, w3, e0, e1, e2, e3, hingeAngle, hingeAngle, hingeAngle], robotId, hingeIds, ctrlSurfIds, propIds)
        # print("wings", [wing0])
        # print("wings", [wdList0])
        wdList0 = np.append(wdList0, [wing0], axis = 0)
        wdList1 = np.append(wdList1, [wing1], axis = 0)
        wdList2 = np.append(wdList2, [wing2], axis = 0)
        wdList3 = np.append(wdList3, [wing3], axis = 0)
        eList = np.append(eList, np.array([e1,e2,e3,e0]).T, axis = 0)
        wList = np.append(wList, (np.array([w1,w2,w3,w0]).T)/147440000, axis = 0)
        # print("wdlist0", wdList0)

        # print("wdList", wdList)
    
        if i in range(1, 50000):
            hf.visualizeZoom(robotId, i, 0, 5000, 5, 5, -70, -30) 

        # if i in range(1, 1000):
        #     hf.visualizeZoom(robotId, i, 0, 500, 10, 2)
        
        # if i in range(1000, 1900):
        #     hf.visualizeZoom(robotId, i, 1000, 500, 2, 5)
        
        # if i in range(1900, 3000):
        #     hf.visualizeZoom(robotId, i, 1900, 200, 5, 5)
# print("wList", wList[:,1])

# Commanded elevon 
plt.figure(1)
plt.plot(eList[:,0])
plt.plot(eList[:,1])
plt.plot(eList[:,2])
plt.plot(eList[:,3])

# Angular velocity on propellers
plt.figure(2)
plt.plot(wList[:,0])
plt.plot(wList[:,1])
plt.plot(wList[:,2])
plt.plot(wList[:,3])

# print("wdTranspose", wdList[:,2][1:].T)
# print("wdTransposeCon", np.concatenate(wdList[:,2][1:].T, axis = 0)[:,0])
# # print("wdListagain", np.concatenate(wdList[:,1].T, axis = 0))
# print("wdListagain", wdList0[1:,0])

print("wdList", wdList0[1:,0])
# alphar
plt.figure(3)
plt.plot(wdList0[1:,0], 'k')
plt.plot(wdList1[1:,0], 'r')
plt.plot(wdList2[1:,0], 'g')
plt.plot(wdList3[1:,0], 'b')

# vNorm
plt.figure(4)
plt.plot(wdList0[1:,1], 'k')
plt.plot(wdList1[1:,1], 'r')
plt.plot(wdList2[1:,1], 'g')
plt.plot(wdList3[1:,1], 'b')


plt.show()

    # print("0 contents",recordedTestHinge[:][0])
    # while(1):
    #     plt.figure(1)
    #     plt.plot(recordedTestHinge[:][0])
    #     plt.plot(recordedTestHinge[:][1])
    #     plt.plot(recordedTestHinge[:][2])
        
    #     plt.figure(2)
    #     plt.plot(recordedvA[:][0], 'r--')
    #     plt.plot(recordedvA[:][1], 'g--')
    #     plt.plot(recordedvA[:][2], 'b--')
    #     plt.plot(recordedvABody[:][0],'r')
    #     plt.plot(recordedvABody[:][1],'g')
    #     plt.plot(recordedvABody[:][2],'b')

    #     plt.figure(3)
    #     plt.plot(recorded_alphaR[:][0], 'r')


    #     plt.show()



