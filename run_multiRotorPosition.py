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
import sys
import os
from datetime import datetime
import time
import sys, os

plotting = False

########## START UNCOMMENT FOR WINDOWS ###########
# from ctypes import windll #new
# timeBeginPeriod = windll.winmm.timeBeginPeriod #new
# timeBeginPeriod(1) #new
########## END UNCOMMENT FOR WINDOWS ############


# Enable Video frame-synced video recording
now = datetime.now()
dateTime = now.strftime("_%d_%m_%Y_%H_%M_%S")
savedFile = (os.path.basename(__file__)[:-3])+ dateTime + '.mp4'
physicsClient = p.connect(p.GUI, options="--mp4=\"MultiRotorPosition_final.mp4\" --mp4fps=30")#or p.DIRECT for non-graphical version

# Initalization Code
# physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # Removes the GUI text boxes
p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1) # Forces frame sync

p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.81)
planeId = p.loadURDF("plane.urdf")



# Load MIST-UAV
robotStartPos = [0,0,1]
robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
robotId = p.loadURDF("MIST.urdf",robotStartPos, robotStartOrientation)

hingeIds = [0, 1, 2]
ctrlSurfIds = [9,  7, 5, 3]
propIds = [10, 8, 6, 4]

for i in range(p.getNumJoints(robotId)):
    p.enableJointForceTorqueSensor(robotId, i, 1)


###################     RUN SIMULATION     #####################

if __name__ == "__main__":
    simDelay = .001
    slowSim = False
    timeStep = .001
    ctrlTimeStep = .001

    simTime = 15 # Sim time in [s]
    simSteps = int(simTime/timeStep)
    ctrlStep = int(ctrlTimeStep/timeStep)
    ctrlUpdateStep = 0

    p.setTimeStep(timeStep)
    p.resetBasePositionAndOrientation(robotId, [-20,0,25], p.getQuaternionFromEuler((3.1415/180)*np.array([0,0,0]))) # Staring position of robot
    p.resetBaseVelocity(robotId, [0,0,0], [0,0,0])

    # p.resetBasePositionAndOrientation(robotId, [0,0,10], [.5,0,0,.5]) # Staring position of robot
    # p.resetBaseVelocity(robotId, [0,2,0], [2,0,0])
    # recordedElevonAngles = [[0.] * int(simSteps), [0.] * int(simSteps), [0.] * int(simSteps), [0.] * int(simSteps)]
    # recordedHinge = [[0.] * int(simSteps), [0.] * int(simSteps), [0.] * int(simSteps), [0.] * int(simSteps)]
    # recordedTestHinge = [[0.] * int(simSteps), [0.] * int(simSteps), [0.] * int(simSteps)]
    # recordedvA = [[0.] * int(simSteps), [0.] * int(simSteps), [0.] * int(simSteps)]
    # recordedvABody = [[0.] * int(simSteps), [0.] * int(simSteps), [0.] * int(simSteps)]
    # recorded_alphaR = [[0.] * int(simSteps)]

    eList = np.array([[0, 0, 0, 0]])
    wList = np.array([[0, 0, 0, 0]])
    wdList0 = np.array([[0, 0, [0, 0, 0], 0, 0, 0,[0, 0, 0],[0, 0, 0]]])
    wdList1 = np.array([[0, 0, [0, 0, 0], 0, 0, 0,[0, 0, 0],[0, 0, 0]]])
    wdList2 = np.array([[0, 0, [0, 0, 0], 0, 0, 0,[0, 0, 0],[0, 0, 0]]])
    wdList3 = np.array([[0, 0, [0, 0, 0], 0, 0, 0,[0, 0, 0],[0, 0, 0]]])
    hingeReactionList = np.array([[ 0, 0, 0]])
    hingeTorqueList = np.array([[ 0, 0, 0]])

    motorValStored = np.array([[0,0,0,0]]).T
    elevonValStored = np.array([[0,0,0,0]]).T
    eR_old = 0
    error_position_old = 0
    # fm.applyAction([0, 0, 0, 0, .1, .1, .1, .1, 1.57, 1.57, 1.57], robotId, hingeIds, ctrlSurfIds, propIds) #Example applyAction
            # fm.applyAction([0, 0, 0, 0, .1, .1, .1, .1, 0, 0, 0], robotId, hingeIds, ctrlSurfIds, propIds) #Example applyAction
            # fm.applyAction([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], robotId, hingeIds, ctrlSurfIds, propIds) #Example applyAction

    for i in range (simSteps): #Time to run simulation
        p.stepSimulation()
        if slowSim == True:
            time.sleep(simDelay)
        
        step = i

        # if step in range(int(0/timeStep), int(4/timeStep)):
        #     hingeAngle = 0
        #     frameState = "fixedwing"
        #     des_yawW = 0
        #     des_positionW = [0,10,10] 
        #     des_orientationW = p.getQuaternionFromEuler([0,-1.4,3.1415]) # [roll, pitch, yaw]
        #     des_velocityW = [0,0,0]
        #     des_angular_velocityW = [0,0,0]
        #     robotDesiredPoseWorld = des_positionW, des_orientationW, des_velocityW, des_angular_velocityW, des_yawW 
        #     w, e = ctrl.quadAttitudeControl(robotId, step, robotDesiredPoseWorld, frameState, ctrlMode = "position") # starts with w1 instead of w0 to match the motor geometry of the UAV in the paper.  
        #     w1, w2, w3, w0 = w
        #     e1, e2, e3, e0 = e


        # if step in range(int(0/timeStep), int(6/timeStep)):
        #     hingeAngle = 1.57
        #     frameState = "quadrotor"
            
        #     des_yawW = 0
        #     des_orientationW = p.getQuaternionFromEuler([0,0,3.14]) # [roll, pitch, yaw]
        #     des_positionW = [10,0,10] 
        #     des_velocityW = [0,0,0]
        #     des_angular_velocityW = [0,0,0]

        #     robotDesiredPoseWorld = des_positionW, des_orientationW, des_velocityW, des_angular_velocityW, des_yawW 
        #     w, e = ctrl.quadAttitudeControl(robotId, step, robotDesiredPoseWorld, frameState, ctrlMode = "position") # starts with w1 instead of w0 to match the motor geometry of the UAV in the paper.  
        #     w1, w2, w3, w0 = w
        #     e1, e2, e3, e0 = e


        # if step in range(int(6/timeStep), int(8/timeStep)):
        #     hingeAngle = 0
        #     frameState = "fixedwing"
        #     des_yawW = 0
        #     des_positionW = [0,10,40] 
        #     des_orientationW = p.getQuaternionFromEuler([0,-1.4,3.1415]) # [roll, pitch, yaw]
        #     des_velocityW = [0,0,0]
        #     des_angular_velocityW = [0,0,0]
        #     robotDesiredPoseWorld = des_positionW, des_orientationW, des_velocityW, des_angular_velocityW, des_yawW 
        #     w, e = ctrl.quadAttitudeControl(robotId, step, robotDesiredPoseWorld, frameState, ctrlMode = "position") # starts with w1 instead of w0 to match the motor geometry of the UAV in the paper.  
        #     w1, w2, w3, w0 = w
        #     e1, e2, e3, e0 = e

            
        # if step in range(int(8/timeStep), int(10/timeStep)):
        #     hingeAngle = 1.57
        #     frameState = "quadrotor"
            
        #     des_yawW = 0
        #     des_orientationW = p.getQuaternionFromEuler([0,0,3.14]) # [roll, pitch, yaw]
        #     des_positionW = [0,10,40] 
        #     des_velocityW = [0,0,0]
        #     des_angular_velocityW = [0,0,0]

        #     robotDesiredPoseWorld = des_positionW, des_orientationW, des_velocityW, des_angular_velocityW, des_yawW 
        #     w, e = ctrl.quadAttitudeControl(robotId, step, robotDesiredPoseWorld, frameState, ctrlMode = "position") # starts with w1 instead of w0 to match the motor geometry of the UAV in the paper.  
        #     w1, w2, w3, w0 = w
        #     e1, e2, e3, e0 = e
            

        if step in range(int(0/timeStep), int(20/timeStep)):
            hingeAngle = 1.57
            frameState = "quadrotor"
            
            des_yawW = 0
            des_orientationW = p.getQuaternionFromEuler([0,0,3.14]) # [roll, pitch, yaw]
            des_positionW = [0,0,25] 
            des_velocityW = [0,0,0]
            des_angular_velocityW = [0,0,0]

            robotDesiredPoseWorld = des_positionW, des_orientationW, des_velocityW, des_angular_velocityW, des_yawW 
            
            if step > (ctrlUpdateStep + ctrlStep):
                w, e, eR_old, error_position_old = ctrl.quadAttitudeControl(robotId, step, robotDesiredPoseWorld, frameState, eR_old, error_position_old, ctrlMode = "position") # starts with w1 instead of w0 to match the motor geometry of the UAV in the paper.  

                # for m in range(0, len(w)):
                #     clipThrottle = 1e7
                #     if (np.abs(w[m]-motorValStored[m])) > clipThrottle: 
                #         w[m] = motorValStored[m] + np.sign(w[m]-motorValStored[m]) * clipThrottle
                
                motorValStored = w
                elevonValStored = e
                ctrlUpdateStep = step
        
        w1, w2, w3, w0 = motorValStored
        e1, e2, e3, e0 = elevonValStored

        wing0, wing1, wing2, wing3 = fm.applyAction([w0, w1, w2, w3, e0, e1, e2, e3, hingeAngle, hingeAngle, hingeAngle], robotId, hingeIds, ctrlSurfIds, propIds)
        
        hingeReactionList = np.append(hingeReactionList, np.array([[p.getJointState(robotId, hingeIds[0])[2][5], p.getJointState(robotId, hingeIds[1])[2][5], p.getJointState(robotId, hingeIds[2])[2][5]]]), axis = 0)
        hingeTorqueList = np.append(hingeTorqueList, np.array([[p.getJointState(robotId, hingeIds[0])[3], p.getJointState(robotId, hingeIds[1])[3], p.getJointState(robotId, hingeIds[2])[3]]]), axis = 0)
        wdList0 = np.append(wdList0, [wing0], axis = 0)
        wdList1 = np.append(wdList1, [wing1], axis = 0)
        wdList2 = np.append(wdList2, [wing2], axis = 0)
        wdList3 = np.append(wdList3, [wing3], axis = 0)
        eList = np.append(eList, np.array([e1,e2,e3,e0]).T, axis = 0)
        wList = np.append(wList, (np.array([w1,w2,w3,w0]).T)/8800, axis = 0)
        
    
        if i in range(1, 50000):
            hf.visualizeZoom(robotId, i, 0, 5000, 3, 3, -70, -30) 
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1) # Forces frame sync

        # if i in range(1, 1000):
        #     hf.visualizeZoom(robotId, i, 0, 500, 10, 2)
        
        # if i in range(1000, 1900):
        #     hf.visualizeZoom(robotId, i, 1000, 500, 2, 5)
        
        # if i in range(1900, 3000):
        #     hf.visualizeZoom(robotId, i, 1900, 200, 5, 5)

        # p.resetDebugVisualizerCamera(15, -45, -10, [-10,0,25]) # Camera position (distance, yaw, pitch, focuspoint)

if plotting == True:

    # Create time array in seconds
    timeArray = [0.0]*(simSteps+1)
    for t in range(0,simSteps+1):
        timeArray[t] = t*timeStep

    # Commanded elevon 
    figWidth = 10
    figHeight = 2.5
    plt.figure(1, figsize=(figWidth, figHeight))
    e0, = plt.plot(timeArray, np.array(eList[:,3])*180/np.pi, label = "e0")
    # e1, = plt.plot(timeArray, np.array(eList[:,0])*180/np.pi, label = "e1")
    e2, = plt.plot(timeArray, np.array(eList[:,1])*180/np.pi, label = "e2")
    # e3, = plt.plot(timeArray, np.array(eList[:,2])*180/np.pi, label = "e3")
    plt.legend([e0, e2], ["Elevon 0 and 1 White and Red Section", "Elevon 2 and 3, Green and Blue Section"])
    plt.savefig('elevonDeflection.pdf')
    plt.grid(True)
    plt.xlabel("Time[s]")
    plt.ylabel("Elevon Deflection [deg]")
    plt.xticks(np.arange(0, simTime+.1, 1)) 
    plt.yticks(np.arange(-20, 20, 5)) 
    plt.tight_layout()
    plt.savefig('../../simFigs/'+(os.path.basename(__file__)[:-3])+'commandedElevon.pdf')

    ########### Angular velocity on propellers
    plt.figure(2, figsize=(figWidth, figHeight))
    plt.grid(True)
    plt.xlabel("Time[s]")
    plt.ylabel("Motor [RPM]")
    plt.xticks(np.arange(0, simTime+.1, 1)) 
    plt.yticks(np.arange(0, 9000.1, 1000)) 
    m0, = plt.plot(timeArray, wList[:,3], 'k', label = "m0")
    m1, = plt.plot(timeArray, wList[:,0], 'r', label = "m1")
    m2, = plt.plot(timeArray, wList[:,1], 'g', label = "m2")
    m3, = plt.plot(timeArray, wList[:,2], 'b', label = "m3")
    plt.legend([m0, m1, m2, m3], ["Motor 0, White Section", "Motor 1, Red Section", "Motor 2, Green Section", "Motor 3, Blue Section"])
    plt.tight_layout()
    plt.savefig('../../simFigs/'+(os.path.basename(__file__)[:-3])+'motorRPM.pdf')

    ############# Power estimate model
    plt.figure(3, figsize=(figWidth, figHeight))
    plt.grid(True)
    plt.xlabel("Time[s]")
    plt.ylabel("Motor Power [W]")
    plt.xticks(np.arange(0, simTime+.1, 1)) 
    plt.yticks(np.arange(0, 1500.1, 100)) 
    powerM0 = hf.power_required_mt2814(wList[:,3])
    powerM1 = hf.power_required_mt2814(wList[:,0])
    powerM2 = hf.power_required_mt2814(wList[:,1])
    powerM3 = hf.power_required_mt2814(wList[:,2])
    totalPower = powerM0 + powerM1 + powerM2 + powerM3
    m0, = plt.plot(timeArray, powerM0, 'k', label = "m0")
    m1, = plt.plot(timeArray, powerM1, 'r', label = "m1")
    m2, = plt.plot(timeArray, powerM2, 'g', label = "m2")
    m3, = plt.plot(timeArray, powerM3, 'b', label = "m3")
    totalPower, = plt.plot(timeArray, totalPower, '--k', label = "totalPower")
    plt.legend([m0, m1, m2, m3, totalPower], ["Motor 0, White Section", "Motor 1, Red Section", "Motor 2, Green Section", "Motor 3, Blue Section", "Total Power"])
    plt.tight_layout()
    plt.savefig('../../simFigs/'+(os.path.basename(__file__)[:-3])+'motorPower.pdf')



    # # alphar
    # plt.figure(4, figsize=(10,3))
    # plt.plot(wdList0[1:,0], 'k')
    # plt.plot(wdList1[1:,0], 'r')
    # plt.plot(wdList2[1:,0], 'g')
    # plt.plot(wdList3[1:,0], 'b')

    # # vNorm
    # plt.figure(5, figsize=(10,3))
    # plt.plot(wdList0[1:,1], 'k')
    # plt.plot(wdList1[1:,1], 'r')
    # plt.plot(wdList2[1:,1], 'g')
    # plt.plot(wdList3[1:,1], 'b')

    # # Rel Fx
    # plt.figure(6, figsize=(10,3))
    # plt.plot(wdList0[1:,4], 'k')
    # plt.plot(wdList1[1:,4], 'r')
    # plt.plot(wdList2[1:,4], 'g')
    # plt.plot(wdList3[1:,4], 'b')

    # # Rel Fz
    # plt.figure(7, figsize=(10,3))
    # plt.plot(wdList0[1:,5], 'k')
    # plt.plot(wdList1[1:,5], 'r')
    # plt.plot(wdList2[1:,5], 'g')
    # plt.plot(wdList3[1:,5], 'b')

    # orientationWing : Roll, pitch, yaw
    plt.figure(8, figsize=(figWidth, figHeight))
    plt.grid(True)
    plt.xlabel("Time[s]")
    plt.ylabel("Angle [deg]")
    plt.xticks(np.arange(0, simTime+.1, 1)) 
    plt.yticks(np.arange(-50, 50, 5)) 
    roll, = plt.plot(timeArray[1:], np.array([x[0] for x in (wdList1[1:,6])])*180/np.pi, 'r', label = "roll") # Roll, so pythonic
    pitch, = plt.plot(timeArray[1:], np.array([x[1] for x in (wdList1[1:,6])])*180/np.pi, 'g', label = "pitch") # Pitch
    yaw, = plt.plot(timeArray[1:], np.array([x[2] for x in (wdList1[1:,6])])*180/np.pi, 'b', label = "yaw") # Yaw
    plt.legend([roll, pitch, yaw], ["Roll (Red Wing Section)", "Pitch (Red Wing Section)", "Yaw (Red Wing Section)"])
    plt.tight_layout()
    plt.savefig('../../simFigs/'+(os.path.basename(__file__)[:-3])+'rollPitchYaw.pdf')

    # positionWing : x,y,z
    plt.figure(9, figsize=(figWidth, figHeight))
    plt.grid(True)
    plt.xlabel("Time[s]")
    plt.ylabel("Position [m]")
    plt.xticks(np.arange(0, simTime+.1, 1)) 
    plt.yticks(np.arange(-50, 50, 5)) 
    x, = plt.plot(timeArray[1:], np.array([x[0] for x in (wdList1[1:,7])]), 'r', label = "x") # x, 
    y, = plt.plot(timeArray[1:], np.array([x[1] for x in (wdList1[1:,7])]), 'g', label = "y") # y
    z, = plt.plot(timeArray[1:], np.array([x[2] for x in (wdList1[1:,7])]), 'b', label = "z") # z
    plt.legend([x, y, z], ["x", "y", "z"])
    plt.tight_layout()
    plt.savefig('../../simFigs/'+(os.path.basename(__file__)[:-3])+'position.pdf')


    plt.figure(10, figsize=(figWidth, figHeight))
    plt.grid(True)
    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    # plt.xticks(np.arange(0, simTime+.1, 1)) 
    plt.xticks(np.arange(-20, 10, 2)) 
    plt.xlim([-20,10])
    plt.yticks(np.arange(20, 30, 1))
    plt.ylim([20,30])
    zPos = np.array([x[2] for x in (wdList1[1:,7])])
    xPos = np.array([x[0] for x in (wdList1[1:,7])])
    xPosition, = plt.plot(xPos, zPos, '--k', label = "x") # x, 
    # y, = plt.plot(timeArray[1:], np.array([x[1] for x in (wdList1[1:,7])]), 'g', label = "y") # y
    # z, = plt.plot(timeArray[1:], np.array([x[2] for x in (wdList1[1:,7])]), 'b', label = "z") # z
    # plt.legend([x, y, z], ["x", "y", "z"])
    targetPos, = plt.plot(0, 25, '*b')
    plt.legend([xPosition, targetPos], ["UAV Position", "Desired Position"])
    plt.tight_layout()
    plt.savefig('../../simFigs/'+(os.path.basename(__file__)[:-3])+'XZposition.pdf')

    # Hinge Reaction torque
    # plt.figure(9, figsize=(10,3))
    # plt.plot(hingeReactionList[:,0], 'k')
    # plt.plot(hingeReactionList[:,1], 'r')
    # plt.plot(hingeReactionList[:,2], 'g')

    # Hinge torque
    plt.figure(11, figsize=(figWidth, figHeight))
    h0, = plt.plot(timeArray, hingeTorqueList[:,0], 'k', label = "h0")
    h1, = plt.plot(timeArray, hingeTorqueList[:,1], 'r', label = "h1")
    h2, = plt.plot(timeArray, hingeTorqueList[:,2], 'g', label = "h2")
    plt.legend([h0, h1, h2], ["Hinge 0", "Hinge 1", "Hinge 2"])
    plt.savefig('hingeTorque.pdf')
    plt.grid(True)
    plt.xlabel("Time[s]")
    plt.ylabel("Hinge Torque [Nm]")
    plt.xticks(np.arange(0, simTime+.1, 1)) 
    plt.yticks(np.arange(-30, 50, 10)) 
    plt.tight_layout()
    plt.savefig('../../simFigs/'+(os.path.basename(__file__)[:-3])+'hingeTorque.pdf')

    # Roll, pitch, yaw



    plt.show()
# plt.savefig('figure1.pdf')


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



