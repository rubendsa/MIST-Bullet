# Initial settings and configuration

import pybullet as p
import time
import pybullet_data

class simSetup:
    def initSettings(self):
        # Initalization Code
        physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # Removes the GUI text boxes
        # physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-9.81)
        self.planeId = p.loadURDF("plane.urdf")


        # Load MIST-UAV
        self.robotStartPos = [0,0,1]
        self.robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
        # robotId = p.loadURDF("MIST.urdf",robotStartPos, robotStartOrientation)
        self.robotId = p.loadURDF("MIST.urdf",self.robotStartPos, self.robotStartOrientation)
        # print(robotId)

        self.hingeIds = [0, 1, 2]
        self.ctrlSurfIds = [9,  7, 5, 3]
        self.propIds = [10, 8, 6, 4]
