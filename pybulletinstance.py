import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
import numpy as np 
import utils

import wingDynamics as wd
import helperFunctions as hf

class PyBulletInstance():

    def __init__(self, GUI=False):
        self.GUI = GUI
        if(self.GUI):
            self.client = bc.BulletClient(connection_mode=pybullet.GUI)
            self.viz_delay_id = pybullet.addUserDebugParameter("viz_delay", 0, 0.02, 0.005)
        else:
            self.client = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client.setGravity(0, 0, -9.81)
        self.robotID = self.client.loadURDF("./MIST.urdf", [0, 0, 1], pybullet.getQuaternionFromEuler([0,0,0]))
        self.hingeIDs = [0, 1, 2]
        self.ctrlSurfIDs = [9, 7, 5, 3]
        self.propIDs = [10, 8, 6, 4]
        self.oneTickDebugItems = [] #one-tick debug items are computationally slow
        pybullet.setTimeStep(0.001)

    def setHingePosition(self, hingePosition):
        hingeForce = 100
        pybullet.setJointMotorControl2(self.robotID,
                            jointIndex=0,
                            controlMode=pybullet.POSITION_CONTROL,
                            targetPosition=hingePosition,
                            force=hingeForce)
        pybullet.setJointMotorControl2(self.robotID,
                            jointIndex=1,
                            controlMode=pybullet.POSITION_CONTROL,
                            targetPosition=hingePosition,
                            force=hingeForce)
        pybullet.setJointMotorControl2(self.robotID,
                            jointIndex=2,
                            controlMode=pybullet.POSITION_CONTROL,
                            targetPosition=hingePosition,
                            force=hingeForce)
    
    #######################
    #                     #
    # GUI Debug Functions #
    #                     #
    #######################

    def getVizDelay(self):
        """
        Reads in the debug parameter for vizualation delay
        TODO it may be useful to generalize this function if more debug parameters become necessary
        """
        return pybullet.readUserDebugParameter(self.viz_delay_id)

    def setWayPointText(self, str, point):
        """
        Puts text at a given point, ideally for waypoints
        """
        debugID = pybullet.addUserDebugText(str, point, lifeTime=0)
        return debugID
    
    def addDebugLine(self, start, end, color=[1,0,1], width=3, oneTick=True):
        """
        Draws a line between given points, by default purple
        Remove line by specifying lifeTime or by passing returned ID to removeDebugItem
        oneTick specifies whether to add to the one tick clear queue
        """
        debugID = pybullet.addUserDebugLine(start, end, color, width, lifeTime=0)
        if oneTick:
            self.oneTickDebugItems.append(debugID)
        return debugID
        
    def removeDebugItem(self, val):
        """
        Removes given debug from GUI give ID
        """
        pybullet.removeUserDebugItem(val)
    
    ####################
    #                  #
    # Helper Functions #
    #                  #
    ####################

    def applyAction(self, actionVector):
        """
        Applies an action vector (in Newtons) and applies it to the associated links
        """
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
        self.client.applyExternalForce(self.robotID, -1, [0,0, Fm0], [0,0,0], 1) #Apply m0 force[N] on link0, w.r.t. local frame
        self.client.applyExternalForce(self.robotID, 0, [0,0, Fm1], [0,0,0], 1) #Apply m1 force[N] on link1, w.r.t. local frame
        self.client.applyExternalForce(self.robotID, 1, [0,0, Fm2], [0,0,0], 1) #Apply m2 force[N] on link2, w.r.t. local frame
        self.client.applyExternalForce(self.robotID, 2, [0,0, Fm3], [0,0,0], 1) #Apply m3 force[N] on link3, w.r.t. local frame

        # Torque for each Motor
        self.client.applyExternalTorque(self.robotID, -1, [0,0, -Mm0], 2) #2 is not a typo #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 
        self.client.applyExternalTorque(self.robotID, 0, [0,0, Mm1], 1) 
        self.client.applyExternalTorque(self.robotID, 1, [0,0, -Mm2], 1) 
        self.client.applyExternalTorque(self.robotID, 2, [0,0, Mm3], 1) 

        #Torque for each Elevon
        # vA, vABody, vNorm, alphar, betar = ws.calcFreeStreamVelocity(self.robotID, 1)
        # eM_0 = 30 * e0 
        # eM_1 = 30 * e1 
        # eM_2 = 30 * e2 
        # eM_3 = 30 * e3 
        
        # self.client.applyExternalTorque(self.robotID, -1, [0, eM_0, 0], 2) #Torque is assumed to be 1/4 thrust
        # self.client.applyExternalTorque(self.robotID, 0, [0, eM_1, 0], 1)
        # self.client.applyExternalTorque(self.robotID, 2, [0, eM_2, 0], 1)
        # self.client.applyExternalTorque(self.robotID, 3, [0, eM_3, 0], 1)    

        # self.client.applyExternalTorque(self.robotId, -1, [0,0,(eM_0+eM_1-eM_2-eM_3)], 2)

        # Visual of propeller spinning (not critical)
        self.client.setJointMotorControl2(self.robotID, self.propIDs[0], pybullet.VELOCITY_CONTROL, targetVelocity=w0*10, force=1000) 
        self.client.setJointMotorControl2(self.robotID, self.propIDs[1], pybullet.VELOCITY_CONTROL, targetVelocity=w1*10, force=1000)
        self.client.setJointMotorControl2(self.robotID, self.propIDs[2], pybullet.VELOCITY_CONTROL, targetVelocity=w2*10, force=1000)
        self.client.setJointMotorControl2(self.robotID, self.propIDs[3], pybullet.VELOCITY_CONTROL, targetVelocity=w3*10, force=1000)
        
        # Control surface deflection [rads]
        self.client.setJointMotorControl2(self.robotID, self.ctrlSurfIDs[0], pybullet.POSITION_CONTROL, targetPosition=e0, force=1000)
        self.client.setJointMotorControl2(self.robotID, self.ctrlSurfIDs[1], pybullet.POSITION_CONTROL, targetPosition=e1, force=1000)
        self.client.setJointMotorControl2(self.robotID, self.ctrlSurfIDs[2], pybullet.POSITION_CONTROL, targetPosition=e2, force=1000)
        self.client.setJointMotorControl2(self.robotID, self.ctrlSurfIDs[3], pybullet.POSITION_CONTROL, targetPosition=e3, force=1000)
        
        # Hinge angle [rads]
        self.client.setJointMotorControl2(self.robotID, self.hingeIDs[0], pybullet.POSITION_CONTROL, targetPosition=h0, maxVelocity=4, force=1000)
        self.client.setJointMotorControl2(self.robotID, self.hingeIDs[1], pybullet.POSITION_CONTROL, targetPosition=h1, maxVelocity=4, force=1000)
        self.client.setJointMotorControl2(self.robotID, self.hingeIDs[2], pybullet.POSITION_CONTROL, targetPosition=h2, maxVelocity=4, force=1000)

    def applySingleLinkForce(self, force):
        """
        Applies a force (in Newtons) to a single link
        """
        self.client.applyExternalForce(self.robotID, 2, force, [0, 0, 0], 1)
        
    def getUAVState(self):
        """
        Obtains the UAV state in separate vectors
        """
        a, b, c, d, e, f, g, h = self.client.getLinkState(self.robotID, 0, 1)
        # position = e # x,y,z
        position = self.computeCenterOfMass()
        orientation = f #Quaternion
        velocity = g 
        angular_velocity = h
        return position, orientation, velocity, angular_velocity 
    
    def getState(self):
        """
        Conglomerates state into a 1D vector for convenience
        """
        p, o, v, a_v = self.getUAVState()
        return np.array([*p, *o, *v, *a_v])

    def computeCenterOfMass(self):
        """
        Calculates the center of mass of the frame
        UAV position refers to this point
        """
        allLinkPositions=[]    #TODO: Refactor this.


        allLinkPositions.append((pybullet.getBasePositionAndOrientation(self.robotID))[0])
        for i in range(0, 3):
            allLinkPositions.append((pybullet.getLinkState(self.robotID, i, 1))[0])

        
        centerOfMass = np.sum(allLinkPositions, axis = 0)/4 #Average x, y, z, of all 4 link CoMs 
        centerOfMass[2] = centerOfMass[2] -.01 # Z intertial offset used in the urdf file
  
        return centerOfMass

    def step(self):
        """
        Steps simulation forward once
        """
        if self.GUI:
            for debugID in self.oneTickDebugItems:
                pybullet.removeUserDebugItem(debugID)
            self.oneTickDebugItems = []
        self.client.stepSimulation()

    def set_to_pos_and_q(self, pos, q):
        """
        Sets the frame so a given position and orientation
        """
        self.client.resetBasePositionAndOrientation(self.robotID, pos, q)

    def reset(self):
        """
        Resets to origin with no perturbation
        """
        self.set_to_pos_and_q([0, 0, 0], [0, 0, 0, 1])

    def reset_random(self): 
        """
        Resets to random location around origin with random orientation
        """
        pos = list(np.random.rand(3))
        q = utils.random_quaternion()
        self.set_to_pos_and_q(pos, q)
    
    def reset_upwards(self):
        """
        Resets facing mostly upwards
        """
        pos = list(np.random.rand(3) * 1.0)
        # q = pybullet.getQuaternionFromEuler(utils.random_upwards_rpy())
        q = pybullet.getQuaternionFromEuler(utils.random_45_updwards_rpy())
        self.set_to_pos_and_q(pos, q)
    
    def old_reset_random(self):
        """
        Old method of reseting with the random 3-vector function which is broken(?)
        """
        pos = list(np.random.rand(3))
        q = pybullet.getQuaternionFromEuler(utils.random_three_vector())
        self.set_to_pos_and_q(pos, q)
    
    def apply_random_force(self, magnitude):
        f = utils.random_force(magnitude)
        self.applySingleLinkForce(f)
