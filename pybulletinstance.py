import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
import numpy as np 

class PyBulletInstance():

    def __init__(self, GUI=False):
        if(GUI):
            self.client = bc.BulletClient(connection_mode=pybullet.GUI)
            self.viz_delay_id = pybullet.addUserDebugParameter("viz_delay", 0, 0.02, 0.005)
        else:
            self.client = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client.setGravity(0,0,-9.81)
        self.robotID = self.client.loadURDF("./MIST.urdf", [0, 0, 1], pybullet.getQuaternionFromEuler([0,0,0]))
        self.hingeIDs = [0, 1, 2]
        self.ctrlSurfIDs = [9, 7, 5, 3]
        self.propIDs = [10, 8, 6, 4]
        self.currently_used = False #utility to allow odd number of batches relative to total number of envs

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
    def get_viz_delay(self):
        return pybullet.readUserDebugParameter(self.viz_delay_id)

    def set_waypoint_text(self, str, point):
        return pybullet.addUserDebugText(str, point, lifeTime=0)
    
    def remove_waypoint_text(self, val):
        pybullet.removeUserDebugItem(val)
    ###################     Helper functions   #####################
    # Action Vector
    def applyAction(self, actionVector):
        w0, w1, w2, w3, c0, c1, c2, c3, h0, h1, h2 = actionVector

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
        self.client.applyExternalTorque(self.robotID, -1, [0,0, -Mm0], 1) #Torque is assumed to be 1/4 thrust TODO: Update with 2nd order motor model. 
        self.client.applyExternalTorque(self.robotID, 0, [0,0, Mm1], 1) 
        self.client.applyExternalTorque(self.robotID, 1, [0,0, -Mm2], 1) 
        self.client.applyExternalTorque(self.robotID, 2, [0,0, Mm3], 1) 


        # Visual of propeller spinning (not critical)
        self.client.setJointMotorControl2(self.robotID, self.propIDs[0], pybullet.VELOCITY_CONTROL, targetVelocity=w0*10, force=1000) 
        self.client.setJointMotorControl2(self.robotID, self.propIDs[1], pybullet.VELOCITY_CONTROL, targetVelocity=w1*10, force=1000)
        self.client.setJointMotorControl2(self.robotID, self.propIDs[2], pybullet.VELOCITY_CONTROL, targetVelocity=w2*10, force=1000)
        self.client.setJointMotorControl2(self.robotID, self.propIDs[3], pybullet.VELOCITY_CONTROL, targetVelocity=w3*10, force=1000)
        
        # Control surface deflection [rads]
        self.client.setJointMotorControl2(self.robotID, self.ctrlSurfIDs[0], pybullet.POSITION_CONTROL, targetPosition=c0, force=1000)
        self.client.setJointMotorControl2(self.robotID, self.ctrlSurfIDs[1], pybullet.POSITION_CONTROL, targetPosition=c1, force=1000)
        self.client.setJointMotorControl2(self.robotID, self.ctrlSurfIDs[2], pybullet.POSITION_CONTROL, targetPosition=c2, force=1000)
        self.client.setJointMotorControl2(self.robotID, self.ctrlSurfIDs[3], pybullet.POSITION_CONTROL, targetPosition=c3, force=1000)
        
        # Hinge angle [rads]
        self.client.setJointMotorControl2(self.robotID, self.hingeIDs[0], pybullet.POSITION_CONTROL, targetPosition=h0, force=1000)
        self.client.setJointMotorControl2(self.robotID, self.hingeIDs[1], pybullet.POSITION_CONTROL, targetPosition=h1, force=1000)
        self.client.setJointMotorControl2(self.robotID, self.hingeIDs[2], pybullet.POSITION_CONTROL, targetPosition=h2, force=1000)

    # State Vector
    def getUAVState(self):
        a, b, c, d, e, f, g, h = self.client.getLinkState(self.robotID, 0, 1)
        # position = e # x,y,z
        position = self.computeCenterOfMass()
        orientation = f #Quaternion
        velocity = g 
        angular_velocity = h
        return position, orientation, velocity, angular_velocity 
    
    #1D state vector
    def getState(self):
        p, o, v, a_v = self.getUAVState()
        return np.array([*p, *o, *v, *a_v])

    def computeCenterOfMass(self):
        allLinkPositions=[]    #TODO: Refactor this.


        allLinkPositions.append((pybullet.getBasePositionAndOrientation(self.robotID))[0])
        for i in range(0, 3):
            allLinkPositions.append((pybullet.getLinkState(self.robotID, i, 1))[0])

        
        centerOfMass = np.sum(allLinkPositions, axis = 0)/4 #Average x, y, z, of all 4 link CoMs 
        centerOfMass[2] = centerOfMass[2] -.01 # Z intertial offset used in the urdf file
  
        return centerOfMass

    def step(self):
        self.client.stepSimulation()
        # time.sleep(0.001)

    def set_to_pos_and_q(self, pos, q):
        self.client.resetBasePositionAndOrientation(self.robotID, pos, q)

    def reset(self):
        self.set_to_pos_and_q([0, 0, 0], [0, 0, 0, 1])
    
    #TODO put all static methods into util clas
    @staticmethod
    def random_three_vector():
        """
        Generates a random 3D unit vector (direction) with a uniform spherical distribution
        Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
        :return:
        """
        phi = np.random.uniform(0,np.pi*2)
        costheta = np.random.uniform(-1,1)

        theta = np.arccos(costheta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return (x,y,z)

    @staticmethod 
    def random_quaternion():
        return pybullet.getQuaternionFromEuler(PyBulletInstance.random_three_vector())

    #resets to a random start position
    #and orientation
    def reset_random(self):
        pos = list(np.random.rand(3))
        q = PyBulletInstance.random_quaternion()
        self.set_to_pos_and_q(pos, q)
