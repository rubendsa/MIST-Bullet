import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data

class PyBulletInstance():

    def __init__(self, GUI=False):
        if(GUI):
            self.client = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.client = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client.setGravity(0,0,-9.81)
        self.robotID = self.client.loadURDF("C:/Users/user/Transformation/MIST_Bullet/MIST.urdf", [0, 0, 1], pybullet.getQuaternionFromEuler([0,0,0]))
        self.hingeIDs = [0, 1, 2]
        self.ctrlSurfIDs = [9, 7, 5, 3]
        self.propIDs = [10, 8, 6, 4]
        self.currently_used = False #utility to allow odd number of batches relative to total number of envs

    def setHingePosition(self, hingePosition):
        hingeForce = 100
        p.setJointMotorControl2(self.robotID,
                            jointIndex=0,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=hingePosition,
                            force=hingeForce)
        p.setJointMotorControl2(self.robotID,
                            jointIndex=1,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=hingePosition,
                            force=hingeForce)
        p.setJointMotorControl2(self.robotID,
                            jointIndex=2,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=hingePosition,
                            force=hingeForce)

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
        self.client.setJointMotorControl2(self.robotID, self.propIDs[0], pybullet.VELOCITY_CONTROL, targetVelocity=m0*10, force=1000) 
        self.client.setJointMotorControl2(self.robotID, self.propIDs[1], pybullet.VELOCITY_CONTROL, targetVelocity=m1*10, force=1000)
        self.client.setJointMotorControl2(self.robotID, self.propIDs[2], pybullet.VELOCITY_CONTROL, targetVelocity=m2*10, force=1000)
        self.client.setJointMotorControl2(self.robotID, self.propIDs[3], pybullet.VELOCITY_CONTROL, targetVelocity=m3*10, force=1000)
        
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
        position = e # x,y,z
        orientation = f #Quaternion
        velocity = g 
        angular_velocity = h
        return position, orientation, velocity, angular_velocity 


    def step(self):
        self.client.stepSimulation()
        # time.sleep(0.001)

    def set_to_pos_and_q(self, pos, q):
        self.client.resetBasePositionAndOrientation(self.robotID, pos, q)
