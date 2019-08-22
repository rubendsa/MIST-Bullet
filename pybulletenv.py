import pybullet 
import pybullet_utils.bullet_client as bullet_client
import pybullet_data 

import numpy as np 
import gym 

from gym import error, spaces
import utils 

'''
Gym Compatible
'''
class PyBulletEnvironment(gym.Env):

    def __init__(self, GUI=False):
        """
        Everything is passed through the local bullet client (self.bc)
        """
        if(GUI):
            self.bc = bullet_client.BulletClient(connection_mode=pybullet.GUI) 
        else:
            self.bc = bullet_client.BulletClient(connection_mode=pybullet.DIRECT) 
        
        self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.bc.setGravity(0, 0, -9.81)
        self.robotID = self.bc.loadURDF("./MIST.urdf", [0, 0, 1], [0, 0, 0, 1])
        self.hingeIDs = [0, 1, 2]
        self.ctrlSurfIDs = [9, 7, 5, 3]
        self.propIDs = [10, 8, 6, 4]

        self.action_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32)

    # @property
    # def action_space(self):
    #     """
    #     Returns a Space object
    #     """
    #     raise self.action_space

    # @property
    # def observation_space(self):
    #     """
    #     Returns a Space object
    #     """
    #     raise self.observation_space

    def set_hinge_pos(self, hinge_pos):
        hinge_f = 100
        self.bc.setJointMotorControl2(self.robotID,
                            jointIndex=0,
                            controlMode=self.bc.POSITION_CONTROL,
                            targetPosition=hinge_pos,
                            force=hingeForce)
        self.bc.setJointMotorControl2(self.robotID,
                            jointIndex=1,
                            controlMode=self.bc.POSITION_CONTROL,
                            targetPosition=hinge_pos,
                            force=hingeForce)
        self.bc.setJointMotorControl2(self.robotID,
                            jointIndex=2,
                            controlMode=self.bc.POSITION_CONTROL,
                            targetPosition=hinge_pos,
                            force=hingeForce)

    def sim_step(self, action):
        w = action[0:4]
        # c = action[4:8]
        # h = action[8:]
        c = [0, 0, 0, 0]
        h = [1.57, 1.57, 1.57]

        Kf = 1
        Km = .1

        Fm = w * Kf 
        Mm = w * Km
        Mm[0] *= -1
        Mm[2] *= -1

        # Fm0 = Kf * w[0]
        # Fm1 = Kf * w[1]  
        # Fm2 = Kf * w[2] 
        # Fm3 = Kf * w[3]
        # Mm0 = Km * w[0]
        # Mm1 = Km * w[1]
        # Mm2 = Km * w[2]
        # Mm3 = Km * w[3]


        for link_idx in range(-1, 3):
            action_idx = link_idx + 1
            #Thrust
            self.bc.applyExternalForce(self.robotID, link_idx, [0, 0, Fm[action_idx]], [0, 0, 0], 1)
            #Torque
            if link_idx == -1: #Reference frame bug workaround
                self.bc.applyExternalTorque(self.robotID, link_idx, [0, 0, Mm[action_idx]], 2)
            else:
                self.bc.applyExternalTorque(self.robotID, link_idx, [0, 0, Mm[action_idx]], 1)
            #Propeller visual (non critical)
            self.bc.setJointMotorControl2(self.robotID, self.propIDs[action_idx], self.bc.VELOCITY_CONTROL, targetVelocity=w[action_idx]*10, force=1000)
            #Control surface deflection [rads]
            self.bc.setJointMotorControl2(self.robotID, self.ctrlSurfIDs[action_idx], self.bc.POSITION_CONTROL, targetPosition=c[action_idx], force=1000)
            #Hinge Angle
            if action_idx != 3: #only 3 hinges
                self.bc.setJointMotorControl2(self.robotID, self.hingeIDs[action_idx], self.bc.POSITION_CONTROL, targetPosition=h[action_idx], force=1000)

        self.bc.stepSimulation()

    def center_of_mass(self):
        """
        Calculates the center of mass of the frame
        UAV position refers to this point
        """
        all_link_pos=[]    #TODO: Refactor this.


        all_link_pos.append((self.bc.getBasePositionAndOrientation(self.robotID))[0])
        for i in range(0, 3):
            all_link_pos.append((self.bc.getLinkState(self.robotID, i, 1))[0])

        
        center_of_mass = np.sum(all_link_pos, axis = 0)/4 #Average x, y, z, of all 4 link CoMs 
        center_of_mass[2] = center_of_mass[2] -.01 # Z intertial offset used in the urdf file
  
        return center_of_mass
    
    def reward(self, pos, vel, ang_vel):
        position_reward = (-1 * np.linalg.norm(pos)) + 5

        return position_reward
    
    """
    Gets state of environment, including reward of current state
    """
    def get_state(self):
        a, b, c, d, e, f, g, h = self.bc.getLinkState(self.robotID, 0, 1)
        position = self.center_of_mass()
        orientation = f #Quaternion
        rot_matrix = self.bc.getMatrixFromQuaternion(orientation)
        velocity = g 
        angular_velocity = h

        rew = self.reward(position, velocity, angular_velocity)

        return np.array([*rot_matrix, *position, *angular_velocity, *velocity]), rew

    
    def step(self, action):
        """
        Gym Env function
        steps environment forward 
        """
        self.sim_step(action)
        obs, rew = self.get_state()
        if rew < -20:
            done = True
        else:
            done = False 
        info = {}
        return obs, rew, done, info

    def set_to_pos_and_q(self, pos, q):
        """
        Sets the frame so a given position and orientation
        """
        self.bc.resetBasePositionAndOrientation(self.robotID, pos, q)

    def reset(self):
        """
        Gym Reset 
        """
        self.set_to_pos_and_q([0, 0, 0], [0, 0, 0, 1])
        # self.reset_random()
        obs, rew = self.get_state()
        return obs

    def reset_random(self): 
        """
        Resets to random location around origin with random orientation
        """
        pos = list(np.random.rand(3))
        q = utils.random_quaternion()
        self.set_to_pos_and_q(pos, q)

    def render(self):
        pass 
    
    def close(self):
        pass 

        
        