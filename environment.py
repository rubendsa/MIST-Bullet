import numpy as np 
import pybullet as p
import time
import pybullet_data

from trajectory import Trajectory
from MIST_run import step, getUAVState, applyAction, set_to_pos_and_q
from pyquaternion import Quaternion


# from tf.transformations import quaternion_matrix, quaternion_from_euler

# def quat_to_rot_mat(q):
#     rot_mat = transform.transformations.quaternion_matrix(q)
#     return rot_mat[0:3,0:3]

# #this is not too bad
# def turn_flat_rot_mat_into_4_x_4(rot_flat):
#     out = np.zeros((4,4))
#     out[0:3,0:3] = np.reshape(rot_flat, (3,3))
#     out[3][3] = 1
#     return out

#i hate flipping this sort of thing to negative TODO figure out how not to be bad
def calc_cost(state, action):
    return (4e-3 * np.linalg.norm(state.pos)) \
        + (2e-5 * np.linalg.norm(state.vel)) \
        + (2e-5 * np.linalg.norm(state.ang_vel)) \
        + (2e-5 * np.linalg.norm(action))

class State():
    def __init__(self, pos, v, rot_v, euler=None, q=None):
        if euler is not None:
            self.euler = euler
            self.q = p.getQuaternionFromEuler(self.euler)
        elif q is not None:
            self.q = q 
            self.euler = p.getEulerFromQuaternion(self.q)
        else:
            print("You've done something terribly wrong with the state init, exiting")
            exit()
        self.rot_mat = p.getMatrixFromQuaternion(self.q)
        self.pos = pos 
        self.ang_vel = v 
        self.vel = rot_v
    
    def as_arr(self):
        return np.reshape(self.rot_mat + self.pos + self.ang_vel + self.vel, (1, -1))

    @staticmethod
    def from_arr(arr):
        arr = arr[0]
        rot_mat = np.reshape(arr[0:9], (3,3))
        # q = p.getQuaternionFromMatrix(rot_mat)
        q = Quaternion(matrix=rot_mat).elements
        pos = arr[9:12]
        rot_v = arr[12:15]
        v = arr[15:18]
        return State(pos, v, rot_v, euler=None, q=q)

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
    
def random_state():
    pos = np.random.normal(size=3)
    v = np.random.normal(size=3)
    rot_v = np.random.normal(size=3)
    q = p.getQuaternionFromEuler(random_three_vector())
    state_out = State(pos, v, rot_v, q=q)
    return state_out

class Environment():

    def __init__(self, policy_f, num_threads):
        self.policy_f = policy_f
        self.num_threads = num_threads
        _ = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-10)
        _ = p.loadURDF("plane.urdf")
        
    #this might work yet, debug
    def do_rollouts(self, sess, noise, traj_deque, start_states, rollout_len=10000):
        for start_state in start_states:
            current_state = start_state 
            current_traj = Trajectory()
            set_to_pos_and_q(start_state.pos, start_state.q)
            for _ in range (rollout_len):
                pos, q, v, ang_v = getUAVState()
                state = State(pos, v, ang_v, euler=None, q=q)
                action = self.policy_f.forward(sess, state.as_arr()) #(1,4)
                action = np.reshape(action, [-1])
                # print(action)
                current_noise = noise.get_noise()
                current_action = action + current_noise

                current_action = np.concatenate((current_action, [0, 0, 0, 0, 1.57, 1.57, 1.57]))
                applyAction(current_action)
                step()
                
                current_cost = calc_cost(state, action)
                current_traj.push_back(state.as_arr(), action, current_noise, current_cost)
            traj_deque.append(current_traj)


        