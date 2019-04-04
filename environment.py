import time
import tensorflow as tf
from multiprocessing import Process, Queue, Manager
from nn_utils import restore_most_recent
from policy_f import PolicyFunction
from collections import deque

import numpy as np
import pybullet_data
from pyquaternion import Quaternion
import pybullet as p
from pybulletinstance import PyBulletInstance
from trajectory import Trajectory

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
        return np.concatenate((self.rot_mat, self.pos, self.ang_vel, self.vel))

    @staticmethod
    def from_arr(arr):
        rot_mat = np.reshape(arr[0:9], (3,3))
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


 #BIG UGLY
def one_rollout_process(noise, start_state, traj_list, rollout_len=1000):
    env = PyBulletInstance()

    policy_f = PolicyFunction(input_dim=18, output_dim=4)
    save_dir = "C:/Users/user/Transformation/MIST_Bullet/models"

    sess = tf.Session()
    init=tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    restore_most_recent(save_dir, saver, sess)

    start_state = State.from_arr(start_state) 
    current_state = start_state
    traj = Trajectory()
    # print(start_state)
    env.set_to_pos_and_q(start_state.pos, start_state.q)
    x = 0
    for _ in range (rollout_len):
        x += 1
        pos, q, v, ang_v = env.getUAVState()

        state = State(pos, v, ang_v, euler=None, q=q)
        action = policy_f.forward(sess, np.reshape(state.as_arr(), (1, 18))) #(1,4)
        action = np.reshape(action, [-1])
        current_noise = noise.get_noise()
        current_action = action + current_noise

        current_action = np.concatenate((current_action, [0, 0, 0, 0, 1.57, 1.57, 1.57]))
        env.applyAction(current_action)
        env.step()
        
        current_cost = calc_cost(state, action)
        traj.push_back(state.as_arr(), action, current_noise, current_cost)
    traj_list.append(traj)


class Environment():

    def __init__(self, policy_f, num_instances=20):
        self.policy_f = policy_f
        self.num_instances = num_instances
        # self.envs = []
        # for _ in range(self.num_instances):
        #     self.envs.append(PyBulletInstance())

    # @staticmethod
    # def one_rollout(env, sess, noise, policy_f, start_state, traj_deque, rollout_len=1000):
    #     current_state = start_state 
    #     traj = Trajectory()
    #     env.set_to_pos_and_q(start_state.pos, start_state.q)
    #     for _ in range (rollout_len):
    #         pos, q, v, ang_v = env.getUAVState()

    #         state = State(pos, v, ang_v, euler=None, q=q)
    #         action = policy_f.forward(sess, np.reshape(state.as_arr(), (1, 18))) #(1,4)
    #         action = np.reshape(action, [-1])
    #         current_noise = noise.get_noise()
    #         current_action = action + current_noise

    #         current_action = np.concatenate((current_action, [0, 0, 0, 0, 1.57, 1.57, 1.57]))
    #         env.applyAction(current_action)
    #         env.step()
            
    #         current_cost = calc_cost(state, action)
    #         traj.push_back(state.as_arr(), action, current_noise, current_cost)
    #     # traj_deque.append(traj)
    #     return 0

    #ALL HAIL THE GREAT SPAGHETTI FUNCTION
    def do_rollouts(self, sess, noise, traj_deque, start_states, rollout_len=10000):
        manager = Manager()
        traj_list = manager.list()
        #attempt 4
        while len(start_states) > 0:
            processes = []
            for i in range(self.num_instances):
                if len(start_states) > 0:
                    state_to_pass = start_states.pop().as_arr()
                    # print("Flag 1")
                    # print(state_to_pass)
                    pr = Process(target=one_rollout_process, args=(noise, state_to_pass, traj_list, rollout_len))
                    pr.start()
                    processes.append(pr)
                    # print(collected_trajectories)
            
            # print("ahh")
            for x in processes:
                x.join()
            # print("ree")
        for el in traj_list:
            traj_deque.append(el)
        #attmempt 3
        # while len(start_states) > 0:
        #     processes = []
        #     # i = 0
        #     for env in self.envs:
        #         if len(start_states) > 0:
        #             pr = Process(target=Environment.one_rollout, args=(env, sess, noise, self.policy_f, start_states.pop(), traj_deque, rollout_len))
        #             pr.start()
        #             processes.append(pr)
        #             # i += 1

        #     # print(i)
        #     for x in processes:
        #         x.join()


        #Attempt 2
        # while len(start_states) > 0:
        #     current_trajectories = []
        #     for env in self.envs: #Fill envs
        #         if len(start_states) > 0:
        #             current_trajectories.append(Trajectory())
        #             next_start = start_states.pop()
        #             env.set_to_pos_and_q(next_start.pos, next_start.q)
        #             env.currently_used = True 

        #     for _ in range(rollout_len):
        #         states = []
        #         for env in self.envs:
        #             if env.currently_used:
        #                 pos, q, v, ang_v = env.getUAVState()
        #                 state = State(pos, v , ang_v, euler=None, q=q)
        #                 states.append(state)
                
        #         batch_states = [x.as_arr() for x in states]
        #         batch_actions = self.policy_f.forward(sess, batch_states)
        #         noises = []
        #         costs = []
        #         for i, action in enumerate(batch_actions):
        #             current_noise = noise.get_noise()
        #             noises.append(current_noise)
        #             current_action = action + current_noise
        #             current_action = np.concatenate((current_action, [0, 0, 0, 0, 1.57, 1.57, 1.57]))
        #             self.envs[i].applyAction(current_action)
        #             self.envs[i].step()
        #             costs.append(calc_cost(states[i], action))
                
        #         for i, traj in enumerate(current_trajectories):
        #             # print("ah")
        #             # print(batch_states[i], batch_actions[i], noises[i], costs[i])
        #             traj.push_back(batch_states[i], batch_actions[i], noises[i], costs[i])

        #     for traj in current_trajectories:
        #         traj_deque.append(traj)

        #     for env in self.envs: #reset env flags
        #         env.currently_used = False

        #Attempt 1
        # for start_state in start_states:
        #     current_state = start_state 
        #     current_traj = Trajectory()
        #     set_to_pos_and_q(start_state.pos, start_state.q)
        #     for _ in range (rollout_len):
        #         pos, q, v, ang_v = getUAVState()
        #         state = State(pos, v, ang_v, euler=None, q=q)
        #         action = self.policy_f.forward(sess, state.as_arr()) #(1,4)
        #         action = np.reshape(action, [-1])
        #         # print(action)
        #         current_noise = noise.get_noise()
        #         current_action = action + current_noise

        #         current_action = np.concatenate((current_action, [0, 0, 0, 0, 1.57, 1.57, 1.57]))
        #         applyAction(current_action)
        #         step()
                
        #         current_cost = calc_cost(state, action)
        #         current_traj.push_back(state.as_arr(), action, current_noise, current_cost)
        #     traj_deque.append(current_traj)
