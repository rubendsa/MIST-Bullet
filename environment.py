import time
import tensorflow as tf
import multiprocessing
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

class ProcessInput():

    def __init__(self, noise, start_state, rollout_len, load_model):
        self.noise = noise 
        self.start_state = start_state 
        self.rollout_len = rollout_len 
        self.load_model = load_model

class PyBulletProcess(multiprocessing.Process):
    
    def __init__(self, input_queue, output_queue, save_dir):
        multiprocessing.Process.__init__(self)
        self.input_queue = input_queue 
        self.output_queue = output_queue 
        self.save_dir = save_dir

    def run(self):
        env = PyBulletInstance()
        policy_f = PolicyFunction(input_dim=18, output_dim=4)

        sess = tf.Session() 
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()

        while(True):
            pi = self.input_queue.get()
            current_state = pi.start_state  
            traj = Trajectory()
            env.set_to_pos_and_q(current_state.pos, current_state.q)
            if pi.load_model:
                restore_most_recent(self.save_dir, saver, sess)  

            for _ in range(pi.rollout_len):
                pos, q, v, ang_v = env.getUAVState()

                state = State(pos, v, ang_v, euler=None, q=q)
                action = policy_f.forward(sess, np.reshape(state.as_arr(), (1, 18))) #(1,4)
                action = np.reshape(action, [-1])
                current_noise = pi.noise.get_noise()
                current_action = action + current_noise

                current_action = np.concatenate((current_action, [0, 0, 0, 0, 1.57, 1.57, 1.57]))
                env.applyAction(current_action)
                env.step()
                
                current_cost = calc_cost(state, action)
                traj.push_back(state.as_arr(), action, current_noise, current_cost)
            #TODO actually map out this whole spaghetti on the whiteboard
            #At this point the sim has terminated due to timeout (There's actually no other termination case but w/e)
            #RAI arbitrarily has termValue as 1.5
            traj.terminate_traj_and_update_val(state.as_arr(), action, 1.5, 0.99) #todo actual discount factor
            #
            self.output_queue.put(traj)
            self.input_queue.task_done()

class EnvironmentMananger():
    def __init__(self, n_instances=20, save_dir=None):
        self.n_instances = n_instances
        self.task_queue = multiprocessing.JoinableQueue()
        self.result_queue = multiprocessing.Queue()
        self.processes = [PyBulletProcess(self.task_queue, self.result_queue, save_dir) for _ in range(self.n_instances)]
        for p in self.processes:
            p.start()
    
    def terminate_processes(self):
        for p in self.processes:
            p.terminate()
    
    def do_rollouts(self, noise, traj_dequeue, start_states, rollout_len=10000, load_model=True):
        num_sims_to_run = len(start_states)
        for state in start_states:
            pi = ProcessInput(noise, state, rollout_len, load_model)
            self.task_queue.put(pi)
        
        self.task_queue.join()

        for _ in range(len(start_states)):
            traj_dequeue.append(self.result_queue.get()) #this is the only way to do it
        # while not self.result_queue.empty():
        #     traj_dequeue.append(self.result_queue.get()) #this doesn't work as the output_queue is not necessary synced