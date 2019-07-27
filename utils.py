import math
import numpy as np
import scipy.signal
import tensorflow as tf 
import pybullet

############################
#                          #
# Quadrotor Training Utils #
#                          #
############################

def quadrotor_reward(state):
    """
    Reward based on staying still on target
    """
    position_reward = (-1 * np.linalg.norm(state[0:3])) + 5
    orientation_reward = (-1 * np.linalg.norm(pybullet.getEulerFromQuaternion(state[3:7]))) * 0.0 #TODO verify (0, 0, 0) is desired orientation
    velocity_reward = (-1 * np.linalg.norm(state[7:10])) * 0.2
    angular_vel_reward = (-1 * np.linalg.norm(state[10:])) * 0.2
    return position_reward + orientation_reward + velocity_reward + angular_vel_reward

def fixed_wing_reward(state, setpoint, debug=False):
    """
    Reward based on maintaining a straight line velocity towards target
    Distance from target isn't explicity penalized, having a positive velocity towards target is reinforced
    TODO not spaghetti
    """
    position = state[0:3]
    velocity = state[7:10]
    velocity_magnitude = np.linalg.norm(velocity)
    velocity_U = velocity / velocity_magnitude
    offset = setpoint - position #this is the direction the velocity *should* be in
    offset_U = offset / np.linalg.norm(offset)
    error = velocity_U - offset_U
    reward = -1 * np.linalg.norm(error)
    #TODO decide scale / offset ????
    if debug:
        return position, velocity_U, offset_U 
    return reward

def quadrotor_action_mod(a):
    """
    Clips the action and sets joints to quadrotor-only
    """
    #previously had action scale, TODO decide if removal is bad
    # a *= 10 #doesn't work without an action scale, oddly enough
    a *= 50
    a = [clip(x, 17) for x in a]
    a = np.concatenate((a, [0, 0, 0, 0, 1.57, 1.57, 1.57]))
    return a

##################
#                #
# Waypoint Utils #
#                #
##################

def offset_state(state, waypoint):
    """
    Offsets the state by the waypoint's position
    Thus the waypoint will appear to be the origin
    """
    state[0] = state[0] - waypoint[0]
    state[1] = state[1] - waypoint[1]
    state[2] = state[2] - waypoint[2]
    return state

def within_waypoint(state, waypoint, error=0.2):
    """
    Boolean indicating whether the state is within error of the given waypoint
    """
    adjusted_state = offset(state, waypoint)
    return np.linalg.norm(adjusted_state[0:3]) < error 

def within_origin(state, error=0.2):
    """
    Boolean indicating whether the state is within error of the origin
    """
    return np.linalg.norm(state[0:3]) < error 

#####################
#                   #
# Environment Utils #
#                   #
#####################

def random_three_vector():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    This isn't the function we need
    TODO remove
    """
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return (x,y,z)

def random_rpy():
    """
    Generates random values for roll, pitch, and yaw between 0 and 2pi
    """
    r = np.random.uniform(0, np.pi*2)
    p = np.random.uniform(0, np.pi*2)
    y = np.random.uniform(0, np.pi*2)

    return (r, p, y)

def random_quaternion():
    """
    Turns a random rpy into quaternion form
    """
    return pybullet.getQuaternionFromEuler(random_rpy())

def random_force(magnitude):
    """
    Generates a random force between [-magnitude, magnitude] for x, y, and z
    """
    x = np.random.uniform(-magnitude, magnitude)
    y = np.random.uniform(-magnitude, magnitude)
    z = np.random.uniform(-magnitude, magnitude)
    
    return (x, y, z)

##################
#                #
# Learning Utils #
#                #
##################

def discount_cumsum(x, discount):
    """
    OpenAI described this as 'magic from rllab'
    their top researchers earn millions per year
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def gaussian_likelihood(x, mu, log_std, eps=1e-8, name="logp"):
    """
    Produces a gaussian likelihood tensor to sample from
    """
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+eps))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1, name=name)


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    if np.isscalar(shape):
        return (length, shape)
    return (length, *shape)

#################
#               #
# General Utils #
#               #
#################

def clip(val, clip_val):
    """
    Clips a value to be in range [-clip_val, clip_val]
    """
    if val < -clip_val:
        val = -clip_val
    elif val > clip_val:
        val = clip_val 
    return val

class Logger():
    """
    Bare-bones logger
    Queues strings until desired print, then formats in a nice block
    Enter strings using add_string in format ->
    "Name: Value"
    or enter a name and a value using add_named_value to achieve same result
    TODO add log to file 
    """
    def __init__(self):
        self.string_list = [] 
    
    def add_string(self, new_string):
        self.string_list.append(new_string)
    
    def add_named_value(self, name, value):
        if type(value) is int :
            self.string_list.append("{}: {}".format(name, value))
        else: #assuming float
            self.string_list.append("{}: {:.4f}".format(name, value))
    
    def max_len(self):
        return max(map(len, self.string_list)) 
    
    def output(self):
        line_size = self.max_len() + 4
        out = ("#" * line_size) + "\n"
        for val in self.string_list:
            out += "# "
            out += val 
            out += (" " * (line_size - 4 - len(val)))
            out += " #\n"
        out += ("#" * line_size)
        print(out)
        self.string_list = []


    
    

class ProgressBar():
    """
    General-use progress bar the prints to CLI and updates with carriage returns
    Appearance is as such ->
    [=====.....] is 50%
    TODO add optional percentage in addition to bar
    """
    def __init__(self, steps = 20, length=20, end_with_newline=False):
        self.total_steps = steps-1
        self.length = length
        self.empty_bar = length
        self.end_with_newline = end_with_newline


    def print(self, step):
        print("\r[", end="")
        bar_len = math.ceil(((step+1)/self.total_steps) * self.length)
        bar_len = int(bar_len)
        if bar_len > self.length:
            bar_len = self.length
        empty_bar = self.length - bar_len
        for _ in range(bar_len):
            print("=", end="")
        for _ in range(empty_bar):
            print(".", end="")
        print("]", end="")
        if step == self.total_steps and self.end_with_newline: 
            print()
