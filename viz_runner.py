import tensorflow as tf 
import numpy as np 
import time 
# import gym 

from ppo import PPO 
from networks import general_mlp 
from hp import DEFAULT_HPSTRUCT, DEBUG_HPSTRUCT, QUADROTOR_HPSTRUCT
from mpi import proc_id, num_procs
from pybulletinstance import PyBulletInstance

#todo check if slicing is correct
#idk if this is a good signal
# def calc_cost(state, action):
#     return (4e-1 * np.linalg.norm(state[0:3])) \
#         + (2e-2 * np.linalg.norm(state[7:10])) \
#         + (2e-2 * np.linalg.norm(state[10:])) \
#         + (2e-2 * np.linalg.norm(action))
#this just penalizes position deviance from (0, 0, 0)
#being with 5 is rewarded positively
def calc_reward(state):
    return (-1 * np.linalg.norm(state[0:3])) + 5

#modifies action within parameters
def action_mod(a):
    a *= 50 #action scale
    #maybe add clipping?
    a = np.concatenate((a, [0, 0, 0, 0, 1.57, 1.57, 1.57]))
    return a

x_ph = tf.placeholder(dtype=tf.float32, shape=(None, 13), name="x_input")
y_ph, v_ph = general_mlp(x_ph, output_dim=4)


hps = QUADROTOR_HPSTRUCT()

ppo = PPO(x_ph, y_ph, v_ph, discrete=False, hp_struct=hps, name="QuadrotorTest")
ppo.restore()
action_scale = 50
# env = gym.make("LunarLanderContinuous-v2")
env = PyBulletInstance(GUI=True)
env.reset()
o = env.getState()
r = 0
d = False 
ep_ret = 0 
ep_len = 0
# env = gym.wrappers.Monitor(env, "./vids", video_callable=lambda episode_id: True,force=True)
print(o)

for t in range(10000):
    a, _, _ = ppo.get_action_ops(o)
    # print(a)
    action_to_apply = action_mod(a[0])
    env.applyAction(action_to_apply) 
    env.step()
    time.sleep(0.01)
    o = env.getState()
    r = calc_reward(o)
    # print(r)
    if r < -20:
        d = True 

    # print(r)

    if d:
        print(a[0])
        d = False
        env.reset()
        o = env.getState()

