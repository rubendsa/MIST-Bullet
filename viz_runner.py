import tensorflow as tf 
import numpy as np 
import time 
import utils 

from ppo import PPO 
from networks import general_mlp, quadrotor_mlp_1
from hp import QUADROTOR_HPSTRUCT
from pybulletinstance import PyBulletInstance

"""
TODO merge this file and mpi_runner 
into single ppo_runner file 
have ppo_runner take command line args to decide whether to viz, train, or whatever
have a single mpi_start file which has set args for number of threads and args to pass to ppo_runner 
"""


x_ph = tf.placeholder(dtype=tf.float32, shape=(None, 13), name="x_input")
y_ph, v_ph = general_mlp(x_ph, output_dim=4)


hps = QUADROTOR_HPSTRUCT()

ppo = PPO(x_ph, y_ph, v_ph, discrete=False, hp_struct=hps, name="QuadrotorTest")
ppo.restore()
env = PyBulletInstance(GUI=True)
env.reset_upwards()
# env.reset_random() #change to .reset() to start at origin
o = env.getState()
r = 0
d = False 


for t in range(100000):
    a, v_t, _ = ppo.get_action_ops(o)
    if t % 100 == 0:
        print(v_t)
    action_to_apply = utils.quadrotor_action_mod(a[0])

    env.applyAction(action_to_apply) 
    env.step()
    # time.sleep(env.getVizDelay())

    o = env.getState()
    r = utils.quadrotor_reward(o)
    if r < -200:
        d = True 

    if d or t % 2000 == 0:
        print(v_t)
        d = False
        # env.reset_random()
        env.old_reset_random()
        # env.reset()
        o = env.getState()

