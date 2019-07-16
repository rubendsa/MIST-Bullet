import tensorflow as tf 
import numpy as np 
import time 
import utils 

from ppo import PPO 
from networks import general_mlp 
from hp import QUADROTOR_HPSTRUCT
from pybulletinstance import PyBulletInstance
    

x_ph = tf.placeholder(dtype=tf.float32, shape=(None, 13), name="x_input")
y_ph, v_ph = general_mlp(x_ph, output_dim=4)

hps = QUADROTOR_HPSTRUCT()

ppo = PPO(x_ph, y_ph, v_ph, discrete=False, hp_struct=hps, name="QuadrotorTest")
ppo.restore()

env = PyBulletInstance(GUI=True)
env.reset() #change to .reset() to start at origin

o = env.getState()
r = 0
waypoints = [[1,1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]]
waypoint_idx = 0
text_id = env.set_waypoint_text("Waypoint", waypoints[waypoint_idx])

for t in range(int(10e4)):
    a, _, _ = ppo.get_action_ops(o)
    action_to_apply = utils.quadrotor_action_mod(a[0])

    env.applyAction(action_to_apply) 
    env.step()
    time.sleep(env.get_viz_delay()) 

    o = env.getState()
    o = offset(o, waypoints[waypoint_idx])

    if within(o):
        print("good")
        waypoint_idx += 1
        if waypoint_idx >= 4:
            waypoint_idx = 0
        env.remove_waypoint_text(text_id)
        text_id = env.set_waypoint_text("Waypoint", waypoints[waypoint_idx])

    r = utils.quadrotor_reward(o)

    if r < -20:
        env.reset_random()
        o = env.getState()