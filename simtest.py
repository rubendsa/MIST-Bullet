import tensorflow as tf 
import numpy as np

import utils 

from pybulletinstance import PyBulletInstance
from pybulletenv import PyBulletEnvironment 

from ppo import PPO 
from networks import general_mlp 
from hp import QUADROTOR_HPSTRUCT


instance = PyBulletInstance(GUI=False)
instance.reset()
env = PyBulletEnvironment(GUI=False)
env.reset()

o = instance.getState()
r = 0
d = False

for t in range(10000):
    a = np.array([10, 10, 10, 10])
    action_to_apply = utils.quadrotor_action_mod(a)

    instance.applyAction(action_to_apply)
    instance.step()

    o2, _, _, _ = env.step(a)
    o = instance.getState()

    # print("{} : {}".format(o, o2))
    print(o == o2)
    

