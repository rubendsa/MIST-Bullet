import tensorflow as tf 
import numpy as np 
import time 
import utils 

from ppo import PPO 
from networks import general_mlp 
from hp import DEFAULT_HPSTRUCT, DEBUG_HPSTRUCT, QUADROTOR_HPSTRUCT
from mpi import proc_id
from pybulletinstance import PyBulletInstance

x_ph = tf.placeholder(dtype=tf.float32, shape=(None, 13), name="x_input")
y_ph, v_ph = general_mlp(x_ph, output_dim=4)

hps = QUADROTOR_HPSTRUCT()

ppo = PPO(x_ph, y_ph, v_ph, discrete=False, hp_struct=hps, name="QuadrotorTest")
# ppo.restore()

env = PyBulletInstance(GUI=False)
env.reset_random()

o = env.getState()
r = 0
d = False 
ep_ret = 0 
ep_len = 0

print("Starting execution in process {}".format(proc_id()))
for epoch in range(ppo.hps.epochs):
    for t in range(ppo.local_steps_per_epoch):
        a, v_t, logp_t = ppo.get_action_ops(o)

        ppo.buf.store(obs=o, act=a, rew=r, val=v_t, logp=logp_t)
        action_to_apply = utils.quadrotor_action_mod(a[0])
        
        env.applyAction(action_to_apply) 
        env.step()
        o = env.getState()
        r = utils.quadrotor_reward(o)
        ep_ret += r
        ep_len += 1

        if r < -20:
            d = True 

        terminal = d 
        if terminal or (t == ppo.local_steps_per_epoch- 1):
            last_val = r if d else ppo.get_v(o)
            ppo.buf.finish_path(last_val)
            env.reset_random()
            o = env.getState()
            r = 0
            d = False 
            ep_ret = 0 
            ep_len = 0

    

    print("Done with epoch {} in proc #{}".format(epoch, proc_id()))
    ppo.update()

    """
    Saving only needs to occur in 1 process
    TODO save only in the first process that reaches this point
    """
    if epoch % ppo.hps.save_freq == 0 and proc_id() == 0:
        ppo.save() 
        print("Saved!")