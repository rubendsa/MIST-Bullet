import tensorflow as tf 
import numpy as np 
import time 
import utils 
import sys

from ppo import PPO 
from networks import general_mlp, quadrotor_mlp_1
from hp import QUADROTOR_HPSTRUCT
from mpi import proc_id
from pybulletinstance import PyBulletInstance

x_ph = tf.placeholder(dtype=tf.float32, shape=(None, 13), name="x_input")
y_ph, v_ph = general_mlp(x_ph, output_dim=4)

hps = QUADROTOR_HPSTRUCT()

ppo = PPO(x_ph, y_ph, v_ph, discrete=False, hp_struct=hps, name="QuadrotorTest")
# ppo.restore()

logger = utils.Logger() #TODO move to only proc 0

env = PyBulletInstance(GUI=False)
env.old_reset_random()
# env.reset()

o = env.getState()
r = 0
d = False 
ep_ret = 0 
ep_len = 0


print("Starting execution in process {}".format(proc_id()))
sys.stdout.flush()
start = time.time()
for epoch in range(ppo.hps.epochs):
    logger.add_named_value("Epoch", epoch)
    max_ep_ret = float("-inf")
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

        
        """
        3 ways to end rollout and associated terminal value:
        1) Reward to low (reward)
        2) End of epoch (value network)
        3) Time limit reached (value network)
        """
        if r < -20: #very arbitrary  
            d = True 

        if d or t == ppo.local_steps_per_epoch - 1: # or ep_len > ppo.hps.max_steps_per_rollout:
            last_val = r if d else ppo.get_v(o)
            ppo.buf.finish_path(last_val)
            # env.reset()
            env.old_reset_random()

            if d: # or ep_len > ppo.hps.max_steps_per_rollout:
                if (ep_ret / ep_len) > max_ep_ret:
                    max_ep_ret = (ep_ret / ep_len)

            o = env.getState()
            r = 0
            d = False 
            ep_ret = 0 
            ep_len = 0

    logger.add_named_value("Max Ep Ret", max_ep_ret)

    ppo.update(logger)

    #Log in only 1 process
    if proc_id() == 0:
        logger.output()
        sys.stdout.flush()
        # if epoch % ppo.hps.log_freq == 0:
        #     logger.write_to_file()

    """
    Saving only needs to occur in 1 process
    TODO save only in the first process that reaches this point
    """
    if epoch % ppo.hps.save_freq == 0 and proc_id() == 0:
        ppo.save() 
        print("Saved!")