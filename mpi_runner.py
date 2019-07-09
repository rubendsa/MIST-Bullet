import tensorflow as tf 
import numpy as np 
import time 


from ppo import PPO 
from networks import general_mlp 
from hp import DEFAULT_HPSTRUCT, DEBUG_HPSTRUCT, QUADROTOR_HPSTRUCT
from mpi import proc_id
from pybulletinstance import PyBulletInstance

#TODO put all this garbage in HPS
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

env = PyBulletInstance(GUI=False)
env.reset()
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
        # print(a[0])
        action_to_apply = action_mod(a[0])
        
        env.applyAction(action_to_apply) 
        env.step()
        o = env.getState()
        r = calc_reward(o)
        ep_ret += r
        ep_len += 1

        if r < -20:
            d = True 

        terminal = d 
        if terminal or (t == ppo.local_steps_per_epoch- 1):
            # if not terminal:
                # print("Warning: Traj cut off by epoch at {} steps".format(ep_len))
            last_val = r if d else ppo.get_v(o)
            ppo.buf.finish_path(last_val)
            # print(a[0])
            env.reset()
            o = env.getState()
            r = 0
            d = False 
            ep_ret = 0 
            ep_len = 0

    if epoch % ppo.hps.save_freq == 0:
        ppo.save() 

    print("Done with epoch {} in proc #{}".format(epoch, proc_id()))
    ppo.update()
