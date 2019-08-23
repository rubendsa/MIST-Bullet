import tensorflow as tf
import joblib 

from pybulletenv import PyBulletEnvironment

fn = "./data/local/experiment/experiment_2019_08_22_17_19_15_0,001/params.pkl"

env = PyBulletEnvironment(GUI=True)
obs = env.reset()

with tf.compat.v1.Session() as sess:
    data = joblib.load(fn)
    policy = data['algo'].policy
    
    for t in range(100000):
        action = policy.get_action(obs)

        obs, _, _, _ = env.step(action)
    
