import tensorflow as tf
import time
import os 

from enum import Enum

from policy_f import PolicyFunction
from value_f import ValueFunction
from ag_tree import run_ag_tree
from environment import Environment
from nn_utils import restore_from_lowest_cost, restore_from_highest_cost, save_model, save_most_recent, restore_most_recent

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

class LoadType(Enum):
    CREATE_NEW_MODEL = 1
    LOAD_HIGHEST_COST = 2
    LOAD_LOWEST_COST = 3
    LOAD_MOST_RECENT = 4

def run_experiment(load_type, max_time, save_dir):
    start_time = time.time()
    OBS_SIZE = 18
    ACTION_SIZE = 4
    value_f = ValueFunction(input_dim=OBS_SIZE)
    policy_f = PolicyFunction(input_dim=OBS_SIZE, output_dim=ACTION_SIZE)

    sess = tf.Session()
    init=tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    env = Environment(policy_f, num_instances=10)

    best_cost = 100000000
    if load_type == LoadType.CREATE_NEW_MODEL:
        pass
    elif load_type == LoadType.LOAD_HIGHEST_COST:
        best_cost = restore_from_highest_cost(save_dir, saver, sess)
    elif load_type == LoadType.LOAD_LOWEST_COST:
        best_cost = restore_from_lowest_cost(save_dir, saver, sess)
    elif load_type == LoadType.LOAD_MOST_RECENT:
        restore_most_recent(save_dir, saver, sess)

    save_most_recent(save_dir, saver, sess)

    while(True):
        
        # i += 1
        # print("loop {}".format(i))
        start = time.time()
        current_cost = run_ag_tree(sess, value_f, policy_f, env, rollout_len=500, num_initials_trajs=100, num_branches=200, noise_depth=2, discount_factor=0.99)
        end = time.time()
        # print("Time for one loop is {}".format(end-start))
        if current_cost < best_cost:
            best_cost = current_cost 
            save_model(save_dir, saver, sess, best_cost)
        
        #always save most recent model
        save_most_recent(save_dir, saver, sess)

        current_time = time.time()
        duration = current_time - start_time
        if duration > max_time:
            print("End of experiment, time duration reached")
            break 


if __name__ == "__main__":
    save_dir = "C:/Users/user/Transformation/MIST_Bullet/models"
    time_in_seconds = 12 * 3600
    run_experiment(LoadType.LOAD_MOST_RECENT, time_in_seconds, save_dir)
