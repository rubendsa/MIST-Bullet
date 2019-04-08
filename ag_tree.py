#!/usr/bin/env python
import tensorflow as tf 
import time
import numpy as np
import datetime

from collections import deque 
from trajectory import Trajectory
from noise import Noise
from environment import random_state, State


#Algorithm structure based upon https://bitbucket.org/leggedrobotics/rai/src/980faf208d82bd2a25dfdb3d59a976c4acc96550/RAI/include/rai/algorithm/AG_tree.hpp

def get_terminal_states_of_traj_deque(traj_deque):
    output = []
    for traj in traj_deque:
        output.append(traj.get_terminal_state())
    return output

def run_ag_tree(sess, value_f, policy_f, env, rollout_len=400, num_initials_trajs=100, num_branches=200, noise_depth=2, discount_factor=0.99):
    ######################Stage #0: Setup######################
    advtuple_state = deque()
    advtuple_action_noise = deque()
    advtuple_advantage = deque()
    advtuple_gradient = deque()

    
    noise = Noise(Noise.NOISE)
    noise.set_cov(0.2)
    no_noise = Noise(Noise.NO_NOISE)

    #this implementation hurts me on many levels
    value_junction = deque()
    for i in range(noise_depth+1): #TODO i have no idea if this +1 should be here, but it sorta fixes some things?
        branch = deque()
        for q in range(num_branches):
            branch.append(None)
        value_junction.append(branch)


    ####################Stage #1: Simuation####################
    initial_trajectories = deque() #full of Trajectories
    junction_trajectories = deque()
    branch_trajectories = deque() #full of deques full of trajectories - consider renaming TODO 
    for i in range(noise_depth):
        temp = deque()
        branch_trajectories.append(temp)

    start_states = []
    for _ in range(num_initials_trajs):
        start_states.append(random_state())

    # print("Initial Rollouts")
    env.do_rollouts(no_noise, initial_trajectories, start_states, rollout_len, load_model=True) 

    terminal_states = get_terminal_states_of_traj_deque(initial_trajectories)
    terminal_states = np.reshape(terminal_states, [-1, 18])
    terminal_values = value_f.forward(sess, terminal_states)
    for i, traj in enumerate(initial_trajectories):
        traj.update_val_traj_with_new_terminal_val(terminal_values[i], discount_factor)

    #maybe reconsider how this is done?
    num_trajectories_to_sample_from = len(initial_trajectories)
    start_state_juncts = deque()
    indx = deque()
    for i in range(num_branches):
        sampled_traj_idx = np.random.randint(0, num_trajectories_to_sample_from)
        sampled_traj = initial_trajectories[sampled_traj_idx]
        num_states_to_sample_from = len(sampled_traj.states)
        sampled_state_idx = np.random.randint(0, num_states_to_sample_from)
        #
        sampled_state = State.from_arr(sampled_traj.states[sampled_state_idx])
        #
        start_state_juncts.append(sampled_state)
        indx.append([sampled_traj_idx, sampled_state_idx])

    # print("Junction Rollouts")
    env.do_rollouts(noise, junction_trajectories, start_state_juncts, rollout_len, load_model=False)

    for i in range(num_branches):
        value_junction[0][i] = initial_trajectories[indx[i][0]].values[indx[i][1]]

    # print("Branch Rollouts")
    for depth in range(1, noise_depth + 1): #TODO
        nthState = deque()
        for jtraj in junction_trajectories:
            branch_start_state = State.from_arr(jtraj.states[depth])
            nthState.append(branch_start_state) 
        env.do_rollouts(no_noise, branch_trajectories[depth - 1], nthState, rollout_len, load_model=False)
        branch_terminal_states = get_terminal_states_of_traj_deque(branch_trajectories[depth-1])
        branch_terminal_states = np.reshape(branch_terminal_states, [-1, 18]) #bad 
        branch_terminal_values = value_f.forward(sess, branch_terminal_states)

        for i in range(num_branches):
            branch_trajectories[depth - 1][i].update_val_traj_with_new_terminal_val(0.0, discount_factor)
            value_junction[depth][i] = branch_trajectories[depth - 1][i].values[0] 
            advtuple_state.append(junction_trajectories[i].states[depth - 1])
            advtuple_action_noise.append(junction_trajectories[i].noises[depth - 1])
            advtuple_advantage.append(value_junction[depth][i] * discount_factor
                                        + junction_trajectories[i].costs[depth - 1]
                                        - value_junction[depth - 1][i]
                                        )
            current_noise = advtuple_action_noise[-1]
            norm_noise = np.linalg.norm(current_noise)
            advtuple_gradient.append(current_noise / norm_noise * advtuple_advantage[-1])


    ####################Stage #2: Value Function Update####################
    states_to_batch = []
    values_to_batch = []

    for i in range(num_branches):
        for depth in range(noise_depth + 1):
            states_to_batch.append(junction_trajectories[i].states[depth])
            values_to_batch.append(value_junction[depth][i])

    batch_size = 20

    for i in range(len(states_to_batch)):
        state_batch = []
        value_batch = []
        state_batch.append(states_to_batch[i])
        value_batch.append(values_to_batch[i])
        if len(state_batch) == batch_size:
            loss = value_f.train(sess, state_batch, value_batch)
            if loss < 0.0001:
                print("Value Function loss training skip achieved!")
                break
            state_batch = []
            value_batch = []

    ####################Stage #3: Policy Function Update####################

    num_policy_param = sess.run(policy_f.number_param_op)
    param_update = np.zeros(num_policy_param)


    data_len = len(advtuple_gradient)

    for i, state in enumerate(advtuple_state):
        jacobianQ_wrt_action = -advtuple_gradient[i]

        jacobian_action_wrt_params = sess.run(policy_f.jacobian_op_different, feed_dict={policy_f.x:np.reshape(state, [1,-1])}) #feed in state, return jaco
        jacobianQ_wrt_param = np.matmul(jacobianQ_wrt_action, jacobian_action_wrt_params) 

        noise_covariance = noise.get_covariance()
        fim_in_action_space = np.linalg.inv(noise_covariance)
        fim_in_action_space_cholesky = np.linalg.cholesky(fim_in_action_space)
        fim_cholesky = np.matmul(np.transpose(fim_in_action_space_cholesky), jacobian_action_wrt_params)
        
        u, singular_values, matrixV_T = np.linalg.svd(fim_cholesky, full_matrices=False)
        matrixV = np.transpose(matrixV_T)
        
        singular_value_inv_squared_matrix = np.diag( #view vector as diag matrix
            np.square(
                np.reciprocal( #element-wise inverse
                    singular_values
                    )
            )
        )
        natural_gradient_dir = np.matmul(matrixV, np.matmul(singular_value_inv_squared_matrix, np.matmul(matrixV_T, jacobianQ_wrt_param)))

        policy_learning_rate = 2300.0
        beta = np.sqrt(7000.0 / np.dot(natural_gradient_dir, jacobianQ_wrt_param))
        policy_learning_rate = min(policy_learning_rate, beta)

        param_update += ((policy_learning_rate * natural_gradient_dir) / data_len) 
  
    new_params = np.reshape(sess.run(policy_f.all_param), (1,-1)) + np.reshape(param_update, (1, -1)) #reshapes are necessary so the feed to param_assign works, don't change
    sess.run(policy_f.all_parameters_assign_all_op, feed_dict={policy_f.param_assign_placeholder:new_params})

    total_costs = 0
    avg_cost = 0
    for traj in initial_trajectories:
        avg_cost += np.sum(traj.costs)
        total_costs +=1
    avg_costs = avg_cost / total_costs
    print("Average Costs -> {} at time {}".format(avg_costs, datetime.datetime.now()))
    return avg_costs