import tensorflow as tf
import numpy as np 
from nn_utils import get_FC_model
from operator import mul
from noise import Noise
from functools import reduce

class PolicyFunction():
    
    def __init__(self, input_dim=18, output_dim=4, scope="policy_network"):
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, input_dim))

        self.scope = scope

        self.network = get_FC_model(self.x, self.scope, True, hidden_size=[128, 128], output_size=output_dim)
        
        self.param_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        self.number_param_op = tf.identity(tf.constant(value=[sum([reduce(mul, param.get_shape().as_list(), 1) for param in self.param_list])], dtype=tf.int32))
        
        self.all_param = tf.reshape(tf.concat([[tf.reshape(param, [-1])] for param in self.param_list], axis=1), [-1, 1])

        self.jacobian_op = tf.concat([tf.reshape(tf.concat([tf.reshape(tf.gradients(self.network[:,idx], param), [-1]) 
                for param in self.param_list], axis=0), [-1, 1]) for idx in range(output_dim)], axis=1)
        
        self.jacobian_op_different = tf.concat([tf.reshape(tf.concat([tf.reshape(tf.gradients(self.network[:,idx], param), [-1]) 
                for param in self.param_list], axis=0), [1, -1]) for idx in range(output_dim)], axis=0)
            


        self.param_assign_placeholder = tf.placeholder(tf.float32)

        param_assign_split = tf.split(self.param_assign_placeholder, [reduce(mul, param.get_shape().as_list(), 1) for param in self.param_list], 1)
        param_assign_op_list = []

        for idx, param in enumerate(self.param_list):
            reshaped_input_vector = tf.reshape(param_assign_split[idx], shape=tf.shape(param))
            param_assign_op_list += [param.assign(reshaped_input_vector)]

        self.all_parameters_assign_all_op = tf.group(*param_assign_op_list)

    def forward(self, sess, states):
        action = sess.run(self.network, feed_dict={self.x:states})
        return action
    
#oh boy
if __name__ == "__main__":
    
    f = PolicyFunction(2, 4, "policy_network")
    sess = tf.Session()
    init=tf.global_variables_initializer()
    sess.run(init)
    sample_input = [1.5, 2]
    sample_input = np.reshape(sample_input, [1, -1])
    # print(f.jacobian_op) #currently [AP, 4], but we want [4, AP]
    # print(f.jacobian_op_different) #[AP, 4]
    print(sess.run(f.param_list))
    print(sess.run(f.network, feed_dict={f.x:sample_input}))

    test_op = tf.gradients(f.network[:,0], f.param_list)
    print(sess.run(test_op, feed_dict={f.x:sample_input}))

    # param_update = np.zeros((1,12)) #[0][0-5] are weights, [0][6-8] are biases
    # # param_update[0][0] = 2
    # # param_update[0][1] = 3
    # sess.run(f.all_parameters_assign_all_op, feed_dict={f.param_assign_placeholder:param_update})

    # print(sess.run(f.param_list))
    # print(sess.run(f.network, feed_dict={f.x:sample_input}))

    jacobian_action_wrt_params = sess.run(f.jacobian_op_different, feed_dict={f.x:sample_input})
    print(jacobian_action_wrt_params)

    jacobianQ_wrt_action = [-1, 1, 0, 0] #cost wrt action
    jacobianQ_wrt_param = np.matmul(jacobianQ_wrt_action, jacobian_action_wrt_params) 

    noise = Noise(Noise.NOISE)
    noise.set_cov(0.2)

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


    print(natural_gradient_dir)
