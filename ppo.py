import tensorflow as tf 
import numpy as np 
import utils 
from hp import HPStruct, DEFAULT_HPSTRUCT
from mpi import mpi_statistics_scalar, MpiAdamOptimizer, mpi_avg, num_procs, sync_all_params

"""
PPO modified from OpenAI's SpinningUp implementation
It's significantly less interconnected than their's is
The goal of this was to make it general purpose:
Easy to interface with any network architecture or environment, 
and readable enough that implementing in another framework (or TF 2.0, hopefully) should be trivial

This should run fine in a single process, however if necessary the MPI calls can
be converted to single process as such:
Replace mpi_statistics_scalar, mpi_avg, MpiAdamOptimizer with their single process equivalents
Remove all instances of dividing steps_per_epoch amongst processes
Remove sync_all_params call
"""

class PPOBuffer:
    """
    Buffer that holds observation from a single epoch
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(utils.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(utils.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam 
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size 

    def store(self, *, obs, act, rew, val, logp):
        """
        Adds memory to the buffer
        Keyword-only to avoid confusion:
        Inputs are often single-letter variables in practice, which is *super* error-prone
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs 
        self.act_buf[self.ptr] = act 
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val 
        self.logp_buf[self.ptr] = logp 
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Runs along the path to calculate the terminal values
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = utils.discount_cumsum(deltas, self.gamma * self.lam)

        self.ret_buf[path_slice] = utils.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr 

    def get(self):
        """
        Returns all the buffers (with advantage averaged across processes)
        Sets buffer pointers back to start, resulting in a 'fresh' buffer
        """
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf]

class PPO():
    """
    Proximal Policy Optimization (PPO)
    Takes a defined tensorflow graph containing an input, an ouput, and a value output
    Given a full PPOBuffer, will update the network accordingly

    Implemented for TF 1.x, so tf.compat.v1 is needed pretty often
    TODO decide if upgrading is viable
    """

    def __init__(self, x_ph, y_ph, v_ph, discrete=True, hp_struct=None, save_path="./models/", name="ppo_model"):
        """
        x_ph, y_ph, v_ph tensors from an already defined network graph
        discrete indicates the type of network output
            true will yield a categorical policy
            false will yield a gaussian policy
        hp_struct contains hyperparameters, see hp.py
        save_path is the directory to save to
        name is the name to stay with
        """
        self.x_ph = x_ph 
        self.y_ph = y_ph
       
        if hp_struct is None:
            self.hps = DEFAULT_HPSTRUCT()
        else:
            self.hps = hp_struct
        self.save_name = str(save_path + name + ".ckpt") 
        
        self.adv_ph = tf.placeholder(dtype=tf.float32, shape=(None), name="advantage")
        self.ret_ph = tf.placeholder(dtype=tf.float32, shape=(None), name="return")
        self.logp_old_ph = tf.placeholder(dtype=tf.float32, shape=(None), name="logp_old")

        with tf.compat.v1.variable_scope("pi"):
            if discrete: #Categorical policy
                logits = y_ph #technicality
                logp_all = tf.nn.log_softmax(logits)
                self.pi = tf.squeeze(tf.multinomial(logits, 1), axis=1, name="pi")
                self.a_ph = tf.placeholder(dtype=tf.int32, shape=self.pi.get_shape(), name="action")
                act_dim = logits.get_shape()[-1]
                self.logp = tf.reduce_sum(tf.one_hot(self.a_ph, depth=act_dim) * logp_all, axis=1, name="logp")
                self.logp_pi = tf.reduce_sum(tf.one_hot(self.pi, depth=act_dim) * logp_all, axis=1, name="logp_pi")
            else: #Gaussian policy
                mu = y_ph #technicality
                log_std = tf.compat.v1.get_variable(name='log_std', initializer=-0.5*np.ones(mu.get_shape()[-1], dtype=np.float32)) #
                std = tf.exp(log_std)
                self.pi = tf.identity(mu + tf.random.normal(tf.shape(mu)) * std, name="pi")
                self.a_ph = tf.placeholder(dtype=tf.float32, shape=self.pi.get_shape(), name="action")
                self.logp = utils.gaussian_likelihood(self.a_ph, mu, log_std, name="logp")
                self.logp_pi = utils.gaussian_likelihood(self.pi, mu, log_std, name="logp_pi")
        
        with tf.compat.v1.variable_scope("v"):
            self.v = tf.squeeze(v_ph, axis=1, name="value")
        
        self.all_inputs_phs = [self.x_ph, self.a_ph, self.adv_ph, self.ret_ph, self.logp_old_ph]
        self.action_ops = [self.pi, self.v, self.logp_pi]

        #This helps work with multidimensional obs
        self.obs_reshape = self.x_ph.get_shape().as_list()
        self.obs_reshape[0] = 1

        obs_dim = self.x_ph.get_shape()[1:]
        action_dim = self.a_ph.get_shape()[1:]
        if discrete: 
            action_dim = ()

        self.local_steps_per_epoch = int(self.hps.steps_per_epoch / num_procs())
        self.buf = PPOBuffer(obs_dim, action_dim, self.local_steps_per_epoch, self.hps.gamma, self.hps.lam)

        #PPO Objectives
        with tf.compat.v1.variable_scope("ppo_objectives"):
            self.ratio = tf.exp(self.logp - self.logp_old_ph, name="network_ratio")   #pi(a|s) / pi_old(a|s)
            self.min_adv = tf.where(self.adv_ph>0, (1+self.hps.clip_ratio)*self.adv_ph, (1-self.hps.clip_ratio)*self.adv_ph, name="min_advantage")
            self.pi_loss = tf.identity(-tf.reduce_mean(tf.minimum(self.ratio * self.adv_ph, self.min_adv)), name="pi_loss")
            self.v_loss = tf.reduce_mean((self.ret_ph - self.v)**2, name="v_loss")

        #Info
        with tf.compat.v1.variable_scope("ppo_indicators"):
            self.approx_kl = tf.reduce_mean(self.logp_old_ph - self.logp, name="approx_kl")
            self.approx_ent = tf.reduce_mean(-self.logp, name="approx_entropy")
            self.clipped = tf.logical_or(self.ratio > (1+self.hps.clip_ratio), self.ratio < (1-self.hps.clip_ratio), name="clipped_ratio")
            self.clipfrac = tf.reduce_mean(tf.cast(self.clipped, tf.float32), name="clipped_frac")

        #Optim
        with tf.compat.v1.variable_scope("train_ops"):
            self.train_pi_op = MpiAdamOptimizer(learning_rate=self.hps.pi_lr).minimize(self.pi_loss, name="train_pi_op")
            self.train_v_op = MpiAdamOptimizer(learning_rate=self.hps.vf_lr).minimize(self.v_loss, name="train_v_op")

        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(sync_all_params())
        self.saver = tf.compat.v1.train.Saver()
        
        self.writer = tf.compat.v1.summary.FileWriter(save_path, self.sess.graph)
    
    def save(self):
        """
        Saves the model
        """
        self.saver.save(self.sess, self.save_name)
        
    
    def restore(self):
        """
        Restores the model
        """
        self.saver.restore(self.sess, self.save_name)

    def update(self, logger=None):
        """
        Performs a PPO update on the network
        Call only if the PPOBuffer is full
        Handles batching if it is indicated by the HPS
        """
        memories = self.buf.get()
        start_idx = 0
        end_idx = self.hps.batch_size

        #Training
        for i in range(self.hps.train_pi_iters):
            if self.hps.batch_size is None:
                batch = memories
            elif end_idx >= len(memories[0]):
                batch = [x[start_idx:] for x in memories]
                start_idx = 0
                end_idx = self.hps.batch_size
            else:
                batch = [x[start_idx:end_idx] for x in memories]
                start_idx = end_idx
                end_idx += self.hps.batch_size
            inputs = {k:v for k,v in zip(self.all_inputs_phs, batch)}
            _, kl = self.sess.run([self.train_pi_op, self.approx_kl], feed_dict=inputs)
            kl = mpi_avg(kl)
            if kl > 1.5 * self.hps.target_kl:
                print("early stopping at step {} due to reaching max KL".format(i)) #debug
                break
            
        for _ in range(self.hps.train_v_iters):
            self.sess.run(self.train_v_op, feed_dict=inputs)
        
        if logger is not None:
            pi_l_new, v_l_new, kl, cf = self.sess.run([self.pi_loss, self.v_loss, self.approx_kl, self.clipfrac], feed_dict=inputs)
            logger.add_named_value("Pi Loss", pi_l_new)
            logger.add_named_value("Value Loss", v_l_new)
            logger.add_named_value("KL Divergence", kl)
            logger.add_named_value("Clip Fraction", cf)

        

    def get_action_ops(self, obs):
        """
        Returns all action ops
        """
        a, v_t, logp_t = self.sess.run(self.action_ops, feed_dict={self.x_ph: obs.reshape(self.obs_reshape)})
        return a, v_t, logp_t

    def get_v(self, obs):
        """
        Returns output of the value network
        """
        return self.sess.run(self.v, feed_dict={self.x_ph: obs.reshape(self.obs_reshape)})

        