import tensorflow as tf 
import numpy as np 
import scipy.signal 
from hp import HPStruct, DEFAULT_HPSTRUCT
from mpi import mpi_statistics_scalar, MpiAdamOptimizer, mpi_avg, num_procs, sync_all_params


#########################################
#                                       #
#          Utility Functions            #
#                                       #
#########################################
#"Magic from rllab"
def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

# def statistics_scalar(x):
#     x = np.array(x, dtype=np.float32)
#     n = x.size
#     mean = np.mean(x)
#     sum_sq = np.sum((x - mean)**2)

#     std = np.sqrt(sum_sq / n)
#     return mean, std

def gaussian_likelihood(x, mu, log_std, eps=1e-8, name="logp"):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+eps))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1, name=name)

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    if np.isscalar(shape):
        return (length, shape)
    return (length, *shape)

class PPOBuffer:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32) #this is wierd with discrete action spaces
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam 
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size 
    
    #Keyword only to avoid confusion (obs, act, rew are often single-letter variables in practice)
    def store(self, *, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs 
        self.act_buf[self.ptr] = act 
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val 
        self.logp_buf[self.ptr] = logp 
        self.ptr += 1

    #calculates terminal values 
    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr 

    def get(self):
        assert self.ptr == self.max_size #not sure about why this works
        self.ptr, self.path_start_idx = 0, 0

        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf]

class PPO():

    """
    x_ph is the x input placeholder
    y_ph is the output placeholder of an >ALREADY DEFINED< network (logits or mu depending on discrete/continuous)
    v_ph is the placeholder for the value network (generally the same framework as the y_ph network with a single output instead of the action vector)
    discrete indicates whether the output of the network is dicrete or not
        true will yield a categorical policy
        false will yield a gaussian policy
    hp_struct is a "struct" (dummy class) for holding hyperparameters
    save_path is directory to save the session to
    name is name to save with

    all dimensioning happens automatically
    Assumptions:
    placeholders have the correct dims
    y_ph is 2-dimensional, (batch, action_size) (1d action vectors) (the auto-dimensioning breaks if this doesn't hold true)
    x_ph can probably be whatever shape you want (might break buffer)
    v_ph (end of value network) should be size one, but there is no requirement that the network be the same structure as y_ph
        just that x_ph is the starting placeholder for both graphs
    if you don't pass in a hp_struct you're fine with default hyperparameters
    """
    def __init__(self, x_ph, y_ph, v_ph, discrete=True, hp_struct=None, save_path="./models/", name="ppo_model"):
        self.x_ph = x_ph 
        self.y_ph = y_ph
        #hps -> hyperparameters (most are used once so I choose to not expand them into their own variables)
        if hp_struct is None:
            self.hps = DEFAULT_HPSTRUCT()
        else:
            self.hps = hp_struct
        self.save_name = str(save_path + name + ".ckpt") 
        
        self.adv_ph = tf.placeholder(dtype=tf.float32, shape=(None), name="advantage")
        self.ret_ph = tf.placeholder(dtype=tf.float32, shape=(None), name="return")
        self.logp_old_ph = tf.placeholder(dtype=tf.float32, shape=(None), name="logp_old")

        with tf.compat.v1.variable_scope("pi"):
            if discrete: #categorical policy
                logits = y_ph #technicality
                logp_all = tf.nn.log_softmax(logits)
                self.pi = tf.squeeze(tf.multinomial(logits, 1), axis=1, name="pi")
                self.a_ph = tf.placeholder(dtype=tf.int32, shape=self.pi.get_shape(), name="action")
                act_dim = logits.get_shape()[-1]
                self.logp = tf.reduce_sum(tf.one_hot(self.a_ph, depth=act_dim) * logp_all, axis=1, name="logp")
                self.logp_pi = tf.reduce_sum(tf.one_hot(self.pi, depth=act_dim) * logp_all, axis=1, name="logp_pi")
            else: #gaussian policy
                mu = y_ph #technicality
                log_std = tf.compat.v1.get_variable(name='log_std', initializer=-0.5*np.ones(mu.get_shape()[-1], dtype=np.float32)) #
                std = tf.exp(log_std)
                self.pi = tf.identity(mu + tf.random.normal(tf.shape(mu)) * std, name="pi")
                self.a_ph = tf.placeholder(dtype=tf.float32, shape=self.pi.get_shape(), name="action")
                self.logp = gaussian_likelihood(self.a_ph, mu, log_std, name="logp")
                self.logp_pi = gaussian_likelihood(self.pi, mu, log_std, name="logp_pi")
        
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
        # if new:
        #     self.save()
        # else:
        #     self.restore()
        
        self.writer = tf.compat.v1.summary.FileWriter(save_path, self.sess.graph)
    
    def save(self):
        self.saver.save(self.sess, self.save_name)
        # print("Model Saved")
    
    def restore(self):
        self.saver.restore(self.sess, self.save_name)
        # print("Model Restored")

    #This handles batching
    #set batch_size in hps to None to use all full buffer
    def update(self):
        memories = self.buf.get()
        start_idx = 0
        end_idx = self.hps.batch_size

        # pi_l_old, v_l_old, end = self.sess.run([self.pi_loss, self.v_loss, self.approx_ent], feed_dict=inputs) #Logging stuff
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
                #TODO find a better way to print this
                # print("early stopping at step {} due to reaching max KL".format(i))
                break
            
        for _ in range(self.hps.train_v_iters):
            self.sess.run(self.train_v_op, feed_dict=inputs)
        
        #Log if desired

    def get_action_ops(self, obs):
        a, v_t, logp_t = self.sess.run(self.action_ops, feed_dict={self.x_ph: obs.reshape(self.obs_reshape)})
        return a, v_t, logp_t

    def get_v(self, obs):
        return self.sess.run(self.v, feed_dict={self.x_ph: obs.reshape(self.obs_reshape)})

        