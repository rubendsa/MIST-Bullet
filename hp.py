"""
The goal of this is that you can create a new set of hyperparameters and contain it here
Inheritance can be used in cases where only a few modifications between sets is needed
If a class is constructed by omitting the parentheses -> DEFAULT_HPSTRUCT instead of DEFAULT_HPSTRUCT()
The __str__ will not perform as desired
"""

class HPStruct():
    steps_per_epoch = None
    epochs = None 
    gamma = None 
    clip_ratio = None 
    pi_lr = None 
    vf_lr = None 
    train_pi_iters = None 
    train_v_iters = None 
    lam = None 
    max_ep_len = None 
    target_kl = None 
    save_freq = (((((None)))))
    batch_size = None #Leaving this as None will use full buffer

    """
    Returns a string of all values from the base class (override if child class has new values to print)
    For debugging
    """
    def __str__(self):
        return "Steps per Epoch : {}\n".format(self.steps_per_epoch) \
             + "Epochs : {}\n".format(self.epochs) \
             + "Gamma : {}\n".format(self.gamma) \
             + "Clip Ratio : {}\n".format(self.clip_ratio) \
             + "Pi LR : {}\n".format(self.pi_lr) \
             + "Vf LR : {}\n".format(self.vf_lr) \
             + "Train Pi Iterations : {}\n".format(self.train_pi_iters) \
             + "Train Vf Iterations : {}\n".format(self.train_v_iters) \
             + "Lambda : {}\n".format(self.lam) \
             + "Max Ep Length : {}\n".format(self.max_ep_len) \
             + "Target KL : {}\n".format(self.target_kl) \
             + "Save Frequency : {}\n".format(self.save_freq) \
             + "Batch Size : {}\n".format(self.batch_size)

class DEFAULT_HPSTRUCT(HPStruct):
    steps_per_epoch = 4000
    epochs = 10000
    gamma = 0.99
    clip_ratio = 0.2
    pi_lr = 3e-4
    vf_lr = 1e-3
    train_pi_iters = 80
    train_v_iters = 80
    lam = 0.97
    max_ep_len = 1000
    target_kl = 0.01 
    save_freq = 10
    batch_size = 100

class DEBUG_HPSTRUCT(DEFAULT_HPSTRUCT):
    steps_per_epoch = 10
    epochs = 50
    save_freq = 1 

class ATARI_HPSTRUCT(DEFAULT_HPSTRUCT):
    steps_per_epoch = 4000
    epochs = 5000
    pi_lr=2.5e-4
    train_pi_iters = 80
    train_v_iters = 80
    batch_size = 50
    lam = 0.95
    save_freq = 10

class QUADROTOR_HPSTRUCT(DEFAULT_HPSTRUCT):
    steps_per_epoch = int(10e4)
    epochs = int(10e4)
    gamma = 0.99
    clip_ratio = 0.2
    pi_lr = 3e-4
    vf_lr = 1e-3
    train_pi_iters = 80
    train_v_iters = 80
    lam = 0.97
    max_ep_len = 1000
    target_kl = 0.02
    save_freq = 10
    batch_size = int(10e3)



if __name__ == "__main__":
    y = HPStruct()
    print(y.epochs)
    print(y.pi_lr)
    x = DEFAULT_HPSTRUCT() 
    print(x)

    print(x.steps_per_epoch)
    print(x.lam)
    print(x.potatoe) #this should throw an error