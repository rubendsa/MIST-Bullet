"""
The goal of this is that you can create a new set of hyperparameters and contain it here
Inheritance can be used in cases where only a few modifications between sets is needed
If a class is constructed by omitting the parentheses -> DEFAULT_HPSTRUCT instead of DEFAULT_HPSTRUCT()
The __str__ will not perform as desired
TODO do this as a dict
"""

class HPStruct():
    rollout_length = None 
    rollouts_per_epoch = None 
    batch_size = None 
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
    log_freq = None 


class DEFAULT_HPSTRUCT(HPStruct):
    rollout_length = 4000
    rollouts_per_epoch = 1 
    epochs = 10000
    gamma = 0.99 #discount
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


class QUADROTOR_HPSTRUCT(DEFAULT_HPSTRUCT):
    rollout_length = 1250
    rollouts_per_epoch = 200
    batch_size = rollout_length * rollouts_per_epoch
    # epochs = int(10e6 / batch_size)
    epochs = int(10e6)
    gamma = 0.99
    clip_ratio = 0.2
    pi_lr = 3e-4
    vf_lr = 1e-3
    train_pi_iters = 80
    train_v_iters = 80
    lam = 0.97
    max_ep_len = 1000
    target_kl = 0.015
    save_freq = 10



if __name__ == "__main__":
    y = HPStruct()
    print(y.epochs)
    print(y.pi_lr)
    x = DEFAULT_HPSTRUCT() 
    print(x)

    print(x.steps_per_epoch)
    print(x.lam)
    print(x.potatoe) #this should throw an error