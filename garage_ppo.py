from garage.experiment import run_experiment 
from garage.tf.algos import PPO 
from garage.tf.baselines import GaussianMLPBaseline 
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner 
from garage.tf.policies import GaussianMLPPolicy 

from pybulletenv import PyBulletEnvironment 
import tensorflow as tf 

# import gym 

# gym.envs.register(id='PyBulletEnvironment-v0')
# env = gym.make("PyBulletEnvironment-v0")
# print("potatoe")

# env.close()



def run_task(snapshot_config, *_):
    with LocalTFRunner(snapshot_config=snapshot_config, max_cpus=12) as runner:
        env = TfEnv(PyBulletEnvironment())

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                hidden_sizes=(32, 32),
                use_trust_region=True,
            ),
        )

        algo = PPO(
            env_spec=env.spec, 
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            optimizer_args=dict(
                batch_size=32,
                max_epochs=10,
            ),
            stop_entropy_gradient=True,
            entropy_method='max',
            policy_ent_coeff=0.02,
            center_adv=False,
        )

        runner.setup(algo, env)

        runner.train(n_epochs=int(10e6), batch_size=10000, plot=False)


run_experiment(run_task, python_command="python3", snapshot_mode='last', seed=1)