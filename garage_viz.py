from garage.experiment import run_experiment 
from garage.tf.experiment import LocalTFRunner 
from garage.tf.envs import TfEnv

from pybulletenv import PyBulletEnvironment 

res_dir = "/home/deanx252/Research/MIST-Bullet/data/local/experiment/experiment_2019_08_20_16_42_16_0,001"

def run_task(snapshot_config, *_):
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(PyBulletEnvironment(GUI=True))
        runner.env = env
        runner.restore(from_dir=res_dir)
        runner.resume()

run_experiment(
    run_task,
    python_command="python3",
    snapshot_mode='last',
    seed=1
)