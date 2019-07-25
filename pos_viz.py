from pybulletinstance import PyBulletInstance 
import pybullet
import time
import utils 
import numpy as np

"""
Use this to vizualize what the Enviornment Utils from utils.py do
"""
env = PyBulletInstance(GUI=True)

pos = [0, 0, 0]
q = [0, 0, 0, 1]
rpy = [0, 0, 0]

env.client.setGravity(0, 0, 0.0)
for i in range(200):
    env.applyAction([0, 0, 0, 0, 0, 0, 0, 0, 1.57, 1.57, 1.57])
    env.step()


setpoint = [1, 1, 1]
origin = [0, 0, 0]

force_mag = 100
xf_id = pybullet.addUserDebugParameter("x_force", -force_mag, force_mag, 0)
yf_id = pybullet.addUserDebugParameter("y_force", -force_mag, force_mag, 0)
zf_id = pybullet.addUserDebugParameter("z_force", -force_mag, force_mag, 0)

while True:
    f = utils.random_force(2000)
    env.applySingleLinkForce(f)
    for i in range(1000):
        xf = pybullet.readUserDebugParameter(xf_id)
        yf = pybullet.readUserDebugParameter(yf_id)
        zf = pybullet.readUserDebugParameter(zf_id)
        force = [zf, yf, zf]
        env.applySingleLinkForce(force)

        env.step()
        o = env.getState()
        pos, velocity_U, offset_U = utils.fixed_wing_reward(o, setpoint, debug=True)
        env.addDebugLine(origin, velocity_U)
        env.addDebugLine(origin, offset_U, color=[1,0,0])
        env.addDebugLine(origin, setpoint, color=[0,1,0])

        time.sleep(env.getVizDelay())
    env.reset()