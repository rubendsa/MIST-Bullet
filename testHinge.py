import numpy as np
from numpy import linalg as LA
from numpy import loadtxt
import pybullet as p
import time
import pybullet_data
import matplotlib.pyplot as plt
import math

# Functions for Transformer UAV dynamics and control
import wingDynamics
import forceMoment as fm
import controllers as ctrl
import record
import helperFunctions as hf

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # Removes the GUI text boxes
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.81)
planeId = p.loadURDF("plane.urdf")