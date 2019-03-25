import numpy as np
import pybullet as p
import time
import pybullet_data

import sys
import pvlib
import pandas as pd
import numpy
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import operator
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.dates as mdates
import matplotlib
import math

class solar:
    def __init__(self):
        # Added these parameters so calcparams_desoto would run
        self.M = 1
        self.irrad_ref = 1000
        self.temp_ref = 25

        self.latitude = 44.933
        self.longitude = -93.05
        self.tz = 'US/Central'
        self.altitude = 220
        self.name = 'East River Flats'
        self.date = '9/22/2016'
        self.times = pd.date_range(self.date, periods=1440, freq='1min')
        self.times = self.times.tz_localize(self.tz)
        self.T = 25.0 #temperature of the cells [K]
        self.n_s = 8
        self.to_degrees = 180/math.pi

        # default tilt and angle = quadrotor state
        # outside surfaces, 4 of them
        self.surfaceT = [90.0,90.0,90.0,90.0] # tilt (elevation)
        self.surfaceA = [0.0,90.0,180.0,270.0] # azimuth

        self.solpos = pvlib.solarposition.get_solarposition(self.times, self.latitude, self.longitude, altitude=self.altitude, method='pyephem')
        self.apparent_elevation = self.solpos['apparent_elevation']
        self.aod700= 0.105 # 0.09 0.061 0.147
        self.precipitable_water = 1.6 # 1 0.8 3.5
        self.pressure = pvlib.atmosphere.alt2pres(self.altitude)
        self.dni_extra = pvlib.irradiance.extraradiation(self.times.dayofyear, method='pyephem')
        self.solis = pvlib.clearsky.simplified_solis(self.apparent_elevation, self.aod700, self.precipitable_water, self.pressure, self.dni_extra)

        # of whole robot
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        
        # calculated throughout the day

        # total
        self.power = []

        # each surface
        self.P_mpp = [[],[],[],[],[],[],[],[]]
        self.eff_irrad = [[],[],[],[],[],[],[],[]]

        self.o0 = 0
        self.o1 = 0
        self.o2 = 0
        self.o3 = 0



    def updateSurfaces(self, robotId):

        # for each section:
        self.o0 = p.getEulerFromQuaternion(self.getLinkOrientation(robotId,  -1)) #may need to be -1 to 2?
        self.o1 = p.getEulerFromQuaternion(self.getLinkOrientation(robotId,  0))
        self.o2 = p.getEulerFromQuaternion(self.getLinkOrientation(robotId,  1))
        self.o3 = p.getEulerFromQuaternion(self.getLinkOrientation(robotId,  2))

        TA0 = self.convert_RPY_to_TA(self.o0)
        TA1 = self.convert_RPY_to_TA(self.o1)
        TA2 = self.convert_RPY_to_TA(self.o2)
        TA3 = self.convert_RPY_to_TA(self.o3)

        # TA = [A T]
        self.surfaceA = [TA0[0],TA1[0],TA2[0],TA3[0]]
        self.surfaceT = [TA0[1],TA1[1],TA2[1],TA3[1]]


    def getLinkOrientation(self, robotId, linkIndex):
        if linkIndex == -1:
            pose = p.getBasePositionAndOrientation(robotId)
            return pose[1] # Quaternion
        else:
            a, b, c, d, e, f, g, h = p.getLinkState(robotId, linkIndex, 1)
            orientation = f # Quaternion
        return orientation

    def convert_RPY_to_TA(self,orientation):
        roll = orientation[0]
        pitch = orientation[1]
        yaw = orientation[2]

        # assumes radians
        # el = math.asin(math.sin(roll)*math.sin(pitch))
        # az = math.atan(math.cos(roll)*math.tan(pitch)) #TODO: atan2?

        az = yaw

        num = math.sin(pitch) + math.sin(roll)
        den =math.sqrt(2 + 2*math.sin(pitch)*math.sin(roll))

        el = math.asin(num/den)
         

        # convert to degrees
        rotation = [az*self.to_degrees, el*self.to_degrees]
        return rotation

    def calculate_power(self):
        # self.update_surfaces()

        # for each surface with panels
        for i in range(0,4):
            surface_tilt = self.surfaceT[i]
            surface_azimuth = self.surfaceA[i]
            # I_eff - effective irradiance [W/m^2]
            self.eff_irrad[i] = pvlib.irradiance.total_irrad(surface_tilt, surface_azimuth, self.solpos['apparent_zenith'], self.solpos['apparent_azimuth'], self.solis['dhi'], self.solis['dni'], self.solis['ghi'], self.dni_extra, model='reindl')
            module_parameters = {"a_ref":self.n_s*0.027141253, "I_L_ref":6.17, "I_o_ref":1.262*pow(10,-11), "R_sh_ref":425.0*self.n_s/36, "R_s":0.08382*self.n_s/36}
            temp_cell, alpha_isc, EgRef, dEgdT = self.T, 0.00054, 1.124, -0.0002677
            params = pvlib.pvsystem.calcparams_desoto(self.eff_irrad[i]['poa_global'], temp_cell, alpha_isc, module_parameters, EgRef, dEgdT,self.M,self.irrad_ref,self.temp_ref);
            iv_curve = pvlib.pvsystem.singlediode(params[0], params[1], params[2], params[3], params[4], 100)
            where_are_NaNs = numpy.isnan(iv_curve['p_mp'])
            iv_curve['p_mp'][where_are_NaNs] = 0
            self.P_mpp[i] = iv_curve['p_mp']
            # IV curve vs eff_irrad? - 

        # adding up the surfaces
        power1 = [x+y+z+a for x,y,z,a in zip(self.P_mpp[0], self.P_mpp[1], self.P_mpp[2], self.P_mpp[3])]
        # this is a list of power values throughout a day
        self.power = power1

    # def updatePos(self, euler):
    #     self.roll = euler[0]
    #     self.pitch = euler[1]
    #     self.yaw = euler[2]

# print("Test")
# test = solar()
# test.calculate_power()
# print(test.power)
# print(test.eff_irrad[0])