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
from scipy.optimize import fsolve

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

        self.aoi = 0

        # number of shadow test points
        self.num_shadow_tests = 100

        self.battery_power = []
        self.motors_power = 0
        self.length = 0
        self.max_thrust = 0

        # 2 amp hours, 14.8v, --> P = V*I [W*s]
        self.initial_battery = 2.0*60*14.8

        # length of timestep - default timestep is 1/240 second
        self.time_step = 1.0/240.0

        # circuit simulation
        self.R0 = 10
        self.R1 = 10

        self.C = 1e-1
        self.SOC = 100

        self.alpha = 1 # % const relating SOC and Voc set this
        self.k_T = 1 # % motor torque const.
        self.J_T = 1 # % motor torque const.
        self.k_e = 1 # % motor voltage const.
        self.k_T2 = 1 # %motor torque const. (friction/drag)

        self.motor_efficiency = 0.90
        self.MPPT_efficiency = 0.90

        self.wm_c = [0.0,0.0,0.0,0.0] # current motor speeds
        self.wm_p = [0.0,0.0,0.0,0.0] # previous motor speeds

        self.i1 = 0.5
        self.i2 = 0.5
        self.i3 = 0.5
        self.VO = 16.0

        self.sum_i3  = 0.0

        self.x0 = [self.i1, self.i2, self.i3, self.VO]

        self.temp_solar_power = 0

    def power_calc(self, solar_power, w_motors):
        self.wm_p = self.wm_c
        self.wm_c = w_motors

        self.temp_solar_power = solar_power*self.MPPT_efficiency

        # fsolve
        self.x0 = fsolve(self.power_fun, self.x0)
        
        self.i1 = self.x0[0]
        self.i2 = self.x0[1]
        self.i3 = self.x0[2]
        self.VO = self.x0[3]

        self.SOC = self.SOC - self.x0[1]*self.time_step
        self.sum_i3 = self.sum_i3 + self.x0[3]*self.time_step



    def power_fun(self, x):
        i1 = x[0]
        i2 = x[1]
        i3 = x[2]
        VO = x[3]
     

        dwdt = []
        torque = []
        i_motors = []
        v_motors = []
        Pmotors = 0
        for i in range(4):
            dwdt.append((self.wm_c[i] - self.wm_p[i])/self.time_step)
            tq = self.J_T*dwdt[i] + self.wm_c[i]*self.k_T2

            # cannot have negative torque, motors cannot charge battery
            if tq < 0:
                tq = 0

            torque.append(tq)
            i_motors.append(tq/self.k_T)
            v_motors.append(self.k_e*self.wm_c[i])
            Pmotors += i_motors[i]*v_motors[i]

        Psolar = self.temp_solar_power
        
        # power from motors
        Pmotors =  Pmotors*self.motor_efficiency

        # current from ESCs (motors)
        i4 = Pmotors/VO

        # current from MPPT (solar cells)
        i5 = Psolar/VO
        
        # Voc from SOC
        Voc = self.SOC*self.alpha

        # equations (equal to 0)
        z0 = -i1 + i4 - i5
        z1 = i1 - i2 -i3
        z2 = Voc - self.R0*i1 - self.R1*i2 - VO
        z3 = (1./(self.R1*self.C))*self.sum_i3 - i2

        z = [z0,z1,z2,z3]
        return z







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
    
    def getLinkPosition(self, robotId, linkIndex):
        if linkIndex == -1:
            pose = p.getBasePositionAndOrientation(robotId)
            return pose[0]
        else:
            a, b, c, d, e, f, g, h = p.getLinkState(robotId, linkIndex, 1)
            pos = e
        return pos

    def convert_RPY_to_TA(self,orientation):
        roll = orientation[0]
        pitch = orientation[1]
        yaw = orientation[2]

        # assumes radians
        # el = math.asin(math.sin(roll)*math.sin(pitch))
        # az = math.atan(math.cos(roll)*math.tan(pitch)) #TODO: atan2?

        # az = 0
        az = yaw

        num = math.sin(pitch) + math.sin(roll)
        den =math.sqrt(2 + 2*math.sin(pitch)*math.sin(roll))

        # el = math.pi
        el = math.asin(num/den)
         

        # convert to degrees
        rotation = [az*self.to_degrees, el*self.to_degrees]
        return rotation

    # from thrust
    def set_motors_power(self, m):
        # mt28 14 motors 770 kv 
        # assuming prop = 11*3.7CF
        # a = 107.083685545224 # linear fit for now

        a = 249.976491970077
        b = 43.5675424329268
        # calculate power the motors are using, in [W]
        # scale thrust to be 0 to 1
        p = 0
        for value in m:
            if value > self.max_thrust:
                self.max_thrust = value
        for value in m:
            thrust = value/self.max_thrust
            p = p + a*thrust*thrust + b*thrust
        self.motors_power = p # W

    def update_battery(self):
        if len(self.battery_power) == 0:
            for i in range(len(self.power)):
                self.battery_power = self.battery_power + [self.initial_battery]

        for i in range(len(self.battery_power)):
            self.battery_power[i] = self.battery_power[i] + self.time_step*self.power[i] # add solar power
            self.battery_power[i] = self.battery_power[i] - self.time_step*self.motors_power # subtract power from motors
       


        

    def calculate_power(self):
        # self.update_surfaces()

        # for each surface with panels
        for i in range(0,4):
            surface_tilt = self.surfaceT[i]
            surface_azimuth = self.surfaceA[i]
            # I_eff - effective irradiance [W/m^2] https://pvlib-python.readthedocs.io/en/latest/_modules/pvlib/irradiance.html
            self.eff_irrad[i] = pvlib.irradiance.total_irrad(surface_tilt, surface_azimuth, self.solpos['apparent_zenith'], self.solpos['apparent_azimuth'], self.solis['dhi'], self.solis['dni'], self.solis['ghi'], self.dni_extra, model='reindl')
            module_parameters = {"a_ref":self.n_s*0.027141253, "I_L_ref":6.17, "I_o_ref":1.262*pow(10,-11), "R_sh_ref":425.0*self.n_s/36, "R_s":0.08382*self.n_s/36}
            temp_cell, alpha_isc, EgRef, dEgdT = self.T, 0.00054, 1.124, -0.0002677
            params = pvlib.pvsystem.calcparams_desoto(self.eff_irrad[i]['poa_global'], temp_cell, alpha_isc, module_parameters, EgRef, dEgdT,self.M,self.irrad_ref,self.temp_ref);
            iv_curve = pvlib.pvsystem.singlediode(params[0], params[1], params[2], params[3], params[4], 100)
            where_are_NaNs = numpy.isnan(iv_curve['p_mp'])
            iv_curve['p_mp'][where_are_NaNs] = 0
            self.P_mpp[i] = iv_curve['p_mp']
            # IV curve vs eff_irrad? - 
        self.aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth, self.solpos['apparent_zenith'], self.solpos['apparent_azimuth']) #aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
        # self.aoi = pvlib.irradiance.aoi(0.0, 0.0, self.solpos['apparent_zenith'], self.solpos['apparent_azimuth']) #aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)

        # adding up the surfaces
        power1 = [x+y+z+a for x,y,z,a in zip(self.P_mpp[0], self.P_mpp[1], self.P_mpp[2], self.P_mpp[3])]
        # this is a list of power values throughout a day
        self.power = power1
        self.length = len(self.power)

    # def updatePos(self, euler):
    #     self.roll = euler[0]
    #     self.pitch = euler[1]
    #     self.yaw = euler[2]

# print("Test")
# test = solar()
# test.calculate_power()
# print(test.power)
# print(test.eff_irrad[0])