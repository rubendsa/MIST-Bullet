import tensorflow as tf 
import numpy as np 
from collections import deque

class Trajectory():

    def __init__(self, disc_fctr=0.99):
        self.disc_fctr = disc_fctr
        self.clear()

    def clear(self):
        self.states = deque()
        self.actions = deque()
        self.noises = deque()
        self.costs = deque()
        self.values = deque()
        self.accumCost = deque()
        self.terminal_val = 0.0

    #all deques get pushed simultaneously except values and accumCost
    def push_back(self, state, action, noise, cost):
        self.states.append(state)
        self.actions.append(action)
        self.noises.append(noise)
        self.costs.append(cost)

    def get_terminal_state(self):
        return self.states[-1]
    
    def terminate_traj_and_update_val(self, state, action, terminal_val, discount_factor):
        self.terminal_val = terminal_val
        self.disc_fctr = discount_factor
        self.states.append(state) #terminal state 
        self.actions.append(action) #terminal action TODO figure out if None is better see traj.hpp
        self.costs.append(None) #this indicates the end of array 
        self.noises.append(None) #same 

        trajLength = len(self.states)
        trajCounter = trajLength - 1

        self.values = deque()
        self.accumCost = deque()
        for i in range(trajLength):
            self.values.append(None)
            self.accumCost.append(None)
        
        self.values[trajCounter] = terminal_val 
        self.accumCost[trajCounter] = None #this is redundant TODO remove

        while trajCounter > 0:
            trajCounter -= 1
            self.values[trajCounter] = self.disc_fctr * self.values[trajCounter + 1] + self.costs[trajCounter]
            if self.accumCost[trajCounter + 1] is not None: #TODO figure out if this works
                self.accumCost[trajCounter] = self.disc_fctr * self.accumCost[trajCounter + 1] + self.costs[trajCounter]
            else:
                self.accumCost[trajCounter] = self.costs[trajCounter]
        
    def size(self):
        return len(self.states)
    
    def update_val_traj_with_new_terminal_val(self, terminal_val, disc_fctr):
        self.terminal_val = terminal_val
        self.disc_fctr = disc_fctr

        self.values = deque()
        self.accumCost = deque()
        for i in range(self.size()):
            self.values.append(None)
            self.accumCost.append(None)

        decayedTerminalValue = self.terminal_val * self.disc_fctr
        trajCounter = self.size() - 1
        self.accumCost[trajCounter] = None 
        self.values[trajCounter] = self.terminal_val 

        trajCounter -= 1

        while trajCounter > -1:
            if self.accumCost[trajCounter + 1] is not None: #TODO figure out if this works
                self.accumCost[trajCounter] = self.disc_fctr * self.accumCost[trajCounter + 1] + self.costs[trajCounter]
            else:
                self.accumCost[trajCounter] = self.costs[trajCounter]
            self.values[trajCounter] = self.accumCost[trajCounter] + decayedTerminalValue

            trajCounter -= 1

            decayedTerminalValue *= self.disc_fctr

    def __str__(self):
        return "Trajectory of size {}\n".format(self.size())








