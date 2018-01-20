#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 20:29:50 2017

@author: virati
Library to generate synthetic data with a given profile
"""

#All recordings will be constant, with ~20 seconds, first 5 settling
import matplotlib.pyplot as plt
import numpy as np



def gen_recording():
    t = np.linspace(0,20,20 * 422)
    #disease signal
    #This is where the model comes in to generate activity related to disease
    
    #Make the actual signal
    x = Clin * np.sin(2 * np.pi * t)
    
    
    plt.plot(t,x)
    plt.show()
    
    return x
    
def H_brain_sig(Modl, Clin):
    

gen_recording()