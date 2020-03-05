#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:02:48 2020

@author: mrgrà
"""

import numpy as np 
import scipy.special as ss 
import scipy.integrate as integrate




# test
if __name__ == "__main__":

    Mlz = 1000.0 # solar mass
    y = 1.0
    N = 1000 # the number of sample values
    # frequence 
    fmin = 10.0
    fmax = 1000.0
    f = np.linspace(fmin, fmax, N)

    # w
    Mlz *= 4.92549059e-6
    w = np.loadtxt('w.txt',delimiter='\n')
    #w = 8*np.pi*Mlz*f
    

    #Fw_list = PointMassFunc(w, y)
    #print(Fw_list)
    SIS_list=SIS(Mlz,w,y)
    #print(abs(SIS_list))
    
    np.savetxt('try_10_abs.txt', SIS_list.real , delimiter=' ', fmt='%d')
    #np.savetxt('w_try.txt', abs(SIS_list) , delimiter=' ', fmt='%d')
