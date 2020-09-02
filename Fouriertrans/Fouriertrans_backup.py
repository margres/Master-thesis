#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:59:04 2020

@author: mrgr

###### script to calculate the magnification factor

#### it uses the FFT and the analytical FT

## c_omega=freqs/(2j*np.pi)

"""

import numpy as np  

def FFT_direct(xs,ys,bins,n_sample=100):
    '''
    It returns the FT of the ys values. 
    '''
    
    amount_zeros=1000
    zeros=np.zeros(amount_zeros)
    ys_with_zeros=np.concatenate((zeros,ys,zeros))
    omega = np.fft.fftfreq(len(ys_with_zeros), d=(np.max(xs)- np.min(xs))/(n_sample))
    FT=np.fft.ifft(ys_with_zeros) #*len(ys_with_zeros)  # check the 2pi 
    
    return omega,FT

def FT_saddle(omega,T,mu):
    '''
    analytical contribution of saddle points
    '''
    
    return 2j*np.exp(1j*omega*T)*mu

def FT_minimum(omega,T):
    '''
    analytical contribution of minimum points
    '''
    
    return -2*np.exp(1j*omega*T)

def F_c(omega,T,mu):
    
    '''
    total contribution of the crtitical points
    
    '''
    
    return FT_saddle(omega,T,mu)+FT_minimum(omega,T)

def F_d(xs,ys,bins,omega):
    '''
    eq. 5.6  
    '''
    c_omega=omega/(2j*np.pi)
    
    first_term= np.exp(1j*omega*xs[-1])*ys[-1]-np.exp(1j*omega*xs[0])*ys[0]
    
    return c_omega*( first_term - 1j*omega*FFT(xs,ys,bins) )




