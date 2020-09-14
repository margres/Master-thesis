
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:59:04 2020

@author: mrgr

###### script to calculate the magnification factor

#### it uses the FFT and the analytical FT

## c_omega=freqs/(2j*np.pi)



xs -> x-axis coordinates from the fit, time
ys -> y-axis coordinates from the fit, F
"""

import numpy as np  

def FFT(xs,ys):
    '''
    It returns the FT of the ys values. 
    '''
    
    #amount_zeros=1000
    #zeros=np.zeros(amount_zeros)
    #ys=np.concatenate((zeros,ys,zeros))
    omega = np.fft.fftfreq(len(ys),d=xs[1]-xs[0])
    FT=np.fft.ifft(ys, norm='ortho')/(len(ys))
    
    return omega,FT


def FT_clas(omega,T,mu, xs, ys):
    '''
    semi-calssical analytical contribution, eq. 34 and 39 -- Ulmer's paper 

    '''   
    if mu<0:
        return 2j*np.exp(1j*omega*T)*np.sqrt(-mu)
    else:
        return -2*np.exp(1j*omega*T)*np.sqrt(mu)
    

def F_d(xs,ys):
    '''
    eq. 5.6  
    '''
    
    omega,FT=FFT(xs,ys)
    
    c_omega=omega*1j/(np.pi)
     
    first_term= np.exp(1j*omega*xs[-1])*ys[-1]-np.exp(1j*omega*xs[0])*ys[0]
    
    return omega,( first_term - 1j*omega*FT )/c_omega




