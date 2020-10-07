
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:59:04 2020

@author: mrgr

###### script to calculate the magnification factor

#### it uses the FFT and the analytical FT




xs -> x-axis coordinates from the fit, time
ys -> y-axis coordinates from the fit, F
"""

import numpy as np
import numpy.polynomial.polynomial as poly
import scipy.interpolate as si
import matplotlib.pyplot as plt




def FFT(xs,ys, add_zeros='False'):
    
    '''
    It returns the FT of the ys values. 
    '''
    
    dt= xs[1]-xs[0]
    '''
    if add_zeros=='True':
        
        N_zeros = 1000
        t_tail = np.linspace(xs[-1]+dt, xs[-1]+dt*N_zeros, N_zeros)
        t_head = np.linspace(xs[0]-N_zeros*dt, xs[0]-dt, N_zeros)
        t_final = np.concatenate([t_head, xs, t_tail])
        Ftd_final = np.concatenate([np.zeros(N_zeros), ys, np.zeros(N_zeros)])
        N = len(t_final)
    else:
   
        N=len(xs)
        t_final=xs
        Ftd_final=ys
    '''
    N=ys.size + 1000
    # 4. FFT
    ## note: Ftd_final is real, so first half of the FFT series gives usable information
    ##      you can either remove the second half of ifft results, or use a dedicated function ihfft   
    Fw=np.fft.ihfft(ys, n=N) # I can add parameter n with value higher tan len(Ftd_final) and it will automatically create a padding of zeros
    ## multiply back what is divided
    Fw *= N
    ## multiply sampling interval to transfer sum to integral
    Fw *= dt
    freq = np.fft.rfftfreq(N,d=dt)
    omega = freq*2.*np.pi
    
    return omega[1:],Fw[1:]


def FT_clas(freq,T,mu, xs, ys):
    
    '''
    semi-calssical analytical contribution, eq. 34 and 39 -- Ulmer's paper 

    '''   
    if mu<0:
        #saddle point
        return 1j*np.exp(1j*freq*T)*np.sqrt(-mu)

    else:
        #min/max point
        return -1*np.exp(1j*freq*T)*np.sqrt(mu)
    

def Fd_w(xs,ys,t_ori,Ft_ori):
    
    '''
    eq. 5.6  
    
    Note that t_final and Ftd_final may contain zeros and the beginning and the end that's why we 
    use xs and ys on the last equation.
    '''
    
    omega,Fw=FFT(xs,ys)  
    #print(omega.size,Fw.size)
    Fw = Fw*omega/(2j*np.pi)-Ft_ori[0]*np.exp(1j*omega*t_ori[0])/2/np.pi
    #Fw = Fw*omega/(2j*np.pi)#-Ft_ori[0]*np.exp(1j*omega*t_ori[0])/2/np.pi
    
    return omega,Fw


def Hanning_smooth(t,Ft):
    
    print('Applying hanning smooth')
    
    #mask=t>2 #when to start using the window function
    mask=t>1
    
    signal=Ft[mask]
    window_len=100 #decide the lenght of the window
    s=np.r_[signal[window_len-1:0:-1],signal,signal[-2:-window_len-1:-1]]
    print(len(s))
    w=np.hanning(window_len)
    print(len(w))
    
    y=np.convolve(w/w.sum(),s,mode='valid')
    #y=w/w.sum()*s
    y=y[int((window_len/2-1)):-int((window_len/2))]
    
    Ft_wind=np.r_[Ft[~mask],y]
    print('Hanning smooth done')
    
    return Ft_wind



