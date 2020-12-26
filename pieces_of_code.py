#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 10:49:34 2020

@author: mrgr
"""

 
    text=np.arange(tau_list[-1],10,np.diff(t_new)[0])
    asymptote=geom_optics(text) 
    mask_asymptote=Ft_new<Ft_new
    plt.plot(text,asymptote, label='geom')
    #plt.legend() 
    #plt.show()
    
    Ft_new = np.concatenate([Ft_list,asymptote])
    t_new= np.concatenate([tau_list,text])
    #plt.plot(t_new,Ft_new, label='final')