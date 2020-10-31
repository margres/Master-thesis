#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 16:23:07 2020

@author: mrgr
"""

import numpy as np  
import scipy.optimize as op
import matplotlib.pyplot as plt

def PlotPotential(pot, fact, model_lens):
    
    x=np.linspace(0,100,100)
    
    if model_lens == 'SIS':
        pot=x
        
    elif model_lens== 'SIScore':
        
        a,b,c=fact[0],fact[1],fact[2]
        pot=a*(x**2/c**2+b**2)**(0.5)

    elif model_lens == 'powerlaw':
        
        E_r=1
        p=fact[3] #p=1 for SIS
        const=(E_r**(2-p)/p)
        pot=const*x**p
      
     
    elif model_lens == 'softenedpowerlaw':
        
        a,b,c=fact[0],fact[1],fact[2]
        p=fact[3]
        pot=a*(x**2/c**2+b**2)**(p/2) - a*b**p
     
        
    elif model_lens == 'softenedpowerlawkappa':
        
        #isothermal power law for p=0
        #modified Hubble model for p= 0
        #Plummer model for p =-2
        
        a,b,c,p=fact[0],fact[1],fact[2], fact[3]
        
        if p!=0:
            pot=a**(2-p)/(p*(x+1e-5)) * ((b**2+x**2)**(p/2) - b**p )
         
        else:
            pot=a**2/x * np.log(1 + x**2/b**2 )
                  
    x=np.linspace(0,100,100)
    plt.plot(x, pot)