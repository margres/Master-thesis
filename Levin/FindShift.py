#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 11:11:15 2020

@author: mrgr
"""

import numpy as np 
import scipy.optimize as op
import scipy.special as ss
import mpmath 


def FindCrit(y,lens_model,fact):


    N= 100
    rmin = 0.001
    rmax = 5.0
    xt = np.linspace(rmin,rmax,N)
    xt_res=[]
    
    for i in range(N-1):
        if eval(lens_model)(xt[i],y,fact)* eval(lens_model)(xt[i+1],y,fact)<=0:
            tmp=op.brentq(eval(lens_model), xt[i],xt[i+1], args=(y,fact))
            xt_res.append(tmp)
            #print('x minimum: ',tmp)
            #break
            #try 
    try:
        foo = tmp
    except NameError:
        tmp = 0
        print('Could not find the value of the first image, using 0 instead')
        
    return xt_res


def SIScore (x,y,fact):
    
    a=fact[0]
    b=fact[1]
    c=fact[2]
    
    #derivative of the 1D potential
    phip= a*x/(c**2*np.sqrt(b**2+x**2/c**2))
    #derivative of the time delay
    tp= abs(x - y) - phip
    
    return tp

def point (x,y,fact):    
    
    #derivative of the 1D potential
    phip= 1/x
    #derivative of the time delay
    tp= abs(x - y) - phip
    
    return tp


def softenedpowerlaw(x,y,fact):
    
    a,b,c,p=fact[0], fact[1], fact[2],fact[3]
    
    phip= a*p*x*(b**2 + x**2)**(p/2 - 1) 
    
    tp= abs(x - y) - phip
    
    return tp

def softenedpowerlawkappa(x,y,fact):
    
    a,b,p=fact[0], fact[1],fact[3]
    
    if p==0 and b!=0:        
        phip=a**2/x * np.log(1+ x**2/b**2)           
    elif p<4:         
        phip=a**(2-p)/(p*x) * (x**p*(1+b**2/x**2)**(p/2) - b**p)
        #phip= a**(2 - p)/(p*x)  * ((b**2+x**2)**p/2 - b**p)
    else:
        raise Exception('Unsupported lens model')
    
    tp= abs(x - y) - phip
    
    return tp

def TimeDelay(x,y,fact,lens_model):
    
    a,b,c,p=fact[0], fact[1], fact[2],fact[3]
    
    if x==0:
        phi=0
        
    elif lens_model == 'SIScore':
        phi =  a * np.sqrt(b**2+x**2/c**2) #SIS
    elif lens_model == 'point':
        phi = np.log(x)
  
    elif lens_model == 'softenedpowerlaw':
        phi=a*(x**2/c**2+b**2)**(p/2) - a*b**p
        
    elif lens_model == 'softenedpowerlawkappa':        
     
        if p>0 and b==0:            
            phi= 1/p**2 * a**(2-p) *x**p
                 
        elif b!=0 and p!=0:
            if x==0:
                t1=0
            else:
                t1= 1/p**2 * a**(2-p)*x**p *ss.hyp2f1(-p/2, -p/2, 1-p/2, -b**2/x**2)

            #print(ss.hyp2f1(-p/2, -p/2, 1-p/2, -b**2/x**2))
            t2= 1/p*a**(2-p)*b**p*np.log(x/b)
            t3= 1/(2*p) * a**(2-p)*b**p*(np.euler_gamma-ss.digamma(-p/2))   
            phi= t1 - t2 - t3
            
        elif p==0 and b!=0:
            phi=-1/2 * a**2* mpmath.polylog(2,x**2/b**2)
      
            
    else:
        raise Exception("Unsupported lens model !")
 
    psi_m = -(0.5*(abs(x-y))**2 - phi)
    
    return psi_m

def FirstImage(y,fact,lens_model):
    
    '''
    We have to scale everything in respect to rhe first image.
    If we have one more value of x at which we have images we 
    need to realize which is the one related to the first 
    (in the time domain) image.
    '''
    
    xlist=FindCrit(y,lens_model,fact)
    tlist=[]
    try: 
        for x in xlist:
            tlist.append (TimeDelay(x,y,fact,lens_model))
            t=np.min(tlist)
    except:
        
        t=TimeDelay(x,y,fact,lens_model)
        
    return t





if __name__ == '__main__':
    
    a=1
    b=0.5
    c=1
    p=1.5
    fact=[a,b,c,p]
    y=0.3
    lens_model='softenedpowerlaw'
    
    
    t = FirstImage(y,fact,lens_model)
    print('phi_m:',t)