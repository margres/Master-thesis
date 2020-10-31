#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:33:09 2020

@author: mrgr

Functions to plot critical curves and caustics
"""
import scipy.optimize as op
import numpy as np 
import matplotlib.pyplot as plt


def DistMatrix( dpsid11,dpsid22,dpsid12, kappa,gamma):
    
    # Jacobian matrix
    j11 = 1. - kappa - gamma - dpsid11
    j22 = 1. - kappa + gamma - dpsid22
    j12 = -dpsid12
    detA=j11*j22-j12*j12
    
    return detA
    


def SIScore(x12,kappa,gamma,fact, caustics=False):
    
    a=fact[0]
    b=fact[1]
    c=fact[2]
    

    x1=x12[0]
    x2=x12[1]
    
    #first derivative

    dx12dx22 = np.sqrt((x1**2.+x2**2.)/c**2 + b**2)
    dpsi1 = a*x1/dx12dx22/c**2
    dpsi2 = a*x2/dx12dx22/c**2
    
    if caustics==True:
        
        x1_per = x1*(kappa+gamma)
        x2_per = x2*(kappa-gamma)
        
        return np.column_stack((x1_per+dpsi1, x2_per+dpsi2))
        #return npdpsi1,dpsi2
    
    
    
    #second derivatives 
    
    dx12pdx22_32 = ((x1**2.+ x2**2.)/c**2+b**2)**(3/2)
    # d^2psi/dx1^2
    dpsid11 = a*(x2**2+b**2*c**2)/dx12pdx22_32/c**4
    # d^2psi/dx2^2
    dpsid22 = a*(x1**2+b**2*c**2)/dx12pdx22_32/c**4
    # d^2psi/dx1dx2
    dpsid12 = -a*(x1*x2)/dx12pdx22_32/c**4

    detA = DistMatrix(dpsid11,dpsid22,dpsid12, kappa,gamma)
    
    return detA

def powerlaw(x12, kappa, gamma,fact,caustics=False):
    
    x1 = x12[0] 
    x2 = x12[1]
    E_r = 1.

    p=fact[3]
    
    dpsi1=x1*E_r**(2 - p)*(x1**2+x2**2)**(p/2 - 2)
    dpsi2=x2*E_r**(2 - p)*(x1**2+x2**2)**(p/2 - 2)
    
    if caustics==True:
        
        x1_per = x1*(kappa+gamma)
        x2_per = x2*(kappa-gamma)
        
        return np.column_stack((x1_per+dpsi1, x2_per+dpsi2))
        
    dpsid11=E_r**(2 - p)*(x1**2+x2**2)**(p/2 - 2)*((p - 1)*x1**2+x2**2)
    dpsid22=E_r**(2 - p)*(x1**2+x2**2)**(p/2 - 2)*((p - 1)*x2**2+x1**2)
    dpsid12=(p - 2)*x1*x2*E_r**(2 - p)*(x1**2+x2**2)**(p/2 - 2)
    
    detA = DistMatrix(dpsid11,dpsid22,dpsid12, kappa,gamma)
    
    return detA


def pointmass(x12,kappa, gamma, fact,caustics=False):
    
    x1 = x12[0] 
    x2 = x12[1] 
    
    #first derivative
    
    dx12dx22 = x1**2.+x2**2
    dpsi1 = x1/dx12dx22
    dpsi2 = x2/dx12dx22
    
    if caustics==True:
        x1_per = x1*(kappa+gamma)
        x2_per = x2*(kappa-gamma)
        
        return  np.column_stack((x1_per+dpsi1, x2_per+dpsi2))
        #return np.column_stack((dpsi1,dpsi2))

    
    #second derivatives 
    
    dx22mdx12 = x2**2.-x1**2.
    dx12pdx22_2 = (x1**2.+x2**2.)**2.
    
    # d^2psi/dx1^2
    dpsid11 = dx22mdx12/dx12pdx22_2
    # d^2psi/dx2^2
    dpsid22 = -dpsid11
    # d^2psi/dx1dx2
    dpsid12 = -2*x1*x2/dx12pdx22_2
    
    detA = DistMatrix(dpsid11,dpsid22,dpsid12, kappa,gamma)
    
    return detA

def softenedpowerlaw(x12,kappa, gamma, fact,caustics=False):
    
    
    a=fact[0]
    b=fact[1]
    c=fact[2]
    p=fact[3]
    
    x1 = x12[0] 
    x2 = x12[1] 
    

    dpsi1=a*p*x1*(b**2+(x1**2+x2**2)/c**2)**(p/2 - 1)/c**2
    dpsi2=a*p*x2*(b**2+(x1**2+x2**2)/c**2)**(p/2 - 1)/c**2
    
    if caustics==True:
        #perturbation contribution
        x1_per = x1*(kappa+gamma)
        x2_per = x2*(kappa-gamma)
        
        return  np.column_stack((x1_per+dpsi1, x2_per+dpsi2))
        
    #second derivatives 
    
    ppart=(b**2+ (x1**2+x2**2)/c**2)    
    dpsid11=a*p*ppart**(p/2 - 1)/c**2 + 2*a*(p/2 - 1)*p*x1**2*ppart**(p/2 - 2)/c**4
    dpsid22=a*p*ppart**(p/2 - 1)/c**2 + 2*a*(p/2 - 1)*p*x2**2*ppart**(p/2 - 2)/c**4
    dpsid12=2*a*ppart**(p/2 - 2)*p*x1*x2*(p/2 - 1)/c**4
    
    detA = DistMatrix(dpsid11,dpsid22,dpsid12, kappa,gamma)
    
    return detA
    

def softenedpowerlawkappa(x12,kappa, gamma, fact,caustics=False):
    
    a=fact[0]
    b=fact[1]
    p=fact[3]
    
    x1 = x12[0] 
    x2 = x12[1] 
    

    x1x22= (x1**2+x2**2)
    ppart=(x1x22+b**2)
    
    if p!=0:
        dpsi1= x1*a**(2-p)*ppart**(p/2 - 1)/x1x22**(1/2) - x1*a**(2-p)*(ppart**(p/2)-b**p)/(p*x1x22**(3/2))      
        dpsi2= x2*a**(2-p)*ppart**(p/2 - 1)/x1x22**(1/2) - x2*a**(2-p)*(ppart**(p/2)-b**p)/(p*x1x22**(3/2))
    
    else:
        dpsi1= (a**2*x1*(2 - (2*b**2)/(b**2 + x1x22) - np.log(1 + x1x22/b**2)))/x1x22**(3/2)
        dpsi2= (a**2*x2*(2 - (2*b**2)/(b**2 + x1x22) - np.log(1 + x1x22/b**2)))/x1x22**(3/2)
        
    if caustics==True:
        #perturbation contribution
        x1_per = x1*(kappa+gamma)
        x2_per = x2*(kappa-gamma)
        
        return  np.column_stack((x1_per+dpsi1, x2_per+dpsi2))
    
    if p!=0:
        dpsid11 = (a**(2 - p)*(b**p*(-2*x1**2+x2**2) + ppart**(-2 + p/2)*(b**4*(2*x1**2-x2**2) + (-1 + p)*x1x22**2*((-2 + p)*x1**2 + x2**2) + b**2*(-((-4 + p)*x1**4) + 2*x1**2 *x2**2 + (-2 + p)*x2**4))))/(p*x1x22**(5/2))     
        dpsid22 = (a**(2 - p)*(b**p*(-2*x2**2+x1**2) + ppart**(-2 + p/2)*(b**4*(2*x2**2-x1**2) + (-1 + p)*x1x22**2*((-2 + p)*x2**2 + x1**2) + b**2*(-((-4 + p)*x2**4) + 2*x2**2 *x2**2 + (-2 + p)*x1**4))))/(p*x1x22**(5/2))   
        dpsid12 = (a**(2 - p)*x1*x2*(-3*b**p + 3*ppart**(p/2) + p*((-2 + p)*x1x22**2*(b**2 + x1x22)**(-2 + p/2) - 2*x1x22*(b**2 +x1x22)**(-1 + p/2))))/(p*x1x22**(5/2))
        
    else:
        dpsid11= a**2*((-2*(b**2*(x1**2 - x2**2) + (3*x1**2 - x2**2)*x1x22))/(x1x22**(3/2)*(b**2 + x1x22)**2) + (( 2*x1**2 - x2**2)*np.log((b**2 + x1x22)/b**2))/x1x22**(5/2))
        dpsid22= a**2*(( 2*(b**2*(x1**2 - x2**2) + (x1**2 - 3*x2**2)*x1x22))/(x1x22**(3/2)*(b**2 + x1x22)**2) + ((-x1**2 + 2*x2**2)*np.log((b**2 + x1x22)/b**2))/x1x22**(5/2))
        dpsid12=(a**2*x1*x2*((-4*x1x22*(b**2 + 2*x1x22))/(b**2 + x1x22)**2 + 3 *np.log(1 + x1x22/b**2)))/x1x22**(5/2)
    
    detA = DistMatrix(dpsid11,dpsid22,dpsid12, kappa,gamma)
    
    return detA
    

def LensEq(x12,kappa, gamma, fact, lens_model):
    
    x1 = x12[0] 
    x2 = x12[1]
    
    alpha=eval(lens_model)((x1,x2),kappa, gamma, fact, caustics=True)
    beta=np.column_stack((x1,x2))-alpha
    
    '''
    print(x[0][:10])
    print(alpha[:10])
    print(beta[:10])
    '''
    
    return beta


def PlotCurves(xS12,xL12,kappa,gamma,lens_model,fact):
    
    xL1 = xL12[0]
    xL2 = xL12[1]
    
    xS1 = xS12[0]
    xS2 = xS12[1]   
    
    xy_lin=np.linspace(-2,2,1000)
    X,Y = np.meshgrid(xy_lin, xy_lin)
    

    crit_curv=eval(lens_model)((X,Y), kappa, gamma, fact)
    
    plt.figure(dpi=100)
    cp = plt.contour(X,Y, crit_curv,[0], colors='k',linestyles= '-', linewidths=0.1) 
    
    
    #I get the coordinates of the contour plot
    xyCrit_all = cp.collections[0].get_paths() 
    
    print(np.shape(xyCrit_all))
    for i in range(np.shape(xyCrit_all)[0]):
        
        #xyCrit = cp.collections[0].get_paths()[i] 
        xyCrit = xyCrit_all[i].vertices  
        xCrit=xyCrit[:,0] 
        yCrit=xyCrit[:,1]
        plt.plot(xCrit,yCrit,'k--',linewidth=0.7)
    
        xyCaus=LensEq((xCrit,yCrit), kappa, gamma, fact, lens_model)
        plt.plot(xyCaus[:,0],xyCaus[:,1],'k-',linewidth=0.7) 
    
    plt.plot(np.nan,np.nan,'k--',label='critical curves',linewidth=0.7)
    plt.plot(np.nan,np.nan,'k-',label='caustics',linewidth=0.7) 
    
    #I always plot the critical curves and caustics with the lens in the middle
    if xL1!=0 and xL2!=0:
        xS1=xL1-xS1
        xS2=xL2-xS2
        xL1=0
        xL2=0
        
    plt.scatter(xL1, xL2, marker='x',color='r', label='lens')
    plt.scatter(xS1, xS2, marker='*',color='orange', label='source')
    
    plt.legend()
    plt.show()
    
    
    
if __name__ == '__main__':
    
    kappa=0
    gamma=0
    lens_model='softenedpowerlawkappa'
    
    xL1=0
    xL2=0
    xL12=[xL1,xL2]
    
    xS1=0.5
    xS2=0.3
    xS12=[xS1,xS2]
    
    a=1
    b=0.1
    c=1
    p=0
    fact= [a,b,c,p]
    
    PlotCurves(xS12,xL12,kappa,gamma,lens_model, fact)
    

    
    
    
   
