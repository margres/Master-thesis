#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:39:31 2020

@author: lshuns

diffraction integral for point-mass lens model
"""

import numpy as np 
import scipy.special as ss
from scipy.special import jv 
import scipy.integrate as integrate
from Levin import *
from sympy import *
from scipy.special import jv as scipy_besselj

def PointMassFunc(w, y):
    """
    Main function for solving diffraction integral with point-mass lens model
    
    Parameters
    ----------
    w : 1-d array
        The dimensionless frequency.
    
    y : float
        The impact parameter.
    """

    xm = (y + (y*y + 4.0)**0.5)/2.0
    phim = (xm - y)**2.0/2.0 - np.log(xm)

    Fw = np.exp(np.pi*w/4.0 + w*1j/2.0*(np.log(w/2.0)-2.0*phim)) \
        * ss.gamma(1-1j/2.0*w) \
        * KummerComplexFunc(1j/2.0*w, 1.0, 1j/2.0*w*y*y)

    return Fw

#Confluent hypergeometric function 1F1
def KummerComplexFunc(a, b, x):
    """
    Confluent hypergeometric function 1F1
        summing the generalized hypergeometric series:
            Sum(n=0-->Inf)[(a)_n*x^n/{(b)_n*n!}]

    Parameters
    ----------
    a : 1-d array / float
    b : 1-d array / float
    x : 1-d array / float
    """

    # check the size of input    
    a_size = np.ma.size(np.array(a))
    b_size = np.ma.size(np.array(b))
    x_size = np.ma.size(np.array(x))
    size_list = [a_size, b_size, x_size]
    size_list.sort()
    size_max = size_list[-1]

    # at least one input is an array
    if size_max != 1:
        # the input should have either same size or null size (float)
        if (size_list[-2]!=1) and (size_list[-2]!=size_max):
            raise Exception(f"Inconsistent input parameters! \
                Input should be either in same size or be float.\
                Current input size for a={a_size}, b={b_size}, x={x_size}")
        elif (size_list[-3]!=1) and (size_list[-3]!=size_max):
            raise Exception(f"Inconsistent input parameters! \
                Input should be either in same size or be float.\
                Current input size for a={a_size}, b={b_size}, x={x_size}")
        # fill input into same size
        else:
            if (a_size != size_max):    
                a = np.ones(size_max)*a

            if (b_size != size_max):
                b = np.ones(size_max)*b

            if (x_size != size_max):
                x = np.ones(size_max)*x

    # Default tolerance
    # Summing until the specified tolerance is acheived.
    tol = 1e-10

    # maximum number of series 
    # hope never reached
    nmax = 100000

    # start summing
    term = x*a/b
    f = 1.0 + term
    n = 1
    while (n<nmax and np.max(np.absolute(term))>tol):
        n = n + 1
        a = a + 1.0
        b = b + 1.0
        term = x * term * a / b / float(n)
        f = f + term

    if (np.max(np.absolute(term)) > tol):
        print(f"Warning: KummerComplex has n > {nmax} \
            with error {term}")
        
    return f

def SIS(Mlz,w,y):
    #i=1j
    phim=y+0.5
    solv_int=np.zeros(len(w), dtype=np.complex128)
    F=np.zeros(len(w), dtype=np.complex128)
    x = Symbol('x')
    
    for j, wi in enumerate(w):
        solv_int[j]=levin(x*besselj(0,x*wi*y),wi*(0.5*x**2-x+phim),0,np.inf,19)
        F[j]=-I*wi*exp((I*wi*y**2)/2)*solv_int[j]
        
    return F

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
    #w = np.loadtxt('w.txt',delimiter='\n')
    w=[0.1,1]
    #w = 8*np.pi*Mlz*f
    

    #Fw_list = PointMassFunc(w, y)
    #print(Fw_list)
    SIS_list=SIS(Mlz,w,y)
    print(SIS_list)
    
    #np.savetxt('try_10_abs.txt', SIS_list.real , delimiter=' ', fmt='%d')
    #np.savetxt('w_try.txt', abs(SIS_list) , delimiter=' ', fmt='%d')


