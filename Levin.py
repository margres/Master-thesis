import numpy as np
import scipy.special as ss 
from sympy import Symbol,simplify,expand,diff, Function, solve_linear_system_LU, besselj, I,linear_eq_to_matrix,Matrix,exp,N
from string import ascii_lowercase
#import sympy.abc as abc  
import random
from scipy.special import jv as scipy_besselj
import sys
import time
import scipy.linalg as la
from mpmath import fp
# $\int e^{ig(x)}f(x)dx$
# 
# $\int dx\cdot x J(wxy) \cdot e^{iw[0.5x^2-x+\phi]}dx$
# 
# as demonstrated by Levin, we can find an approximation to a less oscillatory solution by collocation.

def diff_eq(F,x,g):
    '''
    input:  F(x), the new not rapidly oscillatory function 
            x, variable in which we need to integrate
            g, exponent of the exponential    
            
    output: differential equation that approximates f(x)
    '''
    return F.diff(x)+I*g.diff(x)*F


def collocation_method(g,f,n_basis):
    '''
    input:  f(x) function in the integral
            g(x), exponent of the exponential 
            n_basis, amount of basis funcitons
            
    output: the function F(x) which has been approximated using the collocation method.          
    
    '''   
    
    x = Symbol('x')
    F = Function('F')(x)

    a_list=[] #list of the constants in the inear combination of n basis funcitons
    u=[] #basis functions

    for i,c in zip(range(1,n_basis+1),ascii_lowercase):
        u.append(x**(i-1)) #monomials
        c = Symbol(str(c))
        a_list.append(c)

    #print(u)
    print(a_list)
    F=np.asarray(a_list).dot(np.asarray(u))
    #print(F)    

    new_eq=diff_eq(F,x,g)-f
    print(new_eq)
    return F,new_eq, a_list

def check_lim(lim):
    '''
    chech the boundaries of the integral
    '''
    if lim == float("inf") or lim == float("-inf"):
        return int(1e6) 
    else:
        return lim


def levin(f,g,lim_inf,lim_sup,n_basis=4):
    '''
    Levin method applied.
    
    '''
    start_time = time.time()
    x = Symbol('x')
    lim_inf=check_lim(lim_inf)
    lim_sup=check_lim(lim_sup)
    
    #I have to add the check if function has only x as a variable

    x_val=random.sample(range(lim_inf, lim_sup), n_basis) #random points in the region of integration
    F,new_eq,variables=collocation_method(g,f,n_basis)
    equations_list=[]
    for i, x_i in enumerate(x_val):
        equations_list.append(simplify(expand(new_eq.subs({x:x_i}))))

    A, b = linear_eq_to_matrix(equations_list, variables)
    elapsed_time = time.time() - start_time
    print('Created set of linear equations after: ',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    #coefficients= list(fp.lu_solve(A, b), variables).args[0])
    A_n=Matrix(np.hstack((A,b)))
 
    
    coefficients_LU= solve_linear_system_LU(A_n,variables)
    elapsed_time = time.time() - start_time
    print('Found the coefficient for the non-rapidly-oscillatory f(x) after:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    coefficients=np.zeros(len(variables),dtype=np.complex128)
    for i,value in enumerate(coefficients_LU.values()):
        coefficients[i]=expand(value)
        #print(sol[i])
        
    #elapsed_time = time.time() - start_time
    F_new=F.copy()
    for coef,coef_val in zip(variables, coefficients):
        F_new=F_new.subs({coef:coef_val})
    evaluate=F_new*exp(I*g)
    solution=N(simplify(evaluate.subs({x:lim_sup})-evaluate.subs({x:lim_inf}))) #integral evaluated at the boundaries
    #print(solution)
    return solution
'''
def linsolv(A,b,variables):
     coefficients= list(linsolve((A, b), variables).args[0])
     return coefficients
 
def LU(A,b,variables):
    P, L, _ = la.lu(A)  
    return P
'''          

if __name__ == "__main__":
    x = Symbol('x')
    J = besselj(0, x)
    #J_lambda = lambdify(x, J, {'besselj': scipy_besselj})
    f=x*J
    g=(0.5*x**2-x)
    print(levin(f,g,0,np.inf,n_basis=4))    







