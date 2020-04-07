import numpy as np
import scipy.special as ss 
from sympy import Symbol,simplify,expand,diff, Function, solve_linear_system_LU, besselj, I,linear_eq_to_matrix,Matrix,exp,N
from string import ascii_lowercase
#import sympy.abc as abc  
import random
from scipy.special import jv as scipy_besselj
import sys
import time
from mpmath import fp

# $\int e^{ig(x)}f(x)dx$
# 
# $\int dx\cdot x J(wxy) \cdot e^{iw[0.5x^2-x+\phi]}dx$
# 
# as demonstrated by Levin, we can find an approximation to a less oscillatory solution by collocation.

def differential_eq(F,x,g):
    '''
    input:  F(x), the new not rapidly oscillatory function 
            x, variable in which we need to integrate
            g, exponent of the exponential    
            
    output: differential equation that approximates f(x)
    '''
    return F.diff(x)+I*g.diff(x)*F


            
    
    
def collocation_method(f,n_basis):
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

    '''
    for i,c in zip(range(1,n_basis+1),ascii_lowercase):
        u.append(x**(i-1)) #monomials
        c = Symbol(str(c))
        a_list.append(c)
    '''

    a_list=coeff_list(n_basis)
    
    #print(u)
    print(a_list)
    F=np.asarray(a_list).dot(np.asarray(u))
    return F, a_list

def check_lim(lim):
    '''
    chech the boundaries of the integral
    '''
    if lim == float("inf") or lim == float("-inf"):
        return int(1e6) 
    else:
        return lim
    

def levin_basic(f,g,lim_inf,lim_sup,n_basis=4):
    '''
    Levin method applied.
    The collocation points of the basis need to be equidistant.
    '''
    start_time = time.time()
    x = Symbol('x')
    lim_inf=check_lim(lim_inf)
    lim_sup=check_lim(lim_sup)
    
    #I have to add the check if function has only x as a variable
  
    x_val=[lim_inf+(j-1)*(lim_sup-lim_inf)/(n_basis-1) for j in range(1,n_basis+1)]
    '''
    for i in range(n_basis):
        t=i/n_basis
        u = 1-t
        x_val.append(lim_inf*u +  lim_sup*t)
        #print(x_val)
    '''    
        
    F,variables=collocation_method(g,f,n_basis)
    new_eq=differential_eq(F,x,g)-f
    print(new_eq)
    
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


def Bes(const,x,n_basis,point,order=1):
    '''
    function that creates the matrix A for a Bessel function
    '''

    A=Matrix(np.zeros([2*n_basis,2*n_basis]))
    Id=Matrix(np.zeros([2*n_basis,2*n_basis]))
    for j in range(n_basis):
        val=point[j]
        for k_i in range(n_basis):
            k_=k_i+1
            A[j,k_i]=(order - 1)/x
            Id[j,k_i]=1
            A[j+n_basis,k_i]=const
            A[j,k_i+n_basis]=-const
            A[j+n_basis,k_i+n_basis]=-order/x
            Id[j+n_basis,k_i+n_basis]=1
        k_i=0
    return A, Id

def coeff_list(n_basis):   
    a_list=[Symbol(j) for j in ascii_lowercase[:2*n_basis]]         
    return a_list

def compactification():
    x=x/(1-x)

def levin_general(f,g,const,lim_inf,lim_sup,n_basis=4):
    '''
    Levin method applied.
    The collocation points of the basis need to be equidistant.
    '''
    #FIND HOW TO DIVIDE X TO THE REST or mmm probably i don't have to
    start_time = time.time()
    a=lim_inf
    b=lim_sup
    n=n_basis
    n2=2*n_basis
    k= Symbol('k')
    #j_0=besselj(0, x)
    #j_1=besselj(1, x)
    
    d=(a+b)/2+0.0000000001
    u=(x-d)**(k-1)
    uprime=(k-1)*(x-d)**(k-2)

    point=[a+(j-1)*(b-a)/(n-1) for j in range(1,n+1)] #NOT SURE ABOUT N OR 2N
    rhs = np.zeros(n2)  #right hand side, aka f(x)
    
    for i,r in enumerate(point):
        rhs[i]=f.subs({x:r})
    rhs=Matrix(rhs)
    
    #levin's approximation
    A, Id=Bes(const,x,n_basis,point)
    A_f=A*u*I*g.diff()+Id*uprime  #here I create the matrix with the contribute of the Bessel function and the exponential
    
    
    for j in range(n2):
        if j<n:
            val=point[j]
        else:
            val=point[j-n]
        #print(j)
        for k_i in range(n2):
            k_=k_i+1
            A_f[j,k_i]=A_f[j,k_i].subs({x:val,k:k_})
            #print(j,k_i)
            k_i=0
            
    if False:       
        c=coeff_list(n)
        c=np.hstack((c,c))
        rhs=rhs.reshape(n2,1)
        A_n=Matrix(np.hstack((A_f,rhs)))      
        coefficients_LU=solve_linear_system_LU(A_n,c)
        
        coefficients=np.zeros(len(c),dtype=np.complex128)
        for i,value in enumerate(coefficients_LU.values()):
            coefficients[i]=expand(value)
            
    if True:
        sol=A_f.LUsolve(rhs)
        coefficients=np.zeros(n2,dtype=np.complex128)
        for i,value in enumerate(sol):
            coefficients[i]=expand(value)
   
    approx_inf=[u.subs({x:1,k:j}) for j in range(n)]
    approx_sup=[u.subs({x:2,k:j}) for j in range(n)]
  
    result=np.dot(approx_sup,coefficients[:n])-np.dot(approx_inf,coefficients[:n])
    
    elapsed_time = time.time() - start_time
    print('Found the coefficient for the non-rapidly-oscillatory f(x) after:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    return result
         

if __name__ == "__main__":
    
    x = Symbol('x')
    J = besselj(0, x)
    #J_lambda = lambdify(x, J, {'besselj': scipy_besselj})
    f=x
    g=(0.5*x**2-x)
    w=J*exp(I*g)

        
    print(levin_general(f,g,1,0.5,1,n_basis=7))    







