import numpy as np
from sympy import Symbol,simplify,expand,diff, Function, solve_linear_system_LU, besselj, I,linear_eq_to_matrix,Matrix,exp,N, symarray
from string import ascii_lowercase
#import sympy.abc as abc  
from scipy.special import jv as scipy_besselj
import sys
import time
from mpmath import fp
import pandas as pd

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
        for k_i in range(n_basis):
            #k_=k_i+1
            A[j,k_i]=(order - 1)/x
            Id[j,k_i]=1
            A[j+n_basis,k_i]=const
            A[j,k_i+n_basis]=-const
            A[j+n_basis,k_i+n_basis]=-order/x
            Id[j+n_basis,k_i+n_basis]=1
        k_i=0
    return A, Id

def coeff_list(n_basis):   
    symbols('b_0:10')
    a_list=[Symbol(j) for j in ascii_lowercase[:2*n_basis]]         
    return a_list


def levin_general(f,g,const,a,b,w_oscillating,n_basis=4):
    '''
    Levin method applied.
    The collocation points of the basis need to be equidistant.
    g: exponent of the exponential
    f: x
    '''

    start_time = time.time()
    n=n_basis
    n2=2*n_basis
    k= Symbol('k')
    #j_0=besselj(0, x)
    #j_1=besselj(1, x)
    #print(N(w_oscillating.subs({x:a})))
    
    d=(a+b)/2+0.0000000001
    
    u=(x-d)**(k-1) #points in the range of integration
    uprime=(k-1)*(x-d)**(k-2) #derivative of u

    point=[a+(j-1)*(b-a)/(n-1) for j in range(1,n+1)]
    rhs = np.zeros(n2)  #right hand side, aka f(x)
    
    for i,r in enumerate(point):
        rhs[i]=f.subs({x:r})
    rhs=Matrix(rhs)
    
    #levin's approximation of the bessel function
    A, Id=Bes(const,x,n_basis,point)
    #A_g=J*g.diff()
    A_g=1
    A_f=A*u*A_g+Id*uprime  #here I create the matrix with the contribute of the Bessel function and the exponential
    

    for j in range(n2):
        if j<n:
            val=point[j]
        else:
            val=point[j-n]
        for k_i in range(n2):
            if k_i<n:
                k_=k_i+1
            else:
                k_=k_i+1 -n
           
            A_f[j,k_i]=A_f[j,k_i].subs({x:val,k:k_})

        k_i=0

    if True:       
        c=symarray('c',n2)
        rhs=rhs.reshape(n2,1)
        A_n=Matrix(np.hstack((A_f,rhs)))      
        coefficients_LU=solve_linear_system_LU(A_n,c)
        
        coefficients=np.zeros(len(c),dtype=np.complex128)
        for i,value in enumerate(coefficients_LU.values()):
            coefficients[i]=expand(value)
            
    if False: #I checked which one is the faster
        sol=A_f.LUsolve(rhs)
        coefficients=np.zeros(n2,dtype=np.complex128)
        for i,value in enumerate(sol):
            coefficients[i]=expand(value)
   
    monomials_inf=[u.subs({x:a,k:j}) for j in range(1,n+1)]
    monomials_sup=[u.subs({x:b,k:j}) for j in range(1,n+1)]
  
    result=N(np.dot(monomials_sup,coefficients[:n])*w_oscillating.subs({x:b})-np.dot(monomials_inf,coefficients[:n])*w_oscillating.subs({x:a}))
    
    elapsed_time = time.time() - start_time
    print('Found the coefficient for the non-rapidly-oscillatory f(x) after:', \
          time.strftime("%H:%M:%S", time.gmtime(elapsed_time))+' with '+str(n_basis)+' basis')
    return result,elapsed_time
         

if __name__ == "__main__":
    
    x = Symbol('x')
    f=x
    g=1
    w_SIS=0.1
    bess_func_arg=w_SIS*1
    J = besselj(0, bess_func_arg*x)
    #J_lambda = lambdify(x, J, {'besselj': scipy_besselj})
    #g=(0.5*x**2-x)
    w_oscillating=J#*exp(I*g)
    basis=[]
    result=[]
    elapsed_time=[]
    range_int=[]
    
    for s in range(1,5):
        if s==1:
            pass
        else:
            s=s*2
            
        for i in range(2,20):    
            r,elaps_time=levin_general(f,1,bess_func_arg,0.0000001,s,w_oscillating,n_basis=i*5) 
            result.append(r)
            elapsed_time.append(elaps_time)
            basis.append(i*5)
            range_int.append(s)
        i=0    
        
    df = pd.DataFrame(list(zip(basis,result,elapsed_time,range_int)),columns=['basis','result','time','integration range'] )
    print(df)
    df.to_csv('dataframe_bessfuncarg_'+str(bess_func_arg)+'g'+str(g), sep='\t')
         
      
    
    #print(levin_general(f,g*w,w*1,0.0000001,10,19)) 
   
    #print(levin_general(f,1,bess_func_arg,0.0000001,1,w_oscillating,19)) 






