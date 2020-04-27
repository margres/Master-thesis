import numpy as np
from sympy import chebyshevt,Symbol,simplify,diff, Function, besselj, I,Matrix,exp,N, symarray
import sys
import time
import pandas as pd

# $\int e^{ig(x)}f(x)dx$
# 
# $\int dx\cdot x J(wxy) \cdot e^{iw[0.5x^2-x+\phi]}dx$
# 
# as demonstrated by Levin, we can find an approximation to a less oscillatory solution by collocation.

#*************** GENERAL METHOD  ***************

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


def variable_change(x,a,b,f,g,w_oscillating):
    '''
    change of the integration variable, From a 0-inf integral to a 0-1 
    '''
    #print(type(w_oscillating))
    b_new=1
    a_new=0
    if b==np.inf:
        #b_new=1
        #a_new=0
        x_change=x/(1-x)
        x_prime=1/(1-x)**2
        
    else:
        x_change=a+x*(b-a)
        x_prime=x_change.diff()
        

    f=f.subs({x:x_change})*x_prime
    g=g.subs({x:x_change})
        
    w_oscillating=simplify(w_oscillating.subs({'x':x_change}))
    print(w_oscillating)
    return a_new,b_new,f,g,w_oscillating


def chebyshev_right_open(n):
    
    nodes=[np.cos(((1 - 2*j + 2*n)*np.pi)/(4*n))**2 for j in range(1,n+1)]

    
    return nodes

def right_IMT(x,a,b,f,g,w_oscillating):

    b_new=1
    a_new=0
    
    IMT_x=1-exp(1-1/(1-x))
    IMT_prime=IMT_x.diff()
    

    f=f.subs({'x':IMT_x})*IMT_prime
    g=g.subs({'x':IMT_x})
        
    w_oscillating=w_oscillating.subs({'x':IMT_x})
    #print('WWWWWWW',w_oscillating)

    
    return a_new,b_new,f,g,w_oscillating

def left_IMT(x,a,b,f,g,w_oscillating):

    b_new=1
    a_new=0
    
    
    IMT_x=exp(1-1/(x))
    IMT_prime=IMT_x.diff()
    
    if isinstance(f, Symbol):   
        f=f.subs({x:IMT_x})*IMT_prime
    if isinstance(g, Symbol):
        g=g.subs({x:IMT_x})
        
    w_oscillating=w_oscillating.subs({x:IMT_x})
    
    
    return a_new,b_new,f,g,w_oscillating

    
    
    

def sub_levin_general(f,g,const,a,b,w_oscillating,IMT,n,amount_bisections,direction,dx):
    '''
    Levin method applied.
    The collocation points of the basis need to be equidistant.
    g: exponent of the exponential
    f: x
    const: constants in the bessel function argument
    a,b: limits of the integral
    w_oscillating: osciallting parts of the integral
    n_basis: number of basis
    '''

    start_time = time.time()
    n2=2*n
    k= Symbol('k')
    x=Symbol('x')

    
    point=chebyshev_right_open(n)
     
    #i check the input, to see if g oscillates or not. 
    
    if g==0:
        A_g=1 
    else:
        A_g=I*g.diff() 
    
    #print('before:\n',b,'\n',f,'\n',g,'\n',w_oscillating)
    if b!=np.inf and IMT==False:
        #x_change=x
        a,b,f,g,w_oscillating=variable_change(x,a,b,f,g,w_oscillating) 
    if b==np.inf:
        a,b,f,g,w_oscillating=variable_change(x,a,b,f,g,w_oscillating) 
    if IMT==True:
        a,b,f,g,w_oscillating=right_IMT(x,a,b,f,g,w_oscillating) 
        ##apply only on right side and then end
    

    #everything is in a 0-1 range
    
    middle_point=0
    sub_range=1/2**amount_bisections
    #direction has +1 if goes on the right and -1 on the left
        
    if IMT==False:
        for am in range(1,amount_bisections+1):
            middle_point+=direction[am-1]*(b-a)/2**am 
        middle_point+=a
        if dx==True:
            a= middle_point 
            b= middle_point+sub_range
        elif dx==False:
            a=middle_point-sub_range
            b=middle_point 
        else:
            pass
        print('bisection n. ', amount_bisections)
        print('start ',a,'end ',b,'sub range of ',sub_range, 'middle point', middle_point)
        #x_change,a,b,f,g,w_oscillating=variable_change(x_change,a,b,f,g,w_oscillating)   
         

    
    #d=(a+b)/2+0.0000000001
    #u=(x-d)**(k-1) #points in the range of integration
    #uprime=(k-1)*(x-d)**(k-2) #derivative of u in terms of x
    
    u=chebyshevt(k,x) #idk the range of k for the chebyshev
    uprime=u.diff(x)


    
    rhs = np.zeros(n2)  #right hand side, aka f(x)

    for i,r in enumerate(point):
        rhs[i]=f.subs({x:r})

    
    #levin's approximation of the bessel function
    A, Id=Bes(const,x_change,n,point)
        
        
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
  
          
    c=symarray('c',n2)
    A_f=np.array(A_f,dtype=np.complex128)
    coefficients=np.linalg.solve(A_f,rhs)
    
    
    sub_n=int((n+1)/2)
    sub_coefficients=[coefficients[j] for j in range(0,n2,2)] #every other coefficient

    l_coefficients=[coefficients[j] for j in range(0,n)]
    r_coefficients=[coefficients[j] for j in range(n,n2)]
    
    sub_monomials_start=[u.subs({x:a,k:j}) for j in range(1,sub_n+1)] 
    sub_monomials_stop=[u.subs({x:b,k:j}) for j in range(1,sub_n+1)]
    #print(monomials_start)
    monomials_start=[u.subs({x:a,k:j}) for j in range(1,n+1)]
    monomials_stop=[u.subs({x:b,k:j}) for j in range(1,n+1)]
    
    
  
    if IMT==True:
        w_a=exp(I*(0.015*1 - 0.01)- 1)*besselj(0, 0.01*1/1 - 1)     
        w_b=1
        #print('w_a',w_a)
        #print('w_b',w_oscillating)
    else:
        w_a=w_oscillating.subs({x:a})
        w_b=w_oscillating.subs({x:b})
    
    try:
        sub_b=(np.dot(sub_monomials_stop,sub_coefficients[:sub_n])*w_b)
        sub_a=np.dot(sub_monomials_start,sub_coefficients[:sub_n])*w_a
        sub_result=complex(N(np.dot(sub_monomials_stop,sub_coefficients[:sub_n])*w_b-np.dot(sub_monomials_start,sub_coefficients[:sub_n])*w_a))
        result=complex(N(np.dot(monomials_stop,coefficients[:n])*w_b-np.dot(monomials_start,coefficients[:n])*w_a))
        #r_result=N(np.dot(sub_monomials_stop,r_coefficients[:sub_n])*w_oscillating.subs({x:b})-np.dot(sub_monomials_start,r_coefficients[:sub_n])*w_oscillating.subs({x:a}))
        #l_result=N(np.dot(sub_monomials_stop,l_coefficients[:sub_n])*w_oscillating.subs({x:b})-np.dot(sub_monomials_start,l_coefficients[:sub_n])*w_oscillating.subs({x:a}))

    except:
        result=N(np.dot(monomials_stop,coefficients[:n])*w_b-np.dot(monomials_start,coefficients[:n])*w_a)
        sub_result=N(np.dot(sub_monomials_stop,sub_coefficients[:sub_n])*w_b-np.dot(sub_monomials_start,sub_coefficients[:sub_n])*w_a)
        #r_result=N(np.dot(sub_monomials_stop,r_coefficients[:sub_n])*w_oscillating.subs({x:b})-np.dot(sub_monomials_start,r_coefficients[:sub_n])*w_oscillating.subs({x:a}))
        #l_result=N(np.dot(sub_monomials_stop,l_coefficients[:sub_n])*w_oscillating.subs({x:b})-np.dot(sub_monomials_start,l_coefficients[:sub_n])*w_oscillating.subs({x:a}))
    

        
    
    
    
    elapsed_time = time.time() - start_time
    
    if IMT==True:
        print(w_oscillating)
        print('b',sub_b)
        print('a',sub_a)
    return result,sub_result

def error(a,b):
    return abs(a-b)

def adaptive_subdivision(f,g,const,start,end,w_oscillating,n,amount_bisections,direction):
    #every integral is in 0-1

    IMT=False
    
    if np.sum(direction)==4. and end==np.inf:
        IMT=True
    elif np.sum(direction)==-4.:
        IMT='l'

        
          
    
    result_1,sub_result_1=sub_levin_general(f,g,const,start,end,w_oscillating,IMT,n,amount_bisections,direction,dx=False)
    result_2,sub_result_2=sub_levin_general(f,g,const,start,end,w_oscillating,IMT,n,amount_bisections,direction,dx=True)
    
    result=result_1+result_2
    sub_result=sub_result_1+sub_result_2
    print('result_1',result_1)
    print('result_2',result_2)
    difference=error(result,sub_result)
    
    difference_1=error(result_1,sub_result_1)
    difference_2=error(result_2,sub_result_2)

    
    if difference_1>difference_2 or np.isnan(result_1): 
        #again over left side
        
        print('here - bisecting LEFT')
        res_side_ok=result_2
        #end=start+new_border
        #start=start
        direction[amount_bisections]=-1
        #r=0
    elif difference_1<=difference_2 or np.isnan(result_2):
        #again on right side
        
        print('here - bisecting RIGHT')
        res_side_ok=result_1
        #start=end-new_border
        #end=end
        direction[amount_bisections]=+1
        #l=0
        
    else: 
        sys.exit()
    
        
    return difference, result, res_side_ok, IMT
    
            
def Levin(x,f,g,const,start,end,w_oscillating,n_basis):
    #point=[start+(j-1)*(end-start)/(n_basis-1) for j in range(1,n_basis+1)]
    IMT=False
    
    #i apply levin for the first time.
    direction=np.zeros(50)
    direction[0]=1
    amount_bisections=0
    result,sub_result=sub_levin_general(f,g,const,start,end,w_oscillating,IMT,n_basis,amount_bisections,direction,dx='')
    difference=error(result,sub_result)
    print('error ',difference)
    print('result ',result)
    print('subresult ',sub_result )
    difference=1 
    results_side_ok=[]
    
    #if we see that we need to integrate more we apply again dividing the range in 2 parts.
    #this counts also the nan, it bisects the integration range, we keep bisecting the one with the singularity
    

    while difference>0.0005 or np.isnan(difference)==True:
        amount_bisections+=1
        difference,result, res_side_ok,IMT=adaptive_subdivision(f,g,const,start,end,w_oscillating,n_basis, amount_bisections,direction)
        results_side_ok.append(res_side_ok)  
        if amount_bisections==4. and IMT==True:
            break
          
        print('error ',difference)
      
        print('sum ',sum(results_side_ok))

    return sum(results_side_ok)   


    
if __name__ == "__main__":
    

    #w_oscillating=J*exp(g)
    x = Symbol('x')
    f=x
    y_n=1
    w_SIS=0.01
    bess_func_arg=w_SIS*y_n
    J = besselj(0, bess_func_arg*x)
    g=w_SIS*(0.5*x**2-x)
    #g=x
    w_oscillating=J*exp(I*g)
    #w_oscillating=J*exp(g)

    basis=[]
    result=[]
    elapsed_time=[]
    range_int=[]
    
    '''
    for s in range(1,5):
        if s==1:
            pass
        else:
            s=s*2
           
    for i in range(4,10):    
        print(i)
        r,elaps_time=levin_general(f,g,bess_func_arg,0.0000001,1,w_oscillating,n_basis=i) 
        result.append(r)
        elapsed_time.append(elaps_time)
        basis.append(i*5)
        range_int.append(s)
    i=0    
        
    df = pd.DataFrame(list(zip(basis,result,elapsed_time,range_int)),columns=['basis','result','time','integration range'] )
    #print(df)
    df.to_csv('dataframe_og_func_small.txt', sep='\t')
    '''
    
    #print(levin_general(f,g*w,w*1,0.0000001,10,19)) 
    n_basis=19
    result=Levin(x,f,g,bess_func_arg,0,1000,w_oscillating,n_basis)
    print('basis',n_basis,'result',result)






