# -*- coding: utf-8 -*-
# @Author: lshuns
# @Date:   2020-08-03 16:53:12
# @Last Modified by:   lshuns
# @Last Modified time: 2020-08-04 15:53:53

### solve the lens equation

######### Coordinates convention:
############# the origin is set in the centre of perturbation (external shear) and the x-aixs is parallel to the direction of shear, that means gamma2 = 0
############# the source is set in the origin (xs=0)

######### Caveat:
############# images with theta ~ [0,0.5*np.pi,np.pi,1.5*np.pi,2.*np.pi] is ignored

import numpy as np  
import scipy.optimize as op


def TFunc(x12, xL12, lens_model, kappa=0, gamma=0):
    """
    the time-delay function (Fermat potential)

    Parameters
    ----------

    x12: a list of 1-d numpy arrays [x1, x2]
        Light impact position, coordinates in the lens plane.
    xL12: a list of 1-d numpy arrays [xL1, xL2]
        lens center position, coordinates in the lens plane.
    lens_model: str
        Lens model, supported model ('point').
    kappa: float (optional, default=0)
        convergence of external shear.
    gamma: float (optional, default=0)
        shear of external shear.
    """

    # geometrical term (including external shear)
    x1 = x12[0]
    x2 = x12[1]
    tau = 0.5*(x1**2.*(1-kappa-gamma) + x2**2.*(1-kappa+gamma))

    # distance between light impact position and lens position
    dx1 = np.absolute(x1-xL12[0])
    dx2 = np.absolute(x2-xL12[1])

    # deflection potential
    if lens_model == 'point':
        tau -= np.log(np.sqrt(dx1**2.+dx2**2))
    
    return tau


def dTFunc(x12, xL12, lens_model, kappa=0, gamma=0):
    """
    the first derivative of time-delay function (Fermat potential)

    Parameters
    ----------

    x12: a list of 1-d numpy arrays [x1, x2]
        Light impact position, coordinates in the lens plane.
    xL12: a list of 1-d numpy arrays [xL1, xL2]
        lens center position, coordinates in the lens plane.
    lens_model: str
        Lens model, supported model ('point').
    kappa: float (optional, default=0)
        convergence of external shear.
    gamma: float (optional, default=0)
        shear of external shear.
    """

    # geometrical term (including external shear)
    x1 = x12[0]
    x2 = x12[1]
    dtaudx1 = (1-kappa-gamma)*x1
    dtaudx2 = (1-kappa+gamma)*x2

    # distance between light impact position and lens position
    dx1 = np.absolute(x1-xL12[0])
    dx2 = np.absolute(x2-xL12[1])

    # deflection potential
    if lens_model == 'point':
        dx12dx22 = dx1**2.+dx2**2
        dtaudx1 -= dx1/dx12dx22
        dtaudx2 -= dx2/dx12dx22

    return dtaudx1, dtaudx2


def ThetaOrRFunc(theta_t, xL12, lens_model, kappa=0, gamma=0, thetaORr='theta'):
    """
    the theta part or the r part of the lens equation
        Note: dx1 = r_t*cosTheta_t, dx2 = r_t*sinTheta_t

    Parameters
    ----------

    theta_t: 1-d numpy arrays
        Angular coordinate of dx(=xI-xL) in lens plane.
    xL12: a list of 1-d numpy arrays [xL1, xL2]
        Lens center position, coordinates in the lens plane.
    lens_model: str
        Lens model, supported model ('point').
    kappa: float (optional, default=0)
        Convergence of external shear.
    gamma: float (optional, default=0)
        Shear of external shear.
    thetaORr: str (optional, default='theta') 
        Return the theta ('theta') or r ('r') part of the lens equation.
    """

    # lens position
    xL1 = xL12[0]
    xL2 = xL12[1]

    if lens_model=='point':
        # external-shear-related constants
        A1 = (1.-kappa-gamma)
        A2 = (1.-kappa+gamma)
        # theta 
        cosTheta = np.cos(theta_t)
        sinTheta = np.sin(theta_t)
        #
        C = xL1*A1*sinTheta - xL2*A2*cosTheta
        rt = C/(A2-A1)/cosTheta/sinTheta
        #
        if thetaORr == 'theta':
            return A1*xL1 + A1*rt*cosTheta - cosTheta/rt
            # return A2*xL2 + A2*rt*sinTheta - sinTheta/rt
        elif thetaORr == 'r':
            return rt
        else:
            raise Exception('Unsupported thetaORr value! using either r or theta!')


def muFunc(x12, xL12, lens_model, kappa=0, gamma=0):
    """
    the magnification factor

    Parameters
    ----------

    x12: a list of 1-d numpy arrays [x1, x2]
        Light impact position, coordinates in the lens plane.
    xL12: a list of 1-d numpy arrays [xL1, xL2]
        lens center position, coordinates in the lens plane.
    lens_model: str
        Lens model, supported model ('point').
    kappa: float (optional, default=0)
        convergence of external shear.
    gamma: float (optional, default=0)
        shear of external shear.
    """

    # distance between light impact position and lens position
    dx1 = np.absolute(x12[0]-xL12[0])
    dx2 = np.absolute(x12[1]-xL12[1])    

    # second order derivative of deflection potential
    if lens_model == 'point':
        dx22mdx12 = dx2**2.-dx1**2.
        dx12pdx22_2 = (dx1**2.+dx2**2.)**2.
        # d^2psi/dx1^2
        dpsid11 = dx22mdx12/dx12pdx22_2
        # d^2psi/dx2^2
        dpsid22 = -dpsid11
        # d^2psi/dx1dx2
        dpsid12 = -2*dx1*dx2/dx12pdx22_2

    # Jacobian matrix
    j11 = 1. - kappa - gamma - dpsid11
    j22 = 1. - kappa + gamma - dpsid22
    j12 = -dpsid12
    # magnification
    mu = 1./(j11*j22-j12*j12)

    return mu


def Images(xL12, lens_model, kappa=0, gamma=0, return_mu=False, return_T=False):
    """
    Solving the lens equation

    Parameters
    ----------
    xL12: a list of 1-d numpy arrays [xL1, xL2]
        lens center position, coordinates in the lens plane.
    lens_model: str
        Lens model, supported model ('point').
    kappa: float (optional, default=0)
        convergence of external shear.
    gamma: float (optional, default=0)
        shear of external shear.
    return_mu: bool, (optional, default=False)
        Return the maginification (or not).
    return_T: bool, (optional, default=False)
        Return the time delay (or not).
    """

    # +++++++++++++ solve the lens equation

    # solve the theta-function
    N_theta_t = 100
    d_theta_t = 1e-3

    node_theta_t = np.array([0, 0.5*np.pi, np.pi, 1.5*np.pi, 2.*np.pi])

    theta_t_res = []

    for i in range(len(node_theta_t)-1):

        theta_t = np.linspace(node_theta_t[i]+d_theta_t, node_theta_t[i+1]-d_theta_t, N_theta_t)
        theta_t_f = ThetaOrRFunc(theta_t, xL12, lens_model, kappa, gamma, 'theta')

        # those with root
        flag_root = (theta_t_f[1:] * theta_t_f[:-1]) <=0
        theta_t_min = theta_t[:-1][flag_root]
        theta_t_max = theta_t[1:][flag_root]
        for j in range(len(theta_t_min)):
            tmp = op.brentq(ThetaOrRFunc, theta_t_min[j],theta_t_max[j], args=(xL12, lens_model, kappa, gamma, 'theta'))
            theta_t_res.append(tmp)

    # corresponding r_t
    theta_t_res = np.array(theta_t_res)
    r_t = ThetaOrRFunc(theta_t_res, xL12, lens_model, kappa, gamma, 'r')
    # true solutions
    true_flag = r_t>1e-5
    theta_t_res = theta_t_res[true_flag]
    r_t = r_t[true_flag]
    nimages = len(theta_t_res)
    # to x, y
    dx1 = r_t*np.cos(theta_t_res)
    dx2 = r_t*np.sin(theta_t_res)
    xI12 = [dx1 + xL12[0], dx2 + xL12[1]]

    # +++++++++++++ magnification 
    if return_mu:
        mag = muFunc(xI12, xL12, lens_model, kappa, gamma)
    else:
        mag = None

    # +++++++++++++ time delay
    if return_T:
        tau = TFunc(xI12, xL12, lens_model, kappa, gamma)
    else:
        tau = None

    return (nimages, xI12, mag, tau)


def saddleORmin(muI, tauI, nimages):
    
    saddle_index=[]
    minimum_index=[]
    
    crit_matrix=np.zeros((int(nimages),3),dtype='U4')
    
    for i in range(nimages):  
        
        crit_matrix[i,0]=i
        
        crit_matrix[i,1]=tauI[i]  
        
        if muI[i]<0:
            crit_matrix[i,2]='s'
        else:
            crit_matrix[i,2]='m'
    
    crit_matrix=crit_matrix[crit_matrix[:,1].argsort()]
    
    #np.save('crit_matrix.npy',crit_matrix)
    
    print(crit_matrix)
    
    for i in range(int(nimages/2)):
        minimum_index.append(int(crit_matrix[np.where(crit_matrix[:,2]=='m')[0][i]][0]))
        saddle_index.append(int(crit_matrix[np.where(crit_matrix[:,2]=='s')[0][i]][0]))
                  
    return (saddle_index, minimum_index)
    
            

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # lens
    lens_model = 'point'
    xL1 = 0.1
    xL2 = 0.1

    # external shear
    kappa = 0
    gamma = 0.2

    n_steps=800
    n_bins=800
    xmin=-1.5
    xmax=1.5

    x_range=xmax-xmin
    x_lin=np.linspace(xmin,xmax,n_steps)
    y_lin=np.linspace(xmin,xmax,n_steps)

    X,Y = np.meshgrid(x_lin, y_lin) # grid of point
    
    tau = TFunc([X,Y], [xL1, xL2], lens_model, kappa, gamma)
    
    nimages, xI12, muI, tauI = Images([xL1, xL2], lens_model, kappa, gamma, return_mu=True, return_T=True) 
    print('number of images', nimages)
    print('positions', xI12)
    print('magnification', muI)
    print('time delay', tauI)

    sorted_critical = saddleORmin(muI, tauI, nimages)
    print(sorted_critical)
    
    #contour plot
    fig = plt.figure(dpi=100)
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 
    # image
    plt.scatter(xI12[0], xI12[1], color='r', s=4)
    # contour
    cp = ax.contour(X, Y, tau, np.linspace(0,1,50), linewidths=0.6, extent=[-2,2,-2,2], colors='black')
    plt.gca().set_aspect('equal', adjustable='box')
    cp.ax.set_ylabel('y', fontsize=13)
    cp.ax.set_xlabel('x', fontsize=13)
    
    plt.title('Contour plot of time delay', fontsize=13)
    # plt.savefig('./test/contour_plot_with_points.png')
    plt.show()