
# -*- coding: utf-8 -*-
# @Author: lshuns & mrgr
# @Date:   2020-07-17 15:49:19
# @Last Modified by:   lshuns
# @Last Modified time: 2020-08-30 12:45:03

### solve the diffraction integral in virture of Fourier transform & histogram counting
###### reference: Nakamura 1999; Ulmer 1995

######### Coordinates convention:
############# the origin is set in the centre of perturbation (external shear) and the x-aixs is parallel to the direction of shear, that means gamma2 = 0
############# the source is set in the origin (xs=0)


import numpy as np
import pandas as pd

import os
import sys

# Self-defined package
sys.path.insert(0,os.path.realpath('..')) 
from Images import TFunc, dTFunc, Images
from Fouriertrans import F_d,FT_clas
from scipy.interpolate import UnivariateSpline


class TreeClass(object):
    """
    A tree class saving node information.
    Parameters
    ----------
    dt_require: float
        Sampling step in time delay function.
    tlim: float
        Sampling limit in time delay function.    
    """

    def __init__(self, dt_require, tlim):

        # general information
        self.dt_require = dt_require
        self.tlim = tlim

        # initial empty good nodes
        self.good_nodes = pd.DataFrame({'x1': [],
                                        'x2': [],
                                        'tau': [],
                                        'weights': []
                                        })

    def SplitFunc(self, weights, x1_list, x2_list, T_list, dT_list):
        """
        Split nodes into bad or good based on the errors of tau values.
        """

        # ++++++++++++ calculate error based on gradient
        error_list = dT_list*(weights**0.5)
        flag_error = (error_list < self.dt_require)

        # +++++++++++++ tlim
        # check the min T in a range (avoid removing potential good nodes)
        T_lim_list = T_list - error_list
        flag_tlim = (T_list < self.tlim)

        # +++++++++++++ total flag
        flag_bad = flag_tlim & np.invert(flag_error)
        flag_good = flag_tlim & flag_error

        # ++++++++++++++ good node information saved to DataFrame
        tmp = pd.DataFrame(data={
            'x1': x1_list[flag_good],
            'x2': x2_list[flag_good],
            'tau': T_list[flag_good],
            'weights': weights*np.ones_like(x1_list[flag_good])
            })
        self.good_nodes = self.good_nodes.append(tmp, ignore_index=True)

        # ++++++++++++++ bad node using simple directory
        self.bad_nodes = [x1_list[flag_bad], x2_list[flag_bad], T_list[flag_bad], dT_list[flag_bad]]


def FtSingularFunc(images_info, tau_list):
    """
    Calculate the singular part of F(t)
    Parameters
    ----------
    images_info: DataFrame
        All images' information.
    tau_list: numpy array
        Sampling of tau where Ftc being calculated.
    """

    # shift tau
    images_info['tauI'] -= np.amin(images_info['tauI'].values)

    Ftc = np.zeros_like(tau_list)
    for index, row in images_info.iterrows():
        tmp = np.zeros_like(tau_list)
        # min 
        if row['typeI'] == 'min':
            tmp[tau_list>=row['tauI']] = 2.*np.pi*(row['muI']**0.5)
            # print(">>>> a min image")
        # max 
        elif row['typeI'] == 'max':
            tmp[tau_list>=row['tauI']] = -2.*np.pi*(row['muI']**0.5)
            # print(">>>> a max image")
        # saddle 
        elif row['typeI'] == 'saddle':
            tmp = -2.*((-row['muI'])**0.5)*np.log(np.absolute(tau_list-row['tauI']))
            # print(">>>> a saddle image")
        else:
            raise Exception("Unsupported image type {:} !".format(row['typeI']))

        Ftc += tmp

    return Ftc

def FtHistFunc(xL12, lens_model, kappa=0, gamma=0, tlim=6., dt=1e-2):
    """
    Calculate F(t) with histogram counting
    Parameters
    ----------
    xL12: a list of 1-d numpy arrays [xL1, xL2]
        lens center position, coordinates in the lens plane.
    lens_model: str
        Lens model, supported model ('point').
    kappa: float (optional, default=0)
        Convergence of external shear.
    gamma: float (optional, default=0)
        Shear of external shear.
    tlim: float (optional, default=10.)
        Sampling limit in time delay function (tmax = tI_max+tlim).
    dt: float (optional, default=1e-2)
        Sampling step in time delay function.
    """

    # calculate the images
    nimages, xI12, muI, tauI, typeI = Images(xL12, lens_model, kappa, gamma, return_mu=True, return_T=True) 
    # collect image info
    images_info = pd.DataFrame(data=
                    {
                    'xI1': xI12[0],
                    'xI2': xI12[1],
                    'muI': muI,
                    'tauI': tauI,
                    'typeI': typeI
                    })

    # tau bounds from images
    tImin = np.amin(images_info['tauI'].values)
    tImax = np.amax(images_info['tauI'].values)

    # tlim is set on the top of tImax
    tlim += tImax

    # initial guess of bounds
    ## x1
    xI1_min = np.amin(images_info['xI1'].values)
    xI1_max = np.amax(images_info['xI1'].values)
    dxI1 = xI1_max - xI1_min
    boundI_x1 = [xI1_min-dxI1, xI1_max+dxI1]
    ## x2
    xI2_min = np.amin(images_info['xI2'].values)
    xI2_max = np.amax(images_info['xI2'].values)
    dxI2 = xI2_max - xI2_min
    boundI_x2 = [xI2_min-dxI2, xI2_max+dxI2]

    # +++ extend bounds until meeting tlim
    while True:
        N_bounds = 1000
        # build the bounds
        tmp2 = np.linspace(boundI_x1[0], boundI_x1[1], N_bounds)
        x1_test = np.concatenate([
                        tmp2,                            # top
                        np.full(N_bounds, boundI_x1[1]), # right
                        tmp2,                            # bottom
                        np.full(N_bounds, boundI_x1[0])  # left 
                        ])

        tmp2 = np.linspace(boundI_x2[0], boundI_x2[1], N_bounds)
        x2_test = np.concatenate([
                        np.full(N_bounds, boundI_x2[1]), # top
                        tmp2,                            # right
                        np.full(N_bounds, boundI_x2[0]), # bottom
                        tmp2                             # left 
                        ])

        # evaluate tau 
        T_tmp = TFunc([x1_test, x2_test], xL12, lens_model, kappa, gamma)

        # break condition
        if np.amin(T_tmp) > tlim:
            break

        # extend bounds
        boundI_x1 = [boundI_x1[0]-0.5*dxI1, boundI_x1[1]+0.5*dxI1]
        boundI_x2 = [boundI_x2[0]-0.5*dxI2, boundI_x2[1]+0.5*dxI2]

    # +++ hist counting
    # initial steps
    N_x = 5000

    # initial nodes
    x1_node = np.linspace(boundI_x1[0], boundI_x1[1], N_x)
    x2_node = np.linspace(boundI_x2[0], boundI_x1[1], N_x)
    dx1 = x1_node[1]-x1_node[0]
    dx2 = x2_node[1]-x2_node[0]
    x1_grid, x2_grid = np.meshgrid(x1_node, x2_node)     
    x1_list = x1_grid.flatten()
    x2_list = x2_grid.flatten()
    T_list = TFunc([x1_list, x2_list], xL12, lens_model, kappa, gamma)
    # gradient for error calculation
    dtaudx1, dtaudx2 = dTFunc([x1_list, x2_list], xL12, lens_model, kappa, gamma)
    dT_list = np.sqrt(np.square(dtaudx1)+np.square(dtaudx2))

    # build Tree
    Tree = TreeClass(dt, tlim)
    Tree.SplitFunc(dx1*dx2, x1_list, x2_list, T_list, dT_list)

    # iterate until bad_nodes is empty
    idx = 0
    while len(Tree.bad_nodes[0]):
        idx +=1
        print('loop', idx)

        # bad nodes
        N_bad = len(Tree.bad_nodes[0])
        print('number of bad_nodes', N_bad)

        # +++ subdivide bad nodes' region
        time1 = time.time()
        # each bad node being subdivided to 4 small ones
        x1_bad = np.repeat(Tree.bad_nodes[0], 4)
        x2_bad = np.repeat(Tree.bad_nodes[1], 4)
        # new nodes
        x1_list = x1_bad + np.tile([-0.25*dx1, 0.25*dx1, -0.25*dx1, 0.25*dx1], N_bad)
        x2_list = x2_bad + np.tile([0.25*dx1, 0.25*dx1, -0.25*dx1, -0.25*dx1], N_bad)
        ##
        time2 = time.time()
        # print('list built finished in', time2-time1)

        # +++ calculate tau & dtau
        T_list = TFunc([x1_list, x2_list], xL12, lens_model, kappa, gamma)
        # gradient for error calculation
        dtaudx1, dtaudx2 = dTFunc([x1_list, x2_list], xL12, lens_model, kappa, gamma)
        dT_list = np.sqrt(np.square(dtaudx1)+np.square(dtaudx2))
        ##
        time1 = time.time()
        # print('T_grid and dT_grids finished in', time1-time2)

        # +++ split good and bad
        dx1 *= 0.5
        dx2 *= 0.5
        Tree.SplitFunc(dx1*dx2, x1_list, x2_list, T_list, dT_list)
        # print('Split finished in', time.time()-time1)

    # re-set origin of tau
    Tree.good_nodes['tau'] -= tImin

    # hist counting
    N_bins = int((tlim-tImin)/dt)
    Ft_list, bin_edges = np.histogram(Tree.good_nodes['tau'].values, N_bins, weights=Tree.good_nodes['weights'].values/dt)
    tau_list = (bin_edges[1:]+bin_edges[:-1])/2.
    plt.plot(tau_list,Ft_list)
    
    # avoid edge effects
    tau_list = tau_list[:-10]
    Ft_list = Ft_list[:-10]
    
    '''
    tau_extension,Ft_extension, n_points=fit_Func(tau_list,Ft_list,'ft')
    plt.plot(tau_extension,Ft_extension)
    plt.xlim(0,2)
    plt.show()
    
    index_extension=np.where(tau_extension>np.max(tau_list))[0][0]
    tau_list_extended=np.append(tau_list,tau_extension[index_extension:])
    Ft_list_extended=np.append(Ft_list,Ft_extension[index_extension:])
    plt.plot(tau_list_extended, Ft_list_extended)
    plt.xlim(0,2)
    plt.show()
    '''   
    # calculate signular part
    Ftc = FtSingularFunc(images_info, tau_list)

    # remove signular part
    Ftd = Ft_list - Ftc

    return tau_list, Ftd, Ft_list, muI,tauI

def fit_Func(a,b,funct):
    
    '''
    fitting of the smoothed curve
    '''
    
    if funct=='ftd': 
        fitting_order=50
        begin_fit=0
        n_sample=len(a)
        xs = np.linspace(np.min(a), np.max(a), n_sample)
        
    elif funct=='ftc': 
        fitting_order=1
        n_sample=100
        begin_fit=np.where(a>np.max(a)-0.2)[0][0] 
        xs = np.linspace(a[begin_fit],np.max(a)+0.1, n_sample)
        
    z = np.polyfit(a[begin_fit:], b[begin_fit:], fitting_order)
    p = np.poly1d(z)
    ys=p(xs)
    
    #index_y_zero=np.where(predicted<=0)[0][0]
    
    return xs,ys, n_sample


def magnification(F):
    return np.abs(F)**2

def extend_Fc(xs,ys):
    
    xs_extension,ys_extension, n_points=fit_Func(xs,ys,'ftc')
    index_extension=np.where(xs_extension>np.max(xs))[0][0]
    xs_extended=np.append(xs,xs_extension[index_extension:])
    ys_extended=np.append(ys,ys_extension[index_extension:])

    
    return xs_extended,ys_extended


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import time 
    
    
    

    
    # lens
    lens_model = 'point'
    xL1 = 0.1
    xL2 = 0.1

    # external shear
    kappa = 0
    gamma = 0.2

    # accuracy
    tlim = 0.5
    dt = 1e-3
    
 
    print('start running...')
    start = time.time()
    tau_list, Ftd, Ft_list, muI, tauI = FtHistFunc([xL1, xL2], lens_model, kappa, gamma, tlim, dt)
    # print(good_nodes)
    
    '''
    np.save('muI.npy', muI)
    np.save('tau_list.npy', tau_list)
    np.save('tauI.npy', tauI)
    np.save('Ftd.npy', Ftd)
    np.save('Ft_list.npy', Ft_list)
    print('finished in', time.time()-start)
    
    
    muI = np.load('muI.npy')
    tauI = np.load('tauI.npy')
    tau_list= np.load('tau_list.npy')
    Ftd = np.load('Ftd.npy')
    Ft_list= np.load('Ft_list.npy')
    '''
    
    outfile = './plot/test_Ft.png'
    plt.plot(tau_list, Ft_list)
    plt.xlabel('time')
    plt.ylabel('F tilde')
    plt.savefig(outfile, dpi=300)
    plt.close()
    #print('Plot saved to', outfile)

    outfile = './plot/test_Ftd.png'
    xs, ys,n_sample=fit_Func(tau_list,Ftd,'ftd')

    np.save('xs.npy', xs)
    np.save('ys.npy', ys)
    
    plt.plot(tau_list, Ftd, color='orange')
    plt.plot(xs, ys, color='green')
    plt.xlabel('time')
    plt.ylabel('F_d  tilde')
    plt.title(str(n_sample)+' sample points')
    plt.savefig(outfile, dpi=300)
    #plt.close()
    plt.show()
    print('Plot saved to', outfile)
    
    xs_extended,ys_extended=extend_Fc(xs,ys)
    plt.plot(xs_extended,ys_extended)
    plt.savefig('./Fc_extension_2.png',dpi=300)
    plt.show()
    
    #uncomment this if you want to extend at higher times    
    #xs= xs_extended
    #ys= ys_extended
    
    omega,F_diff=F_d(xs[10:],ys[10:])
  
    F_clas=np.zeros((4,len(omega)), dtype="complex_")

    
    for i,(m,t) in enumerate(zip(muI,tauI)):
        #print(m,t)
        F_clas[i,:]=FT_clas(omega,t,m, xs, ys)
        #print(F_crit[i,:])
    F_clas=np.sum(F_clas,axis=0)
    
    
    pos_indices=np.where(omega<0)[0][1]-1
    plt.plot(omega[:pos_indices], magnification(F_diff[:pos_indices]), label='F diffraction' )
    #plt.show()
    plt.plot(omega[:pos_indices], magnification(F_clas[:pos_indices]), label='F semi-classical')
    plt.plot(omega[:pos_indices], magnification(F_clas[:pos_indices]+F_diff[:pos_indices]), label='F full')
    plt.xlabel('frequency')
    plt.ylabel('|F|^2 amplification factor')
    #plt.title(str(n_sample)+' sample points')
    plt.legend()
    #plt.xlim(0,150)
    plt.savefig('./magnification_factor_extension_1.png',dpi=300)
    plt.show()

    