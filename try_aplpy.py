#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 19:04:21 2020

@author: mrgr
"""
import aplpy
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy import constants as const
import numpy as np
import glob
import os
import sys
import matplotlib.pyplot as plt
from astropy.utils.data import get_pkg_data_filename
from math import log
import aplpy
from astropy.io import fits
from astropy.io.fits import getheader
from astropy.io.fits import Header
from astropy.wcs import WCS
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
np.warnings.filterwarnings('ignore')
from scipy.optimize import curve_fit
from astropy.coordinates import Distance
from astropy.cosmology import WMAP9 as cosmo


rms_jvla=6.5e-5
rms_gmrt=7.52e-4 
rms_lofar=8.6e-4

v_gmrt=325 #mhz
v_jvla=1500 #mhz
v_lofar=150 #mhza

x_b=2.5333e-5*10**6
x_c=1.8857e-5*10**6
x_d=3.72577e-5*10**6
x_all=rms_jvla
x_g=rms_gmrt*10**6
x_l=rms_lofar*10**6

#lev_jvla=np.dot([3, 4, 5, 6, 16, 32, 64,128],x_all)
lev=[1,2,4,8,16,32,64]
lev_jvla=np.dot(lev,3*x_all)
lev_b=np.dot(lev,3*x_b)
lev_c=np.dot(lev,3*x_c)
lev_d=np.dot(lev,3*x_d)
lev_gmrt=np.dot(lev,3*x_g)
lev_lofar=np.dot(lev,3*x_l)
lev_lofar_highres=np.dot(lev,3*378)
lev_gmrt_highres=np.dot(lev,3*250)
lev_wsclean=np.dot(lev,3*16)

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
z=0.027
Ez = cosmo.H(z).value/cosmo.H0.value
denscrit = (cosmo.critical_density(z))
dens_500=denscrit*500
M_500=0.735*10**(14)*const.M_sun*10**3*u.g /u.kg #from kg to g
R_500=((3*M_500)/(4*np.pi*dens_500))**(1/3)
R_500=(0.631*u.Mpc).to(u.kpc)
print('R_500',R_500)
D=Distance(z=0.027) #distance from z  #122.67 mpc from ned
print('cluster distance',D)
r_com=cosmo.comoving_distance(0.027)  
print('comoving distance',r_com)
d_A = cosmo.angular_diameter_distance(z=0.027)
theta = R_500/d_A

scale=cosmo.kpc_proper_per_arcmin(z)/60/u.arcsec*u.arcmin
Mpc=(1000./scale.value)/3600
kpc_100=(100./scale.value)/3600
R_500_deg=R_500.value/scale.value/3600


fig = plt.figure(figsize=(6, 6))
#fig1 = aplpy.FITSFigure(path_stokesQU + "Pfrac_5arcsec_ExtDepol1.5GHz.fits", slices=[0,0], figure=fig, subplot=(1,1,1))
D = aplpy.FITSFigure('./FITS/new_D_array_image.fits', slices=[0,0],figure=fig,subplot=(1,1,1))
D.set_title('D configuration',size=18)
D.show_contour('./FITS/new_D_array_image.fits',colors='black', \
               smooth=1, levels=lev_d, alpha=0.2)

#D.show_contour('./FITS/MKW8_uvtaper_0_weight_briggs_robust_0._smooth_regrid.image.fits',colors='black', \
#               slices=[0,0],smooth=3, levels=lev_jvla, alpha=0.2)
D.add_scalebar(kpc_100, corner='bottom')
D.scalebar.set_label('100 kpc')
D.show_colorscale(vmin=4e-5*10**6,vmax=0.004*10**6,stretch='log',cmap='Oranges', smooth=1)
D.recenter(220.168,3.488, radius=0.05)
D.add_colorbar()
D.colorbar.set_font(size=14)
D.colorbar.set_axis_label_text('Surface brightness (\u03BCJy/Beam)')
D.colorbar.set_axis_label_font(size=18)
D.add_beam()
D.ticks.set_color('black')
D.tick_labels.set_xformat('hh:mm:ss')
D.tick_labels.set_yformat('dd:mm')
D.axis_labels.set_font(size=18)
D.tick_labels.set_font(size=18)
D.save('D-array_aplpyimage.png', dpi=144)