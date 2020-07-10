### by lshuns

## solve the lens equation using ODE
## ## lens model: Pseudo-Jaffe with external shear (could be zero)

######### Note:
############# the source should locate in the origin (xs=0)

######### Key knowledge:
########### lens equation: x-deflection(x)=xs
########### Jacobian(x) = d(xs)/dx = 1-d(deflection)/dx
########### 'lam' is an artificial parameter to change the lens equation into an odinary differential equation
########### the artificial equation: x-lam*deflection(x)=0
############## note how the artificial equation related to the lens equation (lam=1,xs=0)
########### the odinary differential equation: d(x)/dlam-deflection(x)-lam*d(deflection)/dx*d(x)/dlam = 0
######################################## ======> d(x)/dlam = (delta_ij-lam*d(deflection)/dx)^-1*deflection(x)

import numpy as np  
import scipy as sp

def images_jaffe_ode(sub_para, pert_para, mu = 0, dt = 0):

	s, a, xl1, xl2 = sub_para
	ka, ga = pert_para

	########## solve the lens equation using ODEint
	lam = np.linspace(0.0, 1.0, 10000)
	x0 = [0.0, 0.0]
	ode_res = sp.integrate.odeint(dxdlamODEINT, x0, lam, args=(s, a, xl1, xl2, ka, ga))
	x = ode_res[-1]

	######### magnification 
	if mu == 1:
		mag = np.absolute(1.0/np.linalg.det(jaclamFunc(x, s, a, xl1, xl2, ka, ga)))

		return mag


	######### time delay
	if dt == 1:
		x1, x2 = x
		xp2 = (x1 - xl1)**2 + (x2 - xl2)**2
		s2 = s**2
		a2 = a**2
		sqs = np.sqrt(s2 + xp2)
		sqa = np.sqrt(a2 + xp2)
		phi_sub = (sqs - s) - (sqa - a) - (s*np.log((s + sqs)/2.0/s) - a*np.log((a + sqa)/2.0/a))

		############# geometrical 
		geo = 0.5*(x1**2.+x2**2.)

		############# perturbation
		phi_per = 0.5*(x1**2.+x2**2.)*ka+0.5*(x1**2.-x2**2.)*ga

		return geo-phi_sub-phi_per

	######### the image position 

	return x

# deflection vector
def deflFunc(x, s, a, xl1, xl2, ka, ga):
	x1, x2 = x
	xp2 = (x1 - xl1)**2 + (x2 - xl2)**2
	dxp2dx1 = 2.0*(x1 - xl1)
	dxp2dx2 = 2.0*(x2 - xl2)
	s2 = s**2
	a2 = a**2
	sqs = np.sqrt(s2 + xp2)
	sqa = np.sqrt(a2 + xp2)
	dsqsdx1 = 0.5/sqs*dxp2dx1
	dsqsdx2 = 0.5/sqs*dxp2dx2
	dsqadx1 = 0.5/sqa*dxp2dx1
	dsqadx2 = 0.5/sqa*dxp2dx2
	alp1 = dsqsdx1 - dsqadx1 - s/(s + sqs)*dsqsdx1 + a/(a + sqa)*dsqadx1
	alp2 = dsqsdx2 - dsqadx2 - s/(s + sqs)*dsqsdx2 + a/(a + sqa)*dsqadx2

	return np.array([alp1, alp2])

# Jacobian matrix with lam
def jaclamFunc(x, s, a, xl1, xl2, ka, ga, lam=1.0):
	x1, x2 = x
	xp2 = (x1 - xl1)**2 + (x2 - xl2)**2
	dxp2d1 = 2.0*(x1 - xl1)
	dxp2d2 = 2.0*(x2 - xl2)
	dxp2d11 = 2.0
	dxp2d12 = 0.0
	dxp2d22 = 2.0
	s2 = s**2
	a2 = a**2
	sqs = np.sqrt(s2 + xp2)
	sqa = np.sqrt(a2 + xp2)
	dsqsd1 = 0.5/sqs*dxp2d1
	dsqsd2 = 0.5/sqs*dxp2d2
	dsqad1 = 0.5/sqa*dxp2d1
	dsqad2 = 0.5/sqa*dxp2d2
	dsqsd11 = 0.5/sqs*dxp2d11 - 0.5/sqs/sqs*dsqsd1*dxp2d1
	dsqsd12 = 0.5/sqs*dxp2d12 - 0.5/sqs/sqs*dsqsd2*dxp2d1
	dsqsd22 = 0.5/sqs*dxp2d22 - 0.5/sqs/sqs*dsqsd2*dxp2d2
	dsqad11 = 0.5/sqa*dxp2d11 - 0.5/sqa/sqa*dsqad1*dxp2d1
	dsqad12 = 0.5/sqa*dxp2d12 - 0.5/sqa/sqa*dsqad2*dxp2d1
	dsqad22 = 0.5/sqa*dxp2d22 - 0.5/sqa/sqa*dsqad2*dxp2d2
	j11 = 1.0 - ka - ga - lam*(dsqsd11 - dsqad11  \
          - s/(s + sqs)*dsqsd11 + s/(s + sqs)**2*dsqsd1**2 + a/(a + sqa)*dsqad11 - a/(a + sqa)**2*dsqad1**2)
	j12 = - lam*(dsqsd12 - dsqad12  \
          - s/(s + sqs)*dsqsd12 + s/(s + sqs)**2*dsqsd1*dsqsd2 + a/(a + sqa)*dsqad12 - a/(a + sqa)**2*dsqad1*dsqad2)
	j22 = 1.0 - ka + ga - lam*(dsqsd22 - dsqad22  \
          - s/(s + sqs)*dsqsd22 + s/(s + sqs)**2*dsqsd2**2 + a/(a + sqa)*dsqad22 - a/(a + sqa)**2*dsqad22**2)
	return np.array([[j11, j12], [j12, j22]])

# The function called by odeint
def dxdlamODEINT(x, lam, s, a, xl1, xl2, ka, ga):
    defl = deflFunc(x, s, a, xl1, xl2, ka, ga)
    jaclam = jaclamFunc(x, s, a, xl1, xl2, ka, ga, lam=lam)
    return np.dot(np.linalg.inv(jaclam), defl)


################################## for test 
if __name__ == "__main__":

	from parameters import physics, translation, astronomy
	from distance import distance


	######## -------------------- constants

	C = physics.C # km s^-1
	G = physics.G2 # pc M_sun^-1 (km/s)^2

	############# ------------------------ the physical parameters here should be consistent with those in Jaffe_elli.input
	######################### general
	## source redshift
	zs = 0.8 
	## corresponding angular distance
	Ds = distance(zs, angular=1) # Mpc

	## lens redshift 
	zl = 0.4
	## corresponding angular distance
	Dl = distance(zl, angular=1) # Mpc
	Dls = distance(zs, zl, angular=1) # Mpc

	######################### subhalo
	##### velocity dispersion
	sigma_v_sub = 10.0 # km s^-1

	##### angular critical radius
	b_sub_rad = 4.*np.pi*(sigma_v_sub/C)**2.*Dls/Ds # radian
	# print b_sub_rad
	###################### 1.2855202779e-06 (radian)
	b_sub_arc = b_sub_rad/np.pi*648000. # arcsecond
	# print b_sub_arc
	# ###################### 0.265157591049 (arcsecond)

	# ##### break radius
	a = 2.0
	# a_arc = a*b_sub_arc # arcsecond
	# print a_arc
	# ##################### 0.00255766807884 (arcsecond)

	# ##### core radius
	s = 0.1
	# s_arc = b_sub_arc*s 
	# print s_arc
	# ##################### 0.000127883403942 (arcsecond)

	####### impact parameter
	xl1 = 0.8
	xl2 = 0.8
	# xl_arc = xl1*b_sub_arc
	# print xl_arc
	# ##################### 0.00102306723154 (arcsecond)

	sub_para = (s,a,xl1,xl2)
	pert_para = (1./3.,1./3.)
	
	image = images_jaffe_ode(sub_para,pert_para)
	print image
	print images_jaffe_ode(sub_para,pert_para,mu=1)
	print images_jaffe_ode(sub_para,pert_para,dt=1)
	# x_arc = image[0]*b_sub_arc
	# y_arc = image[1]*b_sub_arc
	# print x_arc
	# print y_arc


	# mu = images_jaffe4(sub_para,mu = 1)
	# print mu


	# dt = images_jaffe4(sub_para,dt = 1)
	# print dt
