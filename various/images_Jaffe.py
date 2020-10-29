### by lshuns

## solve the lens equation to get information of images
## ## lens model: Pseudo-Jaffe + Perturbation

######### Note:
############# the origin is set in the centre of perturbation and the x-aixs is parallel to the direction of shear, that means gamma2 = 0
############# the source is set in the origin (xs=0, ys=0)
############## positions is rescaled using the critical radius (b) of the subhalo (which is different from the Einstein radius (thetaE) or break radius (a))


######### Caveat:
############# images with theta ~ [0,0.5*np.pi,np.pi,1.5*np.pi,2.*np.pi] is ignored

import numpy as np  
import scipy.optimize as op

def images_Jaffe(sub_para, pert_para, mu = 0, dt = 0):

	xl, yl, s, a = sub_para
	kappa, gamma1 = pert_para

	############### solve the theta-function ++++++++++++(start)

	N_theta_t = 100m                  
	d_theta_t = 1e-3

	node_theta_t = np.array([0,0.5*np.pi,np.pi,1.5*np.pi,2.*np.pi])

	theta_t_res = []

	for i in xrange(len(node_theta_t)-1):

		theta_t = np.linspace(node_theta_t[i]+d_theta_t,node_theta_t[i+1]-d_theta_t,N_theta_t)

		for j in xrange(len(theta_t)-1):
			if thetaFunc(theta_t[j],kappa,gamma1,xl,yl,s,a)*thetaFunc(theta_t[j+1],kappa,gamma1,xl,yl,s,a)<=0:
				tmp = op.brentq(thetaFunc, theta_t[j],theta_t[j+1], args=(kappa,gamma1,xl,yl,s,a))
				theta_t_res.append(tmp)

	Ax = 1.-kappa-gamma1
	Ay = 1.-kappa+gamma1
	
	cosTheta = np.cos(theta_t_res)
	sinTheta = np.sin(theta_t_res)
	C = Ay*yl*cosTheta-Ax*xl*sinTheta
	rt = C/(Ax-Ay)/sinTheta/cosTheta

	######### true solutions
	theta_t_res = np.array(theta_t_res)
	rt = np.array(rt)	

	theta_t_res = theta_t_res[np.where(rt>0)]
	rt = rt[np.where(rt>0)]

	nimage = len(theta_t_res)

	############### solve the theta-function ++++++++++++(end)



	######### magnification 
	if mu == 1:

		xt = rt*np.cos(theta_t_res)
		yt = rt*np.sin(theta_t_res)

		xt2 = xt**2.
		yt2 = yt**2.
		rt2 = rt**2.

		psi1 = (s**2.+rt2)**0.5
		psi2 = (a**2.+rt2)**0.5

		############# subhalo
		phixx_sub = xt2/rt2*(1./psi1-1./psi2)-2.*xt2/rt2**2.*(a-s-psi2+psi1)+(a-s-psi2+psi1)/rt2
		phiyy_sub = yt2/rt2*(1./psi1-1./psi2)-2.*yt2/rt2**2.*(a-s-psi2+psi1)+(a-s-psi2+psi1)/rt2
		phixy_sub = xt*yt/rt2*(1./psi1-1./psi2)-2.*yt*xt/rt2**2.*(a-s-psi2+psi1)

		############ perturbation
		phixx_per = kappa+gamma1
		phiyy_per = kappa-gamma1
		phixy_per = 0.

		phixx = phixx_sub+phixx_per
		phiyy = phiyy_sub+phiyy_per
		phixy = phixy_sub+phixy_per

		return (nimage, 1./(1-phixx-phiyy-phixy*phixy+phixx*phiyy))


	######### time delay
	if dt == 1:

		rt2 = rt**2.
		psi1 = (s**2.+rt**2.)**0.5
		psi2 = (a**2.+rt**2.)**0.5

		xt = rt*np.cos(theta_t_res)
		yt = rt*np.sin(theta_t_res)

		x = xt + xl
		y = yt + yl

		############# geometrical 
		geo = 0.5*(x**2.+y**2.)

		############# subhalo
		phi_sub = psi1-s-psi2+a-s*np.log((s+psi1)/2./s)+a*np.log((a+psi2)/2./a)

		############# perturbation
		phi_per = 0.5*(x**2.+y**2.)*kappa+0.5*(x**2.-y**2.)*gamma1

		return (nimage, geo-phi_sub-phi_per, geo, phi_sub,phi_per)



	######### the image position 
	xt = rt*np.cos(theta_t_res)
	yt = rt*np.sin(theta_t_res)

	x = xt + xl
	y = yt + yl

	return (nimage,x,y)



def thetaFunc(theta_t, kappa, gamma1, xl, yl, s, a):
	Ax = 1.-kappa-gamma1
	Ay = 1.-kappa+gamma1
	cosTheta = np.cos(theta_t)
	sinTheta = np.sin(theta_t)

	C = Ay*yl*cosTheta-Ax*xl*sinTheta
	rt = C/(Ax-Ay)/sinTheta/cosTheta
	psi1 = (s**2.+rt**2.)**0.5
	psi2 = (a**2.+rt**2.)**0.5

	return Ax*xl+Ax*rt*cosTheta-cosTheta*(psi1-s-psi2+a)/rt


################################## for test 
if __name__ == "__main__":

	from parameters import physics, translation, astronomy
	from distance import distance

	######## -------------------- constants

	C = physics.C # km s^-1
	G = physics.G2 # pc M_sun^-1 (km/s)^2


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

	## critical surface density 
	Sigma_cr = C**2./4./np.pi/G*Ds/Dl/Dls*1e6
	# print Sigma_cr
	# ###################### 2.8340522327e+15 (M_sun Mpc^-2)



	######################### subhalo
	##### the total mass
	# M_sub = 1e8 ## M_sun

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
	# # print a
	# # ####################### 
	# a_arc = a*b_sub_arc # arcsecond
	# print a_arc
	# ##################### 0.0132578795524 (arcsecond)

	# ##### core radius
	s = 0.1
	# s_arc = b_sub_arc*s 
	# print s_arc
	# ##################### 0.000265157591049 (arcsecond)

	
	##########
	# M = a*np.pi*Dl**2.*b_sub_rad**2.*Sigma_cr
	# print M


	############# position 
	#################### scaled by angular critical radius
	xl = 0.8
	yl = 0.8
	# print xl*b_sub_arc
	# print yl*b_sub_arc
	# ##################### 0.00132578795524 (arcsecond)
	# ##################### 0.00132578795524 (arcsecond)

	sub_para = (xl,yl,s,a)
	
	# kappa = 0.333333
	# gamma1 = 0.333333
	kappa = 1./3.
	gamma1 = 1./3.

	pert_para = (kappa,gamma1)

	image = images_Jaffe(sub_para,pert_para)
	print image
	print images_Jaffe(sub_para,pert_para, mu=1)
	print images_Jaffe(sub_para,pert_para,dt=1)
	# x_arc = image[1]*b_sub_arc
	# y_arc = image[2]*b_sub_arc
	# print x_arc
	# print y_arc


	# mu = images_Jaffe(sub_para,pert_para,mu = 1)
	# print mu


	# dt = images_Jaffe(sub_para,pert_para,dt = 1)[1]

	# dt_sec = (1+zl)/C*Dl*Ds/Dls*b_sub_rad**2.*dt*1e3*translation.PCtoM
	# print dt_sec
	# # dt_day = dt_sec/86400.
	# # print dt_day

	# # ################# for plot
	# # XY_min = 1.0
	# # XY_max = 100.0
	# # N_XY = 10
	# # XY_array = np.linspace(XY_min,XY_max,N_XY)

	# # xi = np.zeros(N_XY)
	# # yi = np.zeros(N_XY)
	# # mu = np.zeros(N_XY)
	# # dt_sec = np.zeros(N_XY)
	# # dt_geo_sec = np.zeros(N_XY)
	# # dt_sub_sec = np.zeros(N_XY)
	# # dt_per_sec = np.zeros(N_XY)


	# # kappa = 0.3125
	# # gamma1 = 0.3125
	# # pert_para = (kappa,gamma1)

	# # for i in xrange(N_XY):
	# # 	xl = XY_array[i]
	# # 	yl = XY_array[i]
	# # 	sub_para = (xl,yl,a)

	# # 	########### image position
	# # 	tmp = images_Jaffe(sub_para,pert_para)
	# # 	xi[i] = tmp[1]
	# # 	yi[i] = tmp[2]

	# # 	############## maginification
	# # 	tmp = images_Jaffe(sub_para,pert_para,mu=1)
	# # 	mu[i] = tmp[1] 
	
	# # 	############# time delay
	# # 	tmp = images_Jaffe(sub_para,pert_para,dt = 1)
	# # 	dt_sec[i] = (1+zl)/C*Dl*Ds/Dls*b_sub_rad**2.*tmp[1]*1e3*translation.PCtoM
	# # 	dt_geo_sec[i] = (1+zl)/C*Dl*Ds/Dls*b_sub_rad**2.*tmp[2]*1e3*translation.PCtoM
	# # 	dt_sub_sec[i] = (1+zl)/C*Dl*Ds/Dls*b_sub_rad**2.*tmp[3]*1e3*translation.PCtoM
	# # 	dt_per_sec[i] = (1+zl)/C*Dl*Ds/Dls*b_sub_rad**2.*tmp[4]*1e3*translation.PCtoM


	# # # import matplotlib.pyplot as plt 

	# # # plt.plot(XY_array,dt_sec,label="total")
	# # # plt.plot(XY_array,dt_geo_sec,label="geometrical")
	# # # plt.plot(XY_array,dt_sub_sec,label="subhalo")
	# # # plt.plot(XY_array,dt_per_sec,label="perturbation")
	# # # plt.legend()
	# # # plt.show()



