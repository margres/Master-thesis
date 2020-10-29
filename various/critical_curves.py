## by lshuns

############# calculate and plot the critical curves and caustics
################## lens model: Pseudo-Jaffe + Perturbation

import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as op


def mu_inver(rt, theta_t, sub_para, pert_para):

	xl, yl, a = sub_para
	kappa, gamma1 = pert_para

	xt = rt*np.cos(theta_t)
	yt = rt*np.sin(theta_t)

	xt2 = xt**2.
	yt2 = yt**2.
	rt2 = rt**2.

	psi = (a**2.+rt2)**0.5

	############# subhalo
	phixx_sub = -xt2/rt**3+1./rt-xt2/rt2/psi+2.*xt2*(-a+psi)/rt2**2.-(-a+psi)/rt2
	phiyy_sub = -yt2/rt**3+1./rt-yt2/rt2/psi+2.*yt2*(-a+psi)/rt2**2.-(-a+psi)/rt2
	phixy_sub = -xt*yt/rt**3-xt*yt/rt2/psi+2.*xt*yt*(-a+psi)/rt2**2.

	############ perturbation
	phixx_per = kappa+gamma1
	phiyy_per = kappa-gamma1
	phixy_per = 0.

	phixx = phixx_sub+phixx_per
	phiyy = phiyy_sub+phiyy_per
	phixy = phixy_sub+phixy_per

	return 1-phixx-phiyy-phixy*phixy+phixx*phiyy

def map(rt,theta_t,sub_para,pert_para):

	xl, yl, a = sub_para
	kappa, gamma1 = pert_para

	xt = rt*np.cos(theta_t)
	yt = rt*np.sin(theta_t)

	x = xt+xl
	y = yt+yl


	######## subhalo
	psi = (a**2.+rt**2.)**0.5
	phix_sub = xt/rt-xt*(psi-a)/rt**2.
	phiy_sub = yt/rt-yt*(psi-a)/rt**2.


	######## perturbation
	phix_per = x*(kappa+gamma1)
	phiy_per = y*(kappa-gamma1)

	return (x-phix_per-phix_sub, y-phiy_per-phiy_sub)



####
xl = 1.
yl = 1.
a = 20.00000000007779

sub_para = (xl,yl,a)

kappa = 0.3125
gamma1 = 0.3125

pert_para = (kappa,gamma1)

N_theta = 100
theta_t = np.linspace(0,2*np.pi,N_theta)  

rt_res = []

N_r = 100
rmin = 0.001
rmax = 5.
rt = np.linspace(rmin,rmax,N_r)

for i in xrange(N_theta):
	for j in xrange(N_r-1):
		if mu_inver(rt[j],theta_t[i],sub_para,pert_para)*mu_inver(rt[j+1],theta_t[i],sub_para,pert_para)<=0:
			tmp = op.brentq(mu_inver, rt[j],rt[j+1], args=(theta_t[i],sub_para,pert_para))
			rt_res.append(tmp)

cosTheta_t = np.cos(theta_t)
sinTheta_t = np.sin(theta_t)


for i in xrange(N_theta):
	xi = rt_res[i]*cosTheta_t[i]+xl
	yi = rt_res[i]*sinTheta_t[i]+yl
	plt.plot(xi,yi,'.',color = 'blue')

	tmp = map(rt_res[i],theta_t[i],sub_para,pert_para)
	xs = tmp[0]
	ys = tmp[1]
	plt.plot(xs,ys,'.',color = 'red')


plt.savefig('../../results/caustics_critical_jaffe/critical_caustics.png')
