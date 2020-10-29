### by lshuns

########## generate the caustics
import numpy as np
import scipy.optimize as op

### The determinant of Jacobian matrix 
# pseudo-Jaffe with external perturbation 
def jacPJFunc(r, theta, s, a, xl1, xl2, ka, ga):
	x1 = xl1 + r*np.cos(theta)
	x2 = xl2 + r*np.sin(theta)
	xp = x1 - xl1
	yp = x2 - xl2

	xp2 = xp**2. + yp**2.
	xp2x = 2.*xp
	xp2y = 2.*yp

	psis = np.sqrt(s**2. + xp2)
	psia = np.sqrt(a**2. + xp2)
	psisx = xp2x/2./psis
	psisy = xp2y/2./psis
	psiax = xp2x/2./psia
	psiay = xp2y/2./psia

	phix = xp/xp2*((psis-s)-(psia-a))
	phiy = yp/xp2*((psis-s)-(psia-a))

	phixx = 1./xp2*((psis-s)-(psia-a))-phix/xp2*xp2x+xp/xp2*(psisx-psiax)
	phixy = -phix/xp2*xp2y+xp/xp2*(psisy-psiay)
	phiyy = 1./xp2*((psis-s)-(psia-a))-phiy/xp2*xp2y+yp/xp2*(psisy-psiay)

	return np.linalg.det(np.array([[1-ka-ga-phixx, -phixy], [-phixy, 1-ka+ga-phiyy]]))


# pseudo-Jaffe with ellipticity and external perturbation 
def jacPJqFunc(r, theta, s, a, q, xl1, xl2,ka,ga):
	x1 = xl1 + r*np.cos(theta)
	x2 = xl2 + r*np.sin(theta)
	xp = x1 - xl1
	yp = x2 - xl2
	e = np.sqrt(1.-q**2.)
	psis = np.sqrt(q**2.*(s**2.+xp**2.)+yp**2.)
	psia = np.sqrt(q**2.*(a**2.+xp**2.)+yp**2.)
	psisx = q**2.*xp/psis
	psisy = yp/psis
	psiax = q**2.*xp/psia
	psiay = yp/psia

	tans = e*xp/(psis+s)
	tana = e*xp/(psia+a)
	tansx = e/(psis+s)-tans/(psis+s)*psisx
	tansy = -tans/(psis+s)*psisy
	tanax = e/(psia+a)-tana/(psia+a)*psiax
	tanay = -tana/(psia+a)*psiay
	
	tanhs = e*yp/(psis+q**2.*s)
	tanha = e*yp/(psia+q**2.*a)
	tanhsx = -tanhs/(psis+q**2.*s)*psisx
	tanhsy = e/(psis+q**2.*s)-tanhs/(psis+q**2.*s)*psisy
	tanhax = -tanha/(psia+q**2.*a)*psiax
	tanhay = e/(psia+q**2.*a)-tanha/(psia+q**2.*a)*psiay
    
	phixx = q/e*(1./(1+tans**2.)*tansx-1./(1+tana**2.)*tanax)
	phixy = q/e*(1./(1+tans**2.)*tansy-1./(1+tana**2.)*tanay)
	phiyy = q/e*(1./(1-tanhs**2.)*tanhsy-1./(1-tanha**2.)*tanhay)

	return np.linalg.det(np.array([[1-ka-ga-phixx, -phixy], [-phixy, 1-ka+ga-phiyy]]))



# deflection vector
# pseudo-Jaffe with external perturbation
def deflPJFunc(x, s, a, xl1, xl2, ka, ga):
	x1, x2 = x
	xp = x1 - xl1
	yp = x2 - xl2
	xp2 = xp**2. + yp**2.
	psis = np.sqrt(s**2. + xp2)
	psia = np.sqrt(a**2. + xp2)
	phix = xp/xp2*((psis-s)-(psia-a))+x1*(ka+ga)
	phiy = yp/xp2*((psis-s)-(psia-a))+x2*(ka-ga)

	return np.array([phix, phiy])

# pseudo-Jaffe with ellipticity and external perturbation 
def deflPJqFunc(x, s, a, q, xl1, xl2,ka,ga):
	x1,x2=x
	xp = x1 - xl1
	yp = x2 - xl2
	e = np.sqrt(1.-q**2.)
	psis = np.sqrt(q**2.*(s**2.+xp**2.)+yp**2.)
	psia = np.sqrt(q**2.*(a**2.+xp**2.)+yp**2.)
	phix = q/e*(np.arctan(e*xp/(psis+s))-np.arctan(e*xp/(psia+a)))+x1*(ka+ga)
	phiy = q/e*(np.arctanh(e*yp/(psis+q**2.*s))-np.arctanh(e*yp/(psia+q**2.*a)))+x1*(ka+ga)

	return np.array([phix,phiy])


###############
if __name__ == '__main__':

	import matplotlib as mpl
	import matplotlib.pyplot as plt

	s = 0.1
	a = 2.0
	q = 0.5
	xl1 = 0.
	xl2 = 0.

	ka = 0.3
	ga = 0.3

	ka_q = 0
	ga_q = 0


	N_theta = 100
	thetaArr = np.linspace(0,2.*np.pi,N_theta)

	N_r = 100
	rArr = np.linspace(0.001,10.,N_r)

	cri_PJ1 = np.zeros(N_theta)
	cri_PJ2 = np.zeros(N_theta)

	cri_PJq1 = np.zeros(N_theta)
	cri_PJq2 = np.zeros(N_theta)


	for i in xrange(N_theta):
		for j in xrange(N_r-1):
			if jacPJFunc(rArr[j],thetaArr[i], s, a, xl1, xl2, ka, ga)*jacPJFunc(rArr[j+1],thetaArr[i], s, a, xl1, xl2, ka, ga)<=0:
				tmp = op.brentq(jacPJFunc, rArr[j], rArr[j+1], args=(thetaArr[i], s, a, xl1, xl2, ka, ga))
				if cri_PJ1[i]==0:
					cri_PJ1[i] = tmp
				elif cri_PJ2[i] ==0:
					cri_PJ2[i] = tmp
				else:
					raise Exception('more than two caustics')

	# 		if jacPJqFunc(rArr[j],thetaArr[i], s, a, q, xl1, xl2, ka_q, ga_q)*jacPJqFunc(rArr[j+1],thetaArr[i], s, a, q, xl1, xl2, ka_q, ga_q)<=0:
	# 			tmp = op.brentq(jacPJqFunc, rArr[j], rArr[j+1], args=(thetaArr[i], s, a, q, xl1, xl2, ka_q, ga_q))
	# 			if cri_PJq1[i]==0:
	# 				cri_PJq1[i] = tmp
	# 			elif cri_PJq2[i] ==0:
	# 				cri_PJq2[i] = tmp
	# 			else:
	# 				raise Exception('more than two caustics')
	# print cri_PJ1
	# print cri_PJ2
	# print cri_PJq1
	# print cri_PJq2

	xi1 = np.array([cri_PJ1*np.cos(thetaArr)+xl1,cri_PJ1*np.sin(thetaArr)+xl2])
	xi2 = np.array([cri_PJ2*np.cos(thetaArr)+xl1,cri_PJ2*np.sin(thetaArr)+xl2])
	
	# xiq1 = np.array([cri_PJq1*np.cos(thetaArr)+xl1,cri_PJq1*np.sin(thetaArr)+xl2])
	# xiq2 = np.array([cri_PJq2*np.cos(thetaArr)+xl1,cri_PJq2*np.sin(thetaArr)+xl2])

	
	########## caustics
	xs1 = xi1-deflPJFunc(xi1, s, a, xl1, xl2, ka, ga)
	xs2 = xi2-deflPJFunc(xi2, s, a, xl1, xl2, ka, ga)
	
	# xsq1 = xiq1-deflPJqFunc(xiq1, s, a, q, xl1, xl2, ka_q, ga_q)
	# xsq2 = xiq2-deflPJqFunc(xiq2, s, a, q, xl1, xl2, ka_q, ga_q)

	plt.plot(xs1[0],xs1[1],color='r')
	plt.plot(xs2[0],xs2[1],color='r',label='Axisymmetric')

	# plt.plot(xsq1[0],xsq1[1],color='b')
	# plt.plot(xsq2[0],xsq2[1],color='b',label='Elliptic')

	plt.plot(0,0,'.',color='k',label='source')
	plt.legend()
	plt.title('$\\kappa=\\gamma=0.3$')
		# ; \\kappa_q=\\gamma_q=0$')

	# plt.show()
	plt.savefig('../../results/caustics_critical_jaffe/axi_per.png')

	# plt.plot(xs1[0],xs1[1],color='r')
	# plt.plot(xs2[0],xs2[1],color='r',label='Axisymmetric')

	# plt.plot(xsq1[0],xsq1[1],color='b')
	# plt.plot(xsq2[0],xsq2[1],color='b',label='Elliptic')

	# plt.plot(0,0,'.',color='k',label='source')
	# plt.legend()
	# plt.title('$\\kappa=\\gamma=1/3; \\kappa_q=\\gamma_q=0$')

	# plt.savefig('../../results/caustics_critical_jaffe/ell_axi_enoper.png')


	# plt.plot(xs1[0],xs1[1],color='r')
	# plt.plot(xs2[0],xs2[1],color='r',label='Axisymmetric')

	# plt.plot(xsq1[0],xsq1[1],color='b')
	# plt.plot(xsq2[0],xsq2[1],color='b',label='Elliptic')

	# plt.plot(0,0,'.',color='k',label='source')
	# plt.legend()
	# plt.title('$\\kappa=\\gamma=1/3; \\kappa_q=\\gamma_q=1/3$')

	# plt.savefig('../../results/caustics_critical_jaffe/ell_axi_per.png')



