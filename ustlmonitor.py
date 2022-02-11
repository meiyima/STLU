from scipy.stats import norm
import numpy as np
import sys, traceback
# import matplotlib.pyplot as plt

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from functools import lru_cache


# Import signal
# signal = np.loadtxt("signal.txt")
# print(signal)

@lru_cache(maxsize=32)
def get_ppf(p:float):
	return norm.ppf(p)

def normalconf(mean, sigma, conf):
	p = 1 - (1 - conf) / 2 

	lower = mean - get_ppf(p) * sigma
	upper = mean + get_ppf(p) * sigma

	# print(lower, upper)

	return (lower, upper)


# print(normalconf(0, 1, 0.9))


def plogsignal(signal, mode):
	# plot
	plt.style.use('seaborn-whitegrid')
	# plt.ion()
	fig = plt.figure()
	ax = plt.axes()
	x = range(0, np.size(signal,0))
	plt.plot(x, signal[:,0], linestyle='-')
	plt.show(block=False)
	if mode == 'save':
		plt.savefig('signal.png')
	elif mode == 'show':
		plt.show()


# plogsignal(signal)



def plogsignal(signal, conf, mode):
	# plot
	plt.style.use('seaborn-whitegrid')
	fig = plt.figure()
	ax = plt.axes()
	x = range(0, np.size(signal,0))
	plt.plot(x, signal[:,0], linestyle='-')

	signalrange = normalconf(signal[:, 0], signal[:, 1], 0.9)
	# print(signalrange)
	plt.plot(x, signalrange[0], linestyle='--')
	plt.plot(x, signalrange[1], linestyle='--')
	plt.xlabel('Time')
	plt.ylabel('Value')
	if mode == 'save':
		plt.savefig('signal_'+ str(conf) +'.png')
	elif mode == 'show':
		plt.show()

	
	

# plogsignal(signal, 0.9, 'show')



# omega - th > 0
def mu(omega, th, cl, t): 
	# print(np.size(omega,0))
	# if (t > np.size(omega,0)):
	# 	print("Error: Trace is not long enough. ")
	# 	sys.exit("Error: Trace is not long enough. ")
	mean = omega[t, 0]
	sigma = omega[t, 1]
	conf = cl
	omega_cl = np.array(normalconf(mean, sigma, conf))
	# print(omega_cl) 
	pho = omega_cl - th 

	return pho

# print(mu(signal, 3, 0.9, 0))




def neg(varphi):
	# pho = np.array([-varphi[1], -varphi[0]])
	pho = - varphi[[1, 0]]
	return pho


# requirement: ((operator, para) (varphi))


def umonitor(requirement, t):
	req = requirement[0]
	varphi = requirement[1]
	# print(varphi)
	pho = np.zeros((1,2))

	if req[0] == "mu":
		th = varphi[0]
		# print(th)
		# print(th)
		cl = varphi[1]
		# print(cl)
		omega = req[1]
		# try:
		# print(omega.shape, th, cl, t)
		pho = mu(omega, th, cl, t)
		# except:
			# print("Error: Trace is not long enough.")
			# sys.exit()

	elif req[0] == "neg":
		# print(varphi)
		pho = umonitor((varphi[0], varphi[1]), t)
		pho = neg(pho)

	elif req[0] == "and":
		# print(varphi)
		pho1 = umonitor((varphi[0][0], varphi[0][1]), t)
		pho2 = umonitor((varphi[1][0], varphi[1][1]), t)

		pho_low = min(pho1[0], pho2[0])
		pho_up = min(pho1[1], pho2[1])

		pho = np.array([pho_low, pho_up])
		# print(pho)



	elif req[0] == "always":
		t1 = req[1][0]
		t2 = req[1][1]
		pho = np.zeros((t2-t1+1, 2))
		for ti in range(t1, t2+1):
			# print(ti, t1, t2)
			pho[ti - t1] = umonitor((varphi[0], varphi[1]), t+ti)
		
		# print(pho)
		pho = np.amin(pho, axis=0)
		# print(pho)

		# pho = np.minimum(pho[0, :], pho[1, :])

	elif req[0] == "eventually":
		t1 = req[1][0]
		t2 = req[1][1]
		pho = np.zeros((t2-t1+1, 2))
		for ti in range(t1, t2+1):
			pho[ti-t1] = umonitor((varphi[0], varphi[1]), t+ti)
		# pho = np.maximum(pho[0, :], pho[1, :])
		pho = np.amax(pho, axis=0)


	elif req[0] == "until":
		# ("until", (t1, t2))), (varphi1, varphi2) 
		t1 = req[1][0]
		t2 = req[1][1] 
		pho1 = np.zeros((t2-t1+1, 2))
		pho2 = np.zeros((t2-t1+1, 2))
		for ti in range(t1, t2+1):
			pho1[ti-t1] = umonitor((varphi[0][0], varphi[0][1]), t+ti)
			pho2[ti-t1] = umonitor((varphi[1][0], varphi[1][1]), t+ti)
		pho3 = np.zeros((t2-t1+1, 2))
		# for t in range(t1, t2+1):
		# 	pho = np.min(pho1[:, :t-t1+1], axis=1)
		# 	pho_low = min(pho[0], pho2[t - t1, 0])
		# 	pho_up = min(pho[1], pho2[t - t1, 1])			
		# 	pho3[t - t1] = np.array([pho_low, pho_up])
		# pho = np.maximum(pho3[0, :], pho3[1, :])

		for ti in range(t1, t2+1):
			pho = np.min(pho1[:ti-t1+1], axis=0)
			pho_low = min(pho[0], pho2[ti - t1, 0])
			pho_up = min(pho[1], pho2[ti - t1, 1])			
			pho3[ti - t1] = np.array([pho_low, pho_up])
		pho = np.max(pho3, axis=0) 

	return pho


def quan_to_boo(quan):
	# print(quan[0], quan[1])
	b = "NaN"
	if quan[0] >= 0:
		b = "Strong Satisfaction"
	elif quan[0] < 0 and quan[1] >= 0:
		b = "Weak Satisfaction"
	elif quan[0] < 0 and quan[1] < 0:
		b = "Strong Violation"
	return b

def quan_to_boo_strong(quan):
	b = "NaN"
	if quan[0] >= 0:
		b = "True"
	else: 
		b = "False"
	return b

def quan_to_boo_weak(quan):
	b = "NaN"
	if quan[1] >= 0:
		b = "True"
	else: 
		b = "False"
	return b



# print(umonitor((("mu", signal), (50, 0.95)), 0))
 #si_pred = ustl.umonitor((("eventually", (0,4)), (("neg", 0), (("mu", signal_pred), [60, 0.95]))), 0)


# -------syntax--------:
# r = varphi, signal, t_0
# varphi := ((operator, para), varphi)
# varphi := (operator, para), (varphi1, varphi2)  
# varphi := ("until", (t1, t2))), (varphi1, varphi2)  
# r = ("neg", 0), (("always", (0,4)), (("mu", 0), [1, 0.9])), signal, 0
# --------------------


