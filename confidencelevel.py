from scipy.integrate import quad


# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import sys, traceback



# Import signal
signal = np.loadtxt("../USTL/signal.txt")
# print(signal)



# mean = signal[0][0] 
# std = signal[0][1]




# x1 = mean + std
# x2 = mean + 2.0 * std
# x1 = 0
# x2 = 16


# print('Integration bewteen {} and {} --> '.format(x1,x2),res)
def normal_distribution_function(x, mean, std):
    value = scipy.stats.norm.pdf(x,mean,std)
    return value

def conf(mean, std, x1, x2):
	res, err = quad(normal_distribution_function, x1, x2, args=(mean,std))
	return res



def mu_strong(omega, th, t):
	mean = omega[t, 0] - th
	std = omega[t, 1]

	if (mean > 0):
		x1 = 0
		x2 = 2 * mean
		cf = conf(mean, std, x1, x2)
	else:
		cf = 0
	return cf


def mu_weak(omega, th, t):
	# print(omega[t, 0], th)
	mean = omega[t, 0] - th
	std = omega[t, 1]

	if (mean < 0):
		x1 = 2 * mean
		x2 = 0 
		cf = conf(mean, std, x1, x2)
	else:
		cf = 0
	return cf

# print(mu_strong(signal, 8, 0))
# print(mu_weak(signal, 8, 0))

def calculatecf_strong(requirement, t):
	req = requirement[0]
	varphi = requirement[1]
	# print(varphi)
	cf = np.zeros((1,2))
	# print(varphi)
	if req[0] == "mu":
		th = varphi
		omega = req[1]
		# try:
		cf = np.array([0, mu_strong(omega, th, t)])
			# cf = 0, mu_strong(omega, th, t)
		# except:
			# print("Error: Trace is not long enough.")
			# sys.exit()


	elif req[0] == "neg":
		c = calculatecf_weak((varphi[0], varphi[1]), t)
		# print(c)
		cf = np.array([0, c[0]])
		# print(varphi)
		

	elif req[0] == "and":
		# print(varphi)
		pho1 = calculatecf_strong((varphi[0][0], varphi[0][1]), t)
		pho2 = calculatecf_strong((varphi[1][0], varphi[1][1]), t)

		pho_low = min(pho1[0], pho2[0])
		pho_up = min(pho1[1], pho2[1])

		cf = np.array([pho_low, pho_up])
		# print(pho)



	elif req[0] == "always":
		t1 = req[1][0]
		t2 = req[1][1]
		cf = np.zeros((t2-t1+1, 2))
		for ti in range(t1, t2+1):
			cf[ti - t1] = calculatecf_strong((varphi[0], varphi[1]), t+ti)
		cf = np.amin(cf, axis=0)

	elif req[0] == "eventually":
		t1 = req[1][0]
		t2 = req[1][1]
		cf = np.zeros((t2-t1+1, 2))
		for ti in range(t1, t2+1):
			cf[ti-t1] = calculatecf_strong((varphi[0], varphi[1]), t+ti)
		cf = np.amax(cf, axis=0)


	elif req[0] == "until":
		t1 = req[1][0]
		t2 = req[1][1] 
		pho1 = np.zeros((t2-t1+1, 2))
		pho2 = np.zeros((t2-t1+1, 2))
		for ti in range(t1, t2+1):
			pho1[ti-t1] = calculatecf_strong((varphi[0][0], varphi[0][1]), t+ti)
			pho2[ti-t1] = calculatecf_strong((varphi[1][0], varphi[1][1]), t+ti)
			# print(pho1[ti-t1])
			# print(pho2[ti-t1])

		pho3 = np.zeros((t2-t1+1, 2))


		for ti in range(t1, t2+1):
			pho = np.min(pho1[:ti-t1+1], axis=0)
			pho_low = min(pho[0], pho2[ti - t1, 0])
			pho_up = min(pho[1], pho2[ti - t1, 1])			
			pho3[ti - t1] = np.array([pho_low, pho_up])
		cf = np.max(pho3, axis=0) 
		# print(cf)

	return cf

def calculatecf_weak(requirement, t):
	req = requirement[0]
	varphi = requirement[1]
	# print(varphi)
	cf = np.zeros((1,2))
	# print(varphi)
	if req[0] == "mu":
		th = varphi
		omega = req[1]
		# print(mu_weak(omega, th, t))
		# try:
		cf = np.array([mu_weak(omega, th, t), 1])
			# cf = 0, mu_strong(omega, th, t)
		# except:
		# 	print("Error: Trace is not long enough.")
		# 	sys.exit()



	elif req[0] == "neg":
		c = calculatecf_strong((varphi[0], varphi[1]), t)
		# print(c)
		cf = np.array([c[1], 1])
		# print(varphi)
		

	elif req[0] == "and":
		# print(varphi)
		pho1 = calculatecf_weak((varphi[0][0], varphi[0][1]), t)
		pho2 = calculatecf_weak((varphi[1][0], varphi[1][1]), t)

		pho_low = max(pho1[0], pho2[0])
		pho_up = max(pho1[1], pho2[1])

		cf = np.array([pho_low, pho_up])
		# print(pho)



	elif req[0] == "always":
		t1 = req[1][0]
		t2 = req[1][1]
		cf = np.zeros((t2-t1+1, 2))
		for ti in range(t1, t2+1):
			cf[ti - t1] = calculatecf_weak((varphi[0], varphi[1]), t+ti)
		cf = np.amax(cf, axis=0)

	elif req[0] == "eventually":
		t1 = req[1][0]
		t2 = req[1][1]
		cf = np.zeros((t2-t1+1, 2))
		for ti in range(t1, t2+1):
			cf[ti-t1] = calculatecf_weak((varphi[0], varphi[1]), t+ti)
		cf = np.amin(cf, axis=0)


	elif req[0] == "until":
		t1 = req[1][0]
		t2 = req[1][1] 
		pho1 = np.zeros((t2-t1+1, 2))
		pho2 = np.zeros((t2-t1+1, 2))
		for ti in range(t1, t2+1):
			pho1[ti-t1] = calculatecf_weak((varphi[0][0], varphi[0][1]), t+ti)
			pho2[ti-t1] = calculatecf_weak((varphi[1][0], varphi[1][1]), t+ti)
			# print(pho1[ti-t1])
			# print(pho2[ti-t1])

		pho3 = np.zeros((t2-t1+1, 2))


		for ti in range(t1, t2+1):
			pho = np.max(pho1[:ti-t1+1], axis=0)
			pho_low = max(pho[0], pho2[ti - t1, 0])
			pho_up = max(pho[1], pho2[ti - t1, 1])			
			pho3[ti - t1] = np.array([pho_low, pho_up])
		cf = np.min(pho3, axis=0) 
		# print(cf)
	return cf



def main():
  #print("Hello World!")
  



	print(calculatecf_strong((("neg", 0), (("mu", signal), (20))), 0))
	print(calculatecf_weak((("neg", 0), (("mu", signal), (20))), 0))
	
	
	v1 = (("mu", signal), (3))
	
	v2 = (("mu", signal), (3))
	
	print(calculatecf_strong((("until",(0,1)), (v1, v2)), 0))
	print(calculatecf_weak((("until",(0,1)), (v1, v2)), 0))
	
	
	print(calculatecf_strong((("always", (1,3)), (("mu", signal), (11.9))), 0))
	print(calculatecf_weak((("always", (1,3)), (("mu", signal), (11.9))), 0))
	
	
	print(calculatecf_strong((("eventually", (1,3)), (("mu", signal), (11.9))), 0))
	print(calculatecf_weak((("eventually", (1,3)), (("mu", signal), (11.9))), 0))
	
	# print(calculatecf_strong((("always", (1,3)), (("neg", 0), (("mu", signal), (1)))), 3))
	
	print(calculatecf_strong((("always", (1,3)), (("mu", signal), (15))), 0))
	print(calculatecf_weak((("always", (1,3)), (("mu", signal), (15))), 0))
	

# main()



#----------------------------------------------------------------------------------------#
# plot integration surface
def drawdistribution(x1, x2, x_min, x_max, mean, std):
	x = np.linspace(x_min, x_max, 100)
	y = scipy.stats.norm.pdf(x,mean,std)
	plt.plot(x,y, color='black')
	ptx = np.linspace(x1, x2, 10)
	pty = scipy.stats.norm.pdf(ptx,mean,std)
	plt.fill_between(ptx, pty, color='#0b559f', alpha='1.0')
	plt.grid()
	plt.xlim(x_min,x_max)
	plt.ylim(0,0.25)
	# plt.title('How to integrate a normal distribution in python ?',fontsize=10)
	plt.xlabel('x')
	plt.ylabel('Normal Distribution')
	plt.savefig("integrate_normal_distribution.png")
	plt.show()
#----------------------------------------------------------------------------------------#


# call plot
# x_min = 0.0 #x-axis range
# x_max = 20.0 #x-axis range
# drawdistribution(9, 11, x_min, x_max, 10, 2)


