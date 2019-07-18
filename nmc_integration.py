import numpy as np
from my_integration import calculat_all
from matplotlib import pyplot as plt

delta_phi = 1e-10
tol = 1e-1
N = 60
xi_ = np.linspace(-0.002, 0.006, 1000)
Mp = 4.341e-9

phi_i_60 = []
phi_i_50 = []

integrands = []

def integrand(phi, xi):
	return 0.5*phi*(Mp**2 + 6*phi**2*xi**2 + phi**2*xi)/(Mp**4 - phi**4*xi**2)

def phi_f(xi):
	return np.sqrt(Mp**2*(np.sqrt(48*(xi**2) + 16*xi + 1) -1 -4*xi)/(8*(xi**2) + 2*xi))
