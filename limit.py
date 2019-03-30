#! /usr/local/bin/python3.7

import numpy as np

def f(x):
	return x

def limit_finder(f, a, res, dx, tol = 1e-3):
	i = 0
	temp = 0
	while(abs(temp - res) >= tol):
		temp += 0.5*(f(a + i*dx) + f(a + (i+1)*dx))*dx
		i += 1
	return a + (i-1)*dx

print(limit_finder(f, 0, 4.5, 1e-2))