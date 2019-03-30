#! /usr/local/bin/python3.7

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import physical_constants
from scipy.integrate import odeint



def i_prime(i, x, y, params):
    l, w_i = params[0] , params[1]
    i_p = 3./2 * i * (w_i * i**2 + x**2 - y**2 - w_i)
    return i_p

def x_prime(i, x, y, params):
    l , w_i = params[0] , params[1]
    x_p = np.sqrt(3./2) * l * y**2 + 3./2 * x * (w_i * i**2 + x**2 - y**2 - 1)
    return x_p

def y_prime(i, x, y, params):
    l , w_i = params[0] , params[1]
    y_p = -1 * np.sqrt(3./2) * l * x * y + 3./2 * y * (w_i * i**2 + x**2 - y**2 + 1)
    return y_p


def system_3d(init_vals, t):
    i = init_vals[0]
    x = init_vals[1]
    y = init_vals[2]
    
    params = [3,0]
    
    ip = i_prime(i, x, y, params)
    xp = x_prime(i, x, y, params)
    yp = y_prime(i, x, y, params)
    
    return [ip, xp, yp]

def backward_integration(f, init_values, n, t_max):
	t = np.linspace(0, t_max, n)
	bSolution = odeint(f, init_values, t)
	i_sol = bSolution[:,0]
	x_sol = bSolution[:,1]
	y_sol = bSolution[:,2]

	#print('Initial Conditions:')
	#print('Matter = %.15f, Radiation = %.15f, Lambda = %.15f'%(m_sol[-1], r_sol[-1], l_sol[-1]))

	total_init = i_sol[-1] + x_sol[-1] + y_sol[-1]
	print('Total = %.15f'%(total_init))

	return [i_sol[-1], x_sol[-1], y_sol[-1]]

def forward_integration(f, init_values, n, t_max):
	t = np.linspace(0, t_max, n)
	fSolution = odeint(f, init_values, t)
	i_sol = fSolution[:,0]
	x_sol = fSolution[:,1]
	y_sol = fSolution[:,2]

	print('Attractor point:')
	print('I = %.15f, X = %.15f, Y = %.15f'%(i_sol[-1], x_sol[-1], y_sol[-1]))

	return [i_sol, x_sol, y_sol]

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')

ax.set_xlim(0,1)
ax.set_ylim(-1,1)
ax.set_zlim(0,1)

ax.set_xticks([0,1])
ax.set_yticks([-1,1])
ax.set_zticks([0,1])

bbox_props = dict(boxstyle='circle,pad=0.30', fc='white', ec='#980000', lw=2)

ax.set_xlabel(r'$I$', fontsize=16)
ax.set_ylabel(r'$X$', fontsize=16)
ax.set_zlabel(r'$Y$', fontsize=16)

ax.view_init(elev=40., azim=25.)
ax.grid(False)

# Initial conditions
eps = 1e-6

x = -1 + 10*eps
i = eps
y = 1 - x**2 - i**2
# x = 0.408248290463784
# y = 0.912870929175286
# i = 0.000000000000006
initials = [i,x,y]

n = 100000

params = [3,0] # Lambda, w_i

#b_sol = backward_integration(system_3d, initials, n, -100)
f_sol = forward_integration(system_3d, initials, n, 10)

ax.plot(f_sol[0], f_sol[1], f_sol[2], zorder=4, color='C0', lw=2)
#ax.plot(b_sol[0], b_sol[1], b_sol[2], color='C0', lw=2)

# Boundary
def circ(x):
    my_circ = np.sqrt(np.abs(1 - x*x))
    return my_circ
s_i = np.linspace(0,1,100)
s_i2 = np.linspace(-1,1,200)
def circ2(x):
    my_circ2 = 1 - x*x
    return my_circ2

# Phase Space Boundaries:
coefs = (1, 1, 1)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1 
# Radii corresponding to the coefficients:
rx, ry, rz = 1/np.sqrt(coefs)

# Set of all spherical angles:
u = np.linspace(-np.pi/2, np.pi/2, 100)
v = np.linspace(0, np.pi/2, 100)

# Cartesian coordinates that correspond to the spherical angles:
# (this is the equation of an ellipsoid):
x = rx * np.outer(np.cos(u), np.sin(v))
y = ry * np.outer(np.sin(u), np.sin(v))
z = rz * np.outer(np.ones_like(u), np.cos(v))

# Surface Plot of the Compact Phase Space:
ax.plot_surface(x, y, z,  rstride=10, cstride=10, color = 'grey', alpha=0.3 ) #, edgecolors = 'black')


ax.plot([0,0],[-1,1],[0,0], 'k-', lw=1)
ax.plot(s_i,circ(s_i),zdir = 'z',color = 'black',alpha = 0.8)
ax.plot(s_i,-circ(s_i),zdir = 'z',color = 'black',alpha = 0.8)
ax.plot(s_i2,circ(s_i2),zdir = 'x',color = 'black',alpha = 0.8)

# Labeling the critical points in the 3D plot
c_points = [[0,-1,0],[0,1,0],[1,0,0],[1 - (params[0]/np.sqrt(6))**2 - (np.sqrt(1 - params[0]**2/6))**2, params[0]/np.sqrt(6), np.sqrt(1 - params[0]**2/6)], [1 - (np.sqrt(3./2)*(1 + params[1])/params[0])**2 - (np.sqrt(3./2)*np.sqrt(1 - params[1]**2)/params[0])**2, np.sqrt(3./2)*(1 + params[1])/params[0], np.sqrt(3./2)*np.sqrt(1 - params[1]**2)/params[0]]]
ax.text(c_points[0][0], c_points[0][1], c_points[0][2], '$X_{-}$', ha='center', va='center', rotation=0,
			size=12, color='black', bbox=bbox_props)
ax.text(c_points[1][0], c_points[1][1], c_points[1][2], '$X_{+}$', ha='center', va='center', rotation=0,
			size=12, color='black', bbox=bbox_props)
ax.text(c_points[2][0], c_points[2][1], c_points[2][2], '$I$', ha='center', va='center', rotation=0,
			size=12, color='black', bbox=bbox_props)
# ax.text(c_points[3][0], c_points[3][1], c_points[3][2], '$C$', ha='center', va='center', rotation=0,
			# size=12, color='black', bbox=bbox_props)
ax.text(c_points[4][0], c_points[4][1], c_points[4][2], '$A$', ha='center', va='center', rotation=0,
			size=12, color='black', bbox=bbox_props)

# Creating the mesh grid and applying the necessary conditions
I, X, Y = np.mgrid[0:1:20j, -1:1:20j, 0:1:20j]
ii = i_prime(I, X, Y, params)
xx = x_prime(I, X, Y, params)
yy = y_prime(I, X, Y, params)

c1 = (I**2 + X**2 + Y**2) < 1
c2 = (I == 0) | (X == 0) | (Y == 0)
c = c1 & c2

ii[~c] = 0
xx[~c] = 0
yy[~c] = 0

ax.quiver(I, X, Y, ii, xx, yy, color='grey', length=0.05, alpha=0.3, arrow_length_ratio=0.2,
          normalize=True)
plt.show()

