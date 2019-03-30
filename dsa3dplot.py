#! /usr/local/bin/python3.7

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from scipy.integrate import odeint
import matplotlib.patheffects as path_effects
from matplotlib.patches import FancyArrowPatch

class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def Matter(m, r, l):
    mp = m * (r - 3*l)
    return mp

def Radiation(m, r, l):
    rp = r * (r - 3*l - 1)
    return rp

def Lambda(m, r, l):
    lp = l * (r - 3*l + 3)
    return lp

def system(init_vals, t):
	m = init_vals[0]
	r = init_vals[1]
	l = init_vals[2]

	mp = Matter(m, r, l)
	rp = Radiation(m, r, l)
	lp = Lambda(m, r, l)

	return [mp, rp, lp]

def backward_integration(f, init_values, n, t_max):
	t = np.linspace(0, t_max, n)
	bSolution = odeint(f, init_values, t)
	m_sol = bSolution[:,0]
	r_sol = bSolution[:,1]
	l_sol = bSolution[:,2]

	print('Initial Conditions:\n')
	print('Matter = %.15f, Radiation = %.15f, Lambda = %.15f'%(m_sol[-1], r_sol[-1], l_sol[-1]))

	total_init = m_sol[-1] + r_sol[-1] + l_sol[-1]
	print('Total = %.15f'%(total_init))

	return [m_sol[-1], r_sol[-1], l_sol[-1]]

def forward_integration(f, init_values, n, t_max):
	t = np.linspace(0, t_max, n)
	fSolution = odeint(f, init_values, t)
	m_sol = fSolution[:,0]
	r_sol = fSolution[:,1]
	l_sol = fSolution[:,2]

	return [m_sol, r_sol, l_sol]

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')

bbox_props = dict(boxstyle='circle,pad=0.30', fc='white', ec='#980000', lw=2)

ax.set_xlabel(r'$\Omega_{m}$', fontsize=16)
ax.set_ylabel(r'$\Omega_{r}$', fontsize=16)
ax.set_zlabel(r'$\Omega_{\Lambda}$', fontsize=16)

ax.text(0, 0, 1, '$\Lambda$', ha='center', va='center', rotation=0,
			size=12, color='black', bbox=bbox_props)
ax.text(0, 1, 0, 'R', ha='center', va='center', rotation=0,
			size=12, color='black', bbox=bbox_props)
ax.text(1, 0, 0, 'M', ha='center', va='center', rotation=0,
			size=12, color='black', bbox=bbox_props)

n = 100000

P_Current_Universe_r_m = [0.3 - (8*1e-5) , 8 * 1e-5, 0.7]
ax.text(P_Current_Universe_r_m[0] , P_Current_Universe_r_m[1],P_Current_Universe_r_m[2], '!',ha='center', va='center', rotation=0,
                 size=12, bbox=dict(boxstyle='circle,pad=0.30', fc='white', ec='C0', lw=2))

b_sol = backward_integration(system, P_Current_Universe_r_m, n, -11)
f_sol = forward_integration(system, b_sol, n, 35)

ax.plot(f_sol[0], f_sol[1], f_sol[2], color='C0', lw=2)

i, j, k = f_sol[0], f_sol[1], f_sol[2]

T1 = 7000
T2 = 8000

a = Arrow3D([i[T1], i[T2]], [j[T1], j[T2]], [k[T1], k[T2]], mutation_scale=20, 
            lw=1, arrowstyle="-|>",color = 'C0')
ax.add_artist(a)

T1 = 27000
T2 = 28000

a = Arrow3D([i[T1], i[T2]], [j[T1], j[T2]], [k[T1], k[T2]], mutation_scale=20, 
            lw=1, arrowstyle="-|>",color = 'C0')
ax.add_artist(a)

T1 = 28000
T2 = 30000

a = Arrow3D([i[T1], i[T2]], [j[T1], j[T2]], [k[T1], k[T2]], mutation_scale=20, 
            lw=1, arrowstyle="-|>",color = 'C0')
ax.add_artist(a)

ax.xaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False)
ax.zaxis.set_rotate_label(False)

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)

ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_zticks([0,1])

ax.view_init(elev=30., azim=+50)
ax.grid(False)

x, y, z = np.mgrid[0:1:20j, 0:1:20j, 0:1:20j]

u = Matter(x,y,z)
v = Radiation(x,y,z)
w = Lambda(x,y,z)

c1 = np.add(np.add(x,y),z) < 1
c2 = (x == 0) | (y == 0) | (z == 0)
c = c1 & c2

u[~c] = 0
v[~c] = 0
w[~c] = 0

ax.plot([1,0,0,1,0,0,0,0],[0,1,0,0,0,0,0,1],[0,0,1,0,0,1,0,0], 'k-', lw=0.5, ls=':')

ax.quiver(x, y, z, u, v, w, color='grey', length=0.05, alpha=0.3, arrow_length_ratio=0.2,
          normalize=True)

plt.show()