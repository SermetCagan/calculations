from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sympy as sm
from matplotlib2tikz import save as tikz_save

plt.rc('text', usetex=True)  
plt.rc('font', family='serif')
mpl.rcParams['font.size'] = 11

def Matter(m,r):
    mp = m * (3 * m + 4 * r - 3)
    return mp

def Radiation(m, r):
    rp = r * (3 * m + 4 * r - 4)
    return  rp

Y, X = np.mgrid[0:1:201j, 0:1:201j]

U = X * (3 * X + 4 * Y -3)
V = Y * (3 * X + 4 * Y -4)
speed = np.sqrt(U*U + V*V)
speed[ speed== 0] = 1.

for i in range(len(U)):
    for j in range(len(V)):
        if(0 < X[i,j] + Y[i,j] < 1.):
            U[i, j] = U[i, j]
            V[i, j] = V[i,j]

        else:
            U[i, j] = 0
            V[i, j] = 0
            
L = 1 - X - Y
for i in range (len(L)):
    for j in range(len(L)):
        if(L[i,j] < 0):
            L[i,j] = 0

fig0, ax0 = plt.subplots(figsize = (5.3,4.5))
strm = ax0.streamplot(X, Y, U, V, color= L ,linewidth=1.3, cmap=plt.cm.inferno)
CLB = fig0.colorbar(strm.lines)
CLB.ax.set_title('$\Omega_\Lambda$')

x1,y1 = np.linspace(0,1,201),np.linspace(0,1,201) 
   
start = [[0,1]]
  
strmS = ax0.streamplot(x1, y1, U, V, start_points=[[0,1],[0,1]], color="crimson", linewidth=2) 

def line1(m):
    r = 3./4 - (3./4)*m
    return r

def line2(m):
    r = 1 - (3./4)*m
    return r

my_m = np.linspace(0,1,20)
my_r1 = line1(my_m)
my_r2 = line2(my_m) 

ax0.plot(my_m, my_r1, color = 'navy', linewidth = 3)
ax0.plot(my_m, my_r2, color = 'red', linewidth = 3)

plt.show()

bbox_props = dict(boxstyle='circle,pad=0.15', fc='white', ec='#980000', lw=1.5) 

T11 = ax0.text(0, 0, '$\Lambda$', ha='center', va='center', rotation=0,
            size=12,color = 'black',
            bbox=bbox_props)
T22 = ax0.text(1, 0, 'M', ha='center', va='center', rotation=0,
            size=12,color = 'black',
            bbox=bbox_props)
T33 = ax0.text(0, 1, 'R', ha='center', va='center', rotation=0,
            size=12,color = 'black',
            bbox=bbox_props)

ax0.text(0.3, 8*1e-5, '!', ha='center', va='center', rotation=0,
            size=12,color = 'black',
            bbox=dict(boxstyle='circle,pad=0.15', fc='white', ec='navy', lw=1.5))
ax0.text(0.3, -0.06, 'Current Status', ha='center', va='center', rotation=0,
            size=11,color = 'black')

ax0.set_xlim([-0.1, 1.1])
ax0.set_ylim([-0.1, 1.1])

ax0.set_xticks((0,0.3,2/3,1),minor=False)
ax0.set_xticklabels([0,0.3,'2/3',1])

ax0.set_yticks((0,0.5,1),minor=False)
ax0.set_yticklabels([0,'1/2',1])

#Borders:
ax0.plot([0,1],[1,0], color ='dimgrey', lw=1.5)
ax0.plot([0,0],[0,1], color ='dimgrey', lw=1.5)
ax0.plot([1,0],[0,0], color ='dimgrey', lw=1.5)

ax0.set_xlabel('$\Omega_m$',size = 15)
ax0.set_ylabel('$\Omega_r$',size = 15)

def line(x):
    line = (-3*x + 2)/4
    return line

def zero(x):
    line = 0 * x
    return line

#lx = np.linspace(0,1,100)
#ax0.fill_between(lx,line(lx),zero(lx),where=line(lx)>zero(lx),facecolor = 'grey', alpha = 0.2,edgecolor = 'blue' ,interpolate=True)
#ax0.scatter(2./3,0,c = 'dimgrey',s = 12)
#ax0.scatter(0,0.5,c = 'dimgrey',s = 12)
#plt.tight_layout()
#
#fig1, ax1 = plt.subplots(figsize = (5.3,4.5))
#ax1.set_xlim([-0.1, 1.1])
#ax1.set_ylim([-0.1, 1.1])
#
#n = 100000
#eps = 1e-5
#
#def MRLsolver(m_initial,r_initial,t_final = 30):
#    t_initial = 0
#    #m_initial = 0.1 #Will not be required since the function takes IVs as inputs
#    #r_initial = 0.8 #Will not be required since the function takes IVs as inputs
#
#    #t_final = 30
#    num_points = n
#    h = (t_final - t_initial)/num_points
#
#    # Define Vectors:
#    t_values = np.zeros(num_points)
#    m_values = np.zeros(num_points)
#    r_values = np.zeros(num_points)
#
#    t_values[0] = t_initial
#    m_values[0] = m_initial
#    r_values[0] = r_initial
#
#    #   ***************************
#    #   * 4th Order Runge - Kutta *
#    #   ***************************
#
#    for i in range(num_points-1):
#        t = t_values[i]
#        m = m_values[i]
#        r = r_values[i]
#
#        #F1:
#        F1m = Matter(m,r)
#        F1r = Radiation(m,r)
#
#        #F2:
#        F2m = Matter(   m + (h/2) * F1m, r + (h/2) * F1r    )
#        F2r = Radiation(m + (h/2) * F1m, r + (h/2) * F1r    )
#
#        #F3:
#        F3m = Matter(   m + (h / 2) * F2m, r + (h / 2) * F2r)
#        F3r = Radiation(m + (h / 2) * F2m, r + (h / 2) * F2r)
#
#        #F4:
#        F4m = Matter(   m + h * F3m      , r + h * F3r     )
#        F4r = Radiation(m + h * F3m      , r + h * F3r     )
#
#        # Final Version of the vectors:
#        t_values[i+1] = t_values[i] + h
#        m_values[i+1] = m_values[i] + (h/6) * (F1m + 2*F2m + 2*F3m + F4m)
#        r_values[i+1] = r_values[i] + (h/6) * (F1r + 2*F2r + 2*F3r + F4r)
#
#    return m_values, r_values #, t_values
#
#m3,r3 = MRLsolver(0.3,8*eps,-13)
#m3 = m3[::-1]
#r3 = r3[::-1]
#
##ax1.plot(m3,r3, ls = ':')
#
#m4,r4 = MRLsolver(0.3,8*eps,13)
##ax1.plot(m4,r4,c = 'crimson')
#
#m5 = np.concatenate([m3,m4])
#r5 = np.concatenate([r3,r4])
#
#ax1.plot(m5,r5)
#
#
#
#def Cosmological_Parameters(m,r):
#    # Cosmological Parameters Plot:
#    fig2 = plt.figure(figsize=(6,10./3))
#    ax2 = fig2.add_subplot(1,1,1)
#  
#    t_max = 15
#  
#    plt.hlines(0,0,t_max,color = 'black',linestyle = ':',linewidth = 0.5)
#    plt.hlines(-1,0,t_max,color = 'blue',linestyle = ':',linewidth = 0.5)
#    plt.hlines(-1./3,0,t_max,color = 'blue',linestyle = ':',linewidth = 0.5)
#   # plt.hlines(-0.67,0,t_max,color = 'blue',linestyle = '-.',linewidth = 0.5)
#  
# 
#    w_eff = r*(1./3) + (1-r-m)*(-1)
#  
#    Omega_lambda = 1-m-r
#    t = np.linspace(0,t_max,len(m)) 
#  
#    ax2.plot(t,m, color = 'crimson', ls = '--' ,lw=2)
#    ax2.plot(t,r, color = 'darkgreen', ls = '-',lw=2)
#    ax2.plot(t,Omega_lambda, color = 'C0', ls = '--',lw=2)
#    #ax2.plot(t,w_p,color = 'orangered',ls = '-',lw = 1.5)
#    ax2.plot(t,w_eff,color = 'navy', ls = '-',lw = 1.5)
#  
#    #ox, = ax2.plot(t,Omega_x,color = 'goldenrod',ls ='-.' )
#    #oy, = ax2.plot(t,Omega_y, color = 'deepskyblue',ls ='-.' )
#  
#  
#    plt.legend(['$\Omega_m$','$\Omega_r$','$\Omega_\Lambda$','$\omega_{eff}$'],
#                bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
#  
#  
#    ax2.set_xlabel('Time')
#    ax2.set_xticks([])
#    ax2.set_yticks((1,0.5,0,-1./3,-1),minor=False)
#    ax2.set_yticklabels([1, 0.5, 0,'-1/3',-1])
#    ax2.set_xlim(0,10)
#    current_time_index = 0
#  
#  
#    for x in range(0,len(m)):
#        if 0.3 - 0.001 <=  m[x] <= 0.3 + 0.001:
#            current_time_index = x
#            exit
#          
#    plt.vlines(t[current_time_index],-1,1,color = 'black',linestyle = '--',linewidth = 1)
#    plt.tight_layout()
#
#Cosmological_Parameters(m5,r5)
