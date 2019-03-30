import matplotlib.pyplot as plt
import numpy as np
from math import *
from sympy import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
'''
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
'''
# **************** Autonomous System **************** 

a = 2

def f(x,y,z):
    return 3.*x*x + 4.*x*y + 6.*x*z - 3.*x 
def g(x,y,z):
    return 4.*y*y + 3.*y*x + 6.*y*z - 4.*y
def h(x,y,z):
    return 6.*z*z + 3.*x*z + 4.*y*z - 6.*z + a * (1 - x - y - z) * np.sqrt(6*z)
    

# ************ Lists - p : past, f : future *************
x = []
y = []
z = []

xp = []
yp = []
zp = []

xf = []
yf = []
zf = []

time = []


#iv1, iv2 = initial values, dt = timestep, ptime = past range, ftime = future range

def sys(iv1, iv2, iv3, dt, ptime, ftime):
    xp.append(iv1)
    yp.append(iv2)
    zp.append(iv3)
    xf.append(iv1)
    yf.append(iv2)
    zf.append(iv3)
    for i in range(ptime):
        xp.append(xp[i] - (f(xp[i],yp[i],zp[i])) * dt)
        yp.append(yp[i] - (g(xp[i],yp[i],zp[i])) * dt)
        zp.append(zp[i] - (h(xp[i],yp[i],zp[i])) * dt)
    for i in range(ftime):
        xf.append(xf[i] + (f(xf[i],yf[i],zf[i])) * dt)
        yf.append(yf[i] + (g(xf[i],yf[i],zf[i])) * dt)
        zf.append(zf[i] + (h(xf[i],yf[i],zf[i])) * dt)
    return xp, yp, zp, xf, yf, zf,



# **************** Plot ****************

fig = plt.figure(figsize=(7,7))

ax = fig.add_subplot(1,1,1, projection='3d')
ax.grid(False)

ax.set_xlabel(r'$\Omega_m$', size=30)
ax.set_xticks([0,1])
ax.set_xticklabels(['$0$','$1$'])
ax.xaxis.set_rotate_label(False)
ax.set_ylabel(r'$\Omega_r$', size=30)
ax.set_yticks([0,1])
ax.set_yticklabels(['$0$','$1$'])
ax.yaxis.set_rotate_label(False)
ax.set_zlabel(r'$\Omega_k$', size=30)
ax.set_zticks([0,1])
ax.set_zticklabels(['$0$','$1$'])
ax.zaxis.set_rotate_label(False)
ax.tick_params(axis='both', which='major', labelsize=24)

ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_zlim([0,1])


tmp_planes = ax.zaxis._PLANES 
ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                     tmp_planes[0], tmp_planes[1], 
                     tmp_planes[4], tmp_planes[5])
view_1 = (20, 60)
view_2 = (25, -45)
init_view = view_1
ax.view_init(*init_view)


ax.plot([1,0],[0,0], 'k-', lw=.3)
ax.plot([0,0],[0,1], 'k-', lw=.3)
ax.plot([1,0],[0,1], 'k-', lw=.3)
ax.plot([0,0],[0,0],[0,1], 'k-', lw=.3)
ax.plot([1,0,0],[0,0,1],[0,1,0], 'k-', lw=.3)

plt.tight_layout()


# **************** Vectors **************** 

u, v, w = np.meshgrid(np.arange(0, 1, .1), np.arange(0, 1, .1), np.arange(0, 1, .1))

N = np.sqrt(f(u,v,w)**2+g(u,v,w)**2+h(u,v,w)**2)

for m in range(0,len(u)):
    for n in range(0,len(v)):
        for k in range(0,len(w)):
            if (u[m,n,k] == 0 or v[m,n,k] == 0 or w[m,n,k] == 0) and (u[m,n,k] + v[m,n,k] + w[m,n,k] <= 1) :
                continue
            else:
                u[m,n,k] = 0
                v[m,n,k] = 0
                w[m,n,k] = 0
                
ax.quiver(u, v, w, f(u,v,w)/N, g(u,v,w)/N, h(u,v,w)/N, color = '#1f77b4', length = 0.05, alpha=0.25)


# ************* Today's Densities *************
#today = [0.3089, 0.0000916, 0.245084]

today = [0.3089, 0.0000916, 0.2450832645]


# ************* Solution Curves **************
#sys(today[0], today[1], today[2], 0.00005, 250000, 100000)

sys(today[0], today[1], today[2], 0.00005, 370000, 120000)

x = xp[::-1] + xf
y = yp[::-1] + yf
z = zp[::-1] + zf


ax.plot(x, y, z, color="#1f77b4", lw=3)

time = np.linspace(0,100,len(x))


# ************ Critical Points *************

cpx = [0,1,0,0,0,1./4.]
cpy = [0,0,1,0,0,0]
cpz = [0,0,0,1,2./3.,3./8.]

ax.plot(cpx,cpy,cpz,'ro',markersize=8, markeredgecolor='k')


# ************ Significant Points ************
spx = []
spy = []
spz = []
spk = []

mreqx = []
mreqy = []
mreqz = []

mseqx = []
mseqy = []
mseqz = []

mqeqx = []
mqeqy = []
mqeqz = []

accx = []
accy = []
accz = []

eos = []
u =[]
eoseff = []

print('--------------------------------------')

for k in range(0,len(x)):
    eos.append(-1-((2*z[k])/(x[k] + y[k] -1)))
    u.append(1 - x[k] - y[k])
    eoseff.append(x[k]+(4./3)*y[k]+2*z[k]-1)
    
for k in range(0,len(x)):
    if len(mseqx) == 0 :
        if round(x[k],4) == round(z[k],4):
            mseqx.append(x[k])
            mseqy.append(y[k])
            mseqz.append(z[k])
            spk.append(k)
            print('Matter - Kinetic Equality')
            print(k, round(x[k],4), round(z[k],4))
            print('--------------------------------------')
            k1 = k
            break
            
for k in range(k1,len(x)):
    if len(mreqx) == 0 :
        if round(x[k],4) == round(y[k],4):
            mreqx.append(x[k])
            mreqy.append(y[k])
            mreqz.append(z[k])
            spk.append(k)
            print('Matter - Radiation Equality')
            print(k, round(x[k],4), round(y[k],4))
            print('--------------------------------------')
            k1 = k
            break
      
for k in range(k1,len(x)):
    if len(mqeqx) == 0 :
        if (2*round(x[k],3) + round(y[k],3) == 1):
            mqeqx.append(x[k])
            mqeqy.append(y[k])
            mqeqz.append(z[k])
            spk.append(k)
            print('Matter - Quintessence Equality')    
            print(k, round(x[k],4), round(u[k],4), round(eoseff[k],4))
            print('--------------------------------------')
            k1 = k
            break

for k in range(k1,len(x)):
    if len(spk) == 3 :
        if (x[k] == today[0]) :
            spk.append(k)
            print('Today')
            print(k, round(x[k],4), round(u[k],4), round(eos[k],4), round(eoseff[k],4))
            print('--------------------------------------')
            break


spx = [mseqx[0], mreqx[0], mqeqx[0], today[0]]
spy = [mseqy[0], mreqy[0], mqeqy[0], today[1]]
spz = [mseqz[0], mreqz[0], mqeqz[0], today[2]]

ax.plot(spx,spy,spz,'o',markersize=8, color='#1f77b4', markeredgecolor='k')

roman = ['I', 'II', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']

ax.text(spx[0]+0.08, spy[0]+0.01, spz[0]+0.05, roman[0], fontsize=21,
                color='black')
ax.text(spx[1]+0.03, spy[1]+0.02, spz[1]+0.05, roman[1], fontsize=21,
                color='black')
ax.text(spx[2]+0.07, spy[2]+0.02, spz[2]+0.05, roman[2], fontsize=21,
                color='black')
ax.text(spx[3]+0.07, spy[3]+0.02, spz[3]+0.05, roman[3], fontsize=21,
                color='black')


# ********** Plot - EoS and Densities **********

fig,ax = plt.subplots(2, figsize = (8,5), sharex=True, gridspec_kw = {'height_ratios':[1, 1]})
fig.subplots_adjust(left=0.1, bottom=0.15, right=0.83, top=0.95, wspace=None, hspace=0.03)


ax[0].grid(False)
ax[0].set_xlim([0,100])
ax[0].set_ylabel('')
ax[0].set_ylim([-1.1,1.1])
ax[0].set_yticks([-1,0,1./3,1])
ax[0].set_yticklabels(['$-1$','$0$','$1/3$','$1$'])
#ax[0].yaxis.set_label_coords(-0.1, .45)
ax[0].tick_params(axis='both', which='major', labelsize=20)
for k in range(0,len(spk)) :
    ax[0].plot([time[spk[k]],time[spk[k]]],[-1.2,1.2], 'k-', lw=1)
ax[0].plot([time[0],time[len(time)-1]],[0,0], 'k--', lw=1)
ax[0].plot([time[0],time[len(time)-1]],[1./3,1./3], 'k--', lw=1)
ax[0].plot(time, eos, color='#1f77b4', lw=2.5, label=r'$\omega_\phi$')
ax[0].plot(time, eoseff, color='#ff7f0e', lw=2.5, label=r'$\omega_{eff}$')
leg1 = ax[0].legend(loc = 'upper right', bbox_to_anchor = (1.252, 1.095), 
            ncol=1, handlelength=1, handleheight=2,  handletextpad=.25, borderpad=0.2, 
            labelspacing=-0.4, prop={'size':26})

leg1.get_frame().set_edgecolor('k')

omegatoday1 = "$" + str(round(x[len(time)-1],2)) + "$"
omegatoday2 = "$" + str(round(u[len(time)-1],2)) + "$"


ax[1].grid(False)
ax[1].set_xlim([0,100])
ax[1].set_xlabel(r'$Time$', size=20)
ax[1].set_xticks([])
ax[1].xaxis.set_label_coords(0.5, -0.1)
ax[1].set_ylabel('')
ax[1].set_ylim([-0.05,1.05])
ax[1].set_yticks([0,x[len(time)-1],u[len(time)-1],1])
ax[1].set_yticklabels(['$0$',omegatoday1,omegatoday2,'$1$'])
#ax[1].yaxis.set_label_coords(-0.1, .4)
ax[1].tick_params(axis='both', which='major', labelsize=20)
ax[1].plot([time[0],time[len(time)-1]],[x[len(time)-1],x[len(time)-1]], 'k--', lw=1)
ax[1].plot([time[0],time[len(time)-1]],[u[len(time)-1],u[len(time)-1]], 'k--', lw=1)
for k in range(0,len(spk)) :
    ax[1].plot([time[spk[k]],time[spk[k]]],[-1.2,1.2], 'k-', lw=1)
ax[1].plot(time, x, color='#d62728', lw=2.5, label=r'$\Omega_m$')
ax[1].plot(time, y, color='#1f77b4', lw=2.5, label=r'$\Omega_r$')
ax[1].plot(time, u, color='#2ca02c', lw=2.5, label=r'$\Omega_\phi$')
leg = ax[1].legend(loc = 'upper right', bbox_to_anchor = (1.24, 1.095), 
            ncol=1, handlelength=1, handleheight=2, handletextpad=.25, borderpad=0.3, 
            labelspacing=-0.4, prop={'size':26})

leg.get_frame().set_edgecolor('k')



plt.tight_layout()

