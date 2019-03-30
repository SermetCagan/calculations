import matplotlib.pyplot as plt
import numpy as np

a = np.sqrt(4.)

def f(x,y):
    return -3.*x + a*(np.sqrt(3./2))*y*y + (3./2)*x*(1. + x*x - y*y)
def g(x,y):
    return -a*(np.sqrt(3./2))*x*y + (3./2)*y*(1. + x*x - y*y)

# initialize lists containing values
x = []
y = []

xp = []
yp = []

xf = []
yf = []

time = []


#iv1, iv2 = initial values, dt = timestep, ptime = past range, ftime = future range

def sys(iv1, iv2, dt, ptime, ftime):
    xp.append(iv1)
    yp.append(iv2)
    xf.append(iv1)
    yf.append(iv2)
    for i in range(ptime):
        xp.append(xp[i] - (f(xp[i],yp[i])) * dt)
        yp.append(yp[i] - (g(xp[i],yp[i])) * dt)
    for i in range(ftime):
        xf.append(xf[i] + (f(xf[i],yf[i])) * dt)
        yf.append(yf[i] + (g(xf[i],yf[i])) * dt)
    return xp, yp, xf, yf



#plot
fig = plt.figure(figsize=(8,4.9))
fig.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.95, wspace=None, hspace=0.03)
ax = fig.add_subplot(1,1,1)

ax.set_xlabel(r'$x$', size=30)
ax.set_xticks([-1,0,1]) 
ax.set_xticklabels(['$-1$','$0$','$1$'])
ax.xaxis.set_label_coords(0.5, -0.13)
ax.set_ylabel(r'$y$', size=30, rotation=0)
ax.set_yticks([0,1]) 
ax.set_yticklabels(['$0$','$1$'])
ax.yaxis.set_label_coords(-0.06, 0.475)
ax.tick_params(axis='both', which='major', labelsize=26)

ax.set_xlim([-1.1,1.1])
ax.set_ylim([-0.1,1.1])

ax.plot([-1,1],[0,0], 'k-', lw=1.5, alpha=0.7)


crc = np.linspace(-np.pi/2, np.pi/2, 100)
crc1 = np.linspace(-np.pi/13, np.pi/13, 100)

ax.plot(np.sin(crc), np.cos(crc),color='k', linestyle = 'solid', lw=1.5, alpha=0.7)

ax.plot(np.sqrt(0.65)*np.sin(crc), np.sqrt(0.65)*np.cos(crc), color='r', alpha=0.6, linestyle = 'dashed', lw=2)
ax.plot(np.sqrt(0.7)*np.sin(crc1), np.sqrt(0.7)*np.cos(crc1), color='r', alpha=0.5, linestyle = 'solid', lw=13)
ax.plot(np.sqrt(0.75)*np.sin(crc), np.sqrt(0.75)*np.cos(crc), color='r', alpha=0.6, linestyle = 'dashed', lw=2)
ax.plot(np.sqrt(0.5)*np.sin(crc), np.sqrt(0.5)*np.cos(crc), color='g', alpha=0.7, linestyle = 'dashed', lw=2)

xx = np.arange(0, 0.275, 0.005)

ax.plot(xx, np.sqrt(37./3)*xx, color='r', linestyle = 'dashed', alpha=0.6, lw=2)
ax.plot(-xx, np.sqrt(37./3)*xx, color='r', linestyle = 'dashed', alpha=0.6, lw=2)


u, v = np.meshgrid(np.arange(-1.2, 1.2, .1), np.arange(-1.2, 1.2, .1))

for m in range(len(u)):
    for n in range(len(u)):
        if (u[m,n]*u[m,n] + v[m,n]*v[m,n] > 0.95) or (v[m,n] < 0) :
            u[m,n] = 0
            v[m,n] = 0
        else:
            continue


N = np.sqrt(f(u,v)**2+g(u,v)**2)

#ax.quiver(u, v, f(u,v)/N, g(u,v)/N, color = 'lightgrey', scale=35, headwidth=4)


# ***********************************************************

sys(np.sqrt(39./800), np.sqrt(1443./2400), 0.01, 1000, 1000)

x = xp[::-1] + xf
y = yp[::-1] + yf

ax.plot(x, y, color="#1f77b4", lw=3)


x = []
y = []

xp = []
yp = []

xf = []
yf = []


# ***********************************************************

sys(-np.sqrt(9./160), np.sqrt(333./480), 0.01, 1000, 1000)

x = xp[::-1] + xf
y = yp[::-1] + yf

ax.plot(x, y, color="#1f77b4", lw=3)


x = []
y = []

xp = []
yp = []

xf = []
yf = []

# ***********************************************************

c1 = [a/(np.sqrt(6.)), np.sqrt(1. - (a*a)/(6.))]
c2 = [np.sqrt(3./2)/a, np.sqrt(3./(2.*(a*a)))]

cpx = [0,1,-1, c1[0], c2[0]]
cpy = [0,0,0, c1[1], c2[1]]

ax.plot(cpx,cpy,'ro', markersize=8, markeredgecolor='k')

plt.tight_layout()