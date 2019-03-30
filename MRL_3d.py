#! /usr/local/bin/python3.7
import matplotlib as mpl
from matplotlib import rc
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patheffects as path_effects
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
        

#plt.rc('text', usetex=True)    #In order to use LaTeX
#plt.rc('font', family='serif')  #In order to use Serif (mathced font with LaTeX)
#mpl.rcParams['font.size'] = 11

'''
params = {'backend': 'pdf',
          'axes.labelsize': 12,
          'text.fontsize': 12,
          'legend.fontsize': 12,
          'xtick.labelsize': 11,
          'ytick.labelsize': 11,
          'text.usetex': True,
          'axes.unicode_minus': True}

mpl.rcParams.update(params)
'''
# Box properties of annotation (critical points etc.)
bbox_props = dict(boxstyle='circle,pad=0.30', fc='white', ec='#980000', lw=2)
'''         
# ==================================================
# 1. Matter - Radiation - Lambda ( M, R, L) System =
# ==================================================
'''

eps = 1e-8

def Matter(m,r,l):
    mp = m * (r - 3*l)
    return mp

def Radiation(m,r,l):
    rp = r * (r - 3*l - 1)
    return rp

def Lambda(m,r,l):
    lp = l*(r - 3*l + 3)
    return lp

# 4D System for Integration
def system(X,t):
    m = X[0]
    r = X[1]
    l = X[2]
    
    
    mp = m * (r - 3*l)
    rp = r * (r - 3*l - 1)
    lp = l*(r - 3*l + 3)
    
    
    return [mp,rp,lp]

'''
# ======================
# 1.1. 3D QUIVER PLOT  =
# ======================
'''

# Object Oriented API for Matplotlib:
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1, projection='3d')

# Set colors of the background
#ax1.w_xaxis.set_pane_color((	0.25,	0.03,	0.05,   0.13))
#ax1.w_yaxis.set_pane_color((	0.25,	0.03,	0.05,   0.13))
#ax1.w_zaxis.set_pane_color((	0.25,	0.03,	0.05,   0.13))



# Labels of the axes
ax1.set_xlabel(r'$\Omega_m$', fontsize = 16)
ax1.set_ylabel(r'$\Omega_r$',fontsize = 16)
ax1.set_zlabel(r'$\Omega_\Lambda$',fontsize = 16)

# To prevent rotating labels
ax1.xaxis.set_rotate_label(False)
ax1.yaxis.set_rotate_label(False)
ax1.zaxis.set_rotate_label(False)

# Limits and ticks of the axes
ax1.set_xlim(0,1)
ax1.set_xticks([0,1])
#-----------------------
ax1.set_ylim(0,1)
ax1.set_yticks((0,1),minor=False)
#-----------------------
ax1.set_zlim(0,1)
ax1.set_zticks((0,1),minor=False)

# Hide grids
ax1.grid(False)

# Make the grid
G_m,G_r,G_l = np.mgrid[0:1:20j,0:1:20j, 0:1:20j]


# Norm:
S = np.sqrt(G_m*G_m + G_r*G_r + G_l*G_l)
S[ S== 0] = 1.

# Make the direction data for the arrows
u = Matter(G_m,G_r,G_l)
v = Radiation(G_m,G_r,G_l)
w = Lambda(G_m,G_r,G_l)

'''
L = 1 - X - Y
for i in range (len(L)):
    for j in range(len(L)):
        if L[i,j] < 0:
            L[i,j] = 0
'''


# Remove the unnecessary part of the plot
for a in range(len(G_m)):
    for b in range(len(G_m)):
        for c in range(len(G_m)):
            
            if  (G_m[a,b,c] == 0 or G_r[a,b,c] == 0 or G_l[a,b,c] == 0) and (G_m[a,b,c] +  G_r[a,b,c] +  G_l[a,b,c]  < 1.01):
                continue
            
            else:
               
                u[a,b,c] = 0
                v[a,b,c] = 0
                w[a,b,c] = 0

# Quiver Plot
qvr = ax1.quiver(G_m,G_r,G_l, u, v, w, color = 'grey',  length = 0.05,alpha= 0.3,arrow_length_ratio = 0.2,normalize = True)


'''
# ==============================================
# = 1.2. Boundaries of the Compact Phase Space =
# ==============================================
'''
#The phase space is basically a pyramid

ax1.plot([0,0],[0,0],[1,0], 'k-', lw=0.5,ls=':')
ax1.plot([1,0],[0,0],[0,0], 'k-', lw=0.5,ls=':')
ax1.plot([0,0],[1,0],[0,0], 'k-', lw=0.5,ls=':')
ax1.plot([0,1],[0,0],[1,0], 'k-', lw=1,ls='-',alpha = 0.7)
ax1.plot([0,0],[0,1],[1,0], 'k-', lw=1,ls='-',alpha = 0.7)
ax1.plot([1,0],[0,1],[0,0], 'k-', lw=1,ls='-',alpha = 0.7)

'''
# ==================
# = 1.3 Conditions =
# ==================
'''

# No existence condition:
P_M = [1, 0, 0] 
P_R = [0, 1, 0]
P_L = [0, 0, 1]
  
# Annotate them on the plot:
Annotation_CriticalPoints = [P_M,P_R,P_L]
Annotation_text_CriticalPoints = ['$M$','$R$','$\Lambda$']

text_counter = 0
for a,b,c in Annotation_CriticalPoints:
    ax1.text(a,b,c,Annotation_text_CriticalPoints[text_counter],ha='center', va='center', rotation=0,
                 size=12, bbox=bbox_props)
    
    text_counter = text_counter + 1
    
P_Current_Universe_r_m = [0.3 - (8*1e-5) , 8 * 1e-5, 0.7]
ax1.text(P_Current_Universe_r_m[0] , P_Current_Universe_r_m[1],P_Current_Universe_r_m[2], '!',ha='center', va='center', rotation=0,
                 size=12, bbox=dict(boxstyle='circle,pad=0.30', fc='white', ec='C0', lw=2))

Conditions = [[0.3,0.1,0.6],[0.3 , 0.3, 0.4],[0.3 , 0.6 , 0.1]]
#ax1.legend(loc='best', shadow=False, fontsize='small')
ax1.view_init(elev=30., azim=+50)

n = 100000
t_max = 35
t = np.linspace(0,t_max,n)

# Backward Time Interval
t_max_backward = -11.
t_backward = np.linspace(0,t_max_backward,n)

def Backward_Integration(Current_Conditions):
    
    
    z_backward = odeint(system,Current_Conditions,t_backward)
    
    m_initial = z_backward[:,0]
    r_initial = z_backward[:,1]
    l_initial = z_backward[:,2]
    
    
    print('Initial Conditions: \n --------------')
    print('m = %.15f,r = %.15f,l = %.15f'%(m_initial[n-1],r_initial[n-1],l_initial[n-1]))
    
    # i_initial + j_initial + x_initial + y_initial must be equal to 1.
    initial_conditions_total = m_initial[n-1]+ r_initial[n-1] + l_initial[n-1]
    print('Total Energy Density = %.15f'%initial_conditions_total)
    
    # Return to array of initial conditions of the current status
    return [m_initial[n-1],r_initial[n-1],l_initial[n-1]]

def Forward_Integration_MRL(Initial_Conditions,c = 'crimson'):
    
    z_f = odeint(system,Initial_Conditions,t)
    
    m_f = z_f [:,0]
    r_f = z_f [:,1]
    l_f = z_f [:,2]
    
    
    ax1.plot(m_f,r_f,l_f, color = c, lw = 2, 
        label = r'$i = %.2f, j = %.2f , x = %.2f$'%(m_f[0],r_f[0],l_f[0]))
    plt.show()
    return [m_f,r_f,l_f]

z1 = Backward_Integration(P_Current_Universe_r_m)
z1_f = Forward_Integration_MRL(z1,'C0')

# i = z1_f[0]
# j = z1_f[1] 
# x = z1_f[2] 

# T1 = 7000
# T2 = 8000

# a = Arrow3D([i[T1], i[T2]], [j[T1], j[T2]], [x[T1], x[T2]], mutation_scale=20, 
#             lw=1, arrowstyle="-|>",color = 'C0')
# ax1.add_artist(a)

# T1 = 27000
# T2 = 28000

# a = Arrow3D([i[T1], i[T2]], [j[T1], j[T2]], [x[T1], x[T2]], mutation_scale=20, 
#             lw=1, arrowstyle="-|>",color = 'C0')
# ax1.add_artist(a)

# T1 = 28000
# T2 = 30000

# a = Arrow3D([i[T1], i[T2]], [j[T1], j[T2]], [x[T1], x[T2]], mutation_scale=20, 
#             lw=1, arrowstyle="-|>",color = 'C0')
# ax1.add_artist(a)

            
z2 = Forward_Integration_MRL([0.3,0.7,0])


plt.show()



'''
lines = res.lines.get_paths()
#for l in lines:
#    plot(l.vertices.T[0],l.vertices.T[1],'k')


fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
for line in lines:
    old_x = line.vertices.T[0]
    old_y = line.vertices.T[1]
    # apply for 2d to 3d transformation here
    new_z = np.exp(-(old_x ** 2 + old_y ** 2) / 4)
    new_x = 1.2 * old_x
    new_y = 0.8 * old_y
    ax3.plot(new_x, new_y, new_z, 'k')


x1,y1 = np.linspace(0,1,200),np.linspace(0,1,200) 
X,Y = np.meshgrid(x1,y1)
U = Matter(X,Y,0)
V = Radiation(X,Y,0)
   
start = [[0,2],[.8,0],[.6,7]]
  
strm = ax1.streamplot( X,Y,U, V,linewidth=.2) 
strmS = ax1.streamplot(x1,y1, U, V, start_points=start, color="crimson", linewidth=1)
'''

