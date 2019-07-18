import numpy as np
import sys, os
import getdist.plots as gplot
from matplotlib import pyplot as plt

xi_ = np.linspace(-0.002,0.005,1000)

f = open("/Users/sermetcagan/github/calculations/60efolding_values.txt",'r')
lines = f.readlines()
x60 = []
y60 = []
x50 = []
y50 = []

x60 = [float([lines[i].strip("\n").split(" ") for i in range(len(lines))][j][0]) for j in range(len(lines))]
y60 = [float([lines[i].strip("\n").split(" ") for i in range(len(lines))][j][1]) for j in range(len(lines))]
f.close()

f = open("/Users/sermetcagan/github/calculations/50efolding_values.txt",'r')
lines = f.readlines()
x50 = [float([lines[i].strip("\n").split(" ") for i in range(len(lines))][j][0]) for j in range(len(lines))]
y50 = [float([lines[i].strip("\n").split(" ") for i in range(len(lines))][j][1]) for j in range(len(lines))]
f.close()

# %matplotlib qt

# g = gplot.getSinglePlotter(chain_dir=[r'/Volumes/Newton/planck/2018/COM_CosmoParams_fullGrid_R3.00/base_r/plikHM_TT_lowl_lowE',
#                                       r'/Volumes/Newton/planck/2018/COM_CosmoParams_fullGrid_R3.00/base_r/plikHM_TTTEEE_lowl_lowE',
#                                       r'/Volumes/Newton/planck/2018/COM_CosmoParams_fullGrid_R3.00/base_r/CamSpecHM_TTTEEE_lowl_lowE_lensing',
#                                       r'/Volumes/Newton/planck/2015/COM_CosmoParams_fullGrid_R2.00/base_r/plikHM_TT_lowTEB'])
# g.settings.legend_frame = False
# g.settings.legend_loc = 'best'
# g.settings.figure_legend_frame = False

# roots = []
# roots.append('base_r_plikHM_TT_lowl_lowE')
# roots.append('base_r_plikHM_TT_lowl_lowE_post_BAO')
# roots.append('base_r_plikHM_TTTEEE_lowl_lowE')
# roots.append('base_r_CamSpecHM_TTTEEE_lowl_lowE_lensing')
# roots.append('base_r_plikHM_TT_lowTEB')
# roots.append('base_r_plikHM_TT_lowTEB_post_BAO')
# pairs = [('ns','r')]

# g.plots_2d(roots, param_pairs=pairs, legend_labels=[], filled=True, shaded=False)
# g.add_line([0.96694214876], [0.132231404959], label=['N = 60'], ls='None', zorder=3, color='red', marker='o', markeredgewidth=7)
# g.add_line([0.960396039604], [0.158415841584], label=['N = 50'], ls='None', zorder=3, color='red', marker='o', markeredgewidth=2)

# leg1 = g.add_legend(['Planck TT','Planck TT + BAO','Planck TTTEEE','CamSpec TTTEEE + Lensing','Planck 2015 TT', 'Planck 2015 TT + BAO'], colored_text=True, fontsize=13, legend_loc='upper left', figure=False)
# leg2 = g.subplots[0,0].legend(['N = 60', 'N = 50'], loc='upper right', frameon=False)
# g.subplots[0,0].add_artist(leg1)

plt.scatter(x60, y60, c=xi_, s=1, cmap = 'inferno', vmin=xi_.min(), vmax=xi_.max())
plt.scatter(x50, y50, c=xi_, s=1, cmap = 'inferno', vmin=xi_.min(), vmax=xi_.max())

plt.colorbar(label = r'$\xi$')
plt.xlim(0.94,0.98)
plt.ylim(0.00,0.20)

plt.show()

#g.export('nm.pdf',adir='coupling/150205193/figures/')