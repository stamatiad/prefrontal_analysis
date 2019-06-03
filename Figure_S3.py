#import numpy as np
#import time
## this lampda is array friendly:
#l = lambda x: np.exp(np.multiply(-0.0085,x))
## test two scenarios:
#
#tic = time.perf_counter()
#for x in range(10):
#    l(x)
#toc = time.perf_counter()
#print('non array friendly time {}'.format(toc-tic))
#
#tic = time.perf_counter()
#l(np.arange(10))
#toc = time.perf_counter()
#print('array friendly time {}'.format(toc-tic))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import notebook_module as nb

x = np.asarray([ 0.0010,    0.0010,    0.0010,    0.0010,    0.0010,    0.0010,    0.0010,    0.0010,    0.0010,    0.0010,    0.0010,    0.0010,    0.0010,6.0000, 8.6000, 12.6000, 22.6000, 22.6000, 24.7000, 24.9000, 24.9000, 44.8000, 84.5000, 188.5000 ])
y = np.asarray([0.2401,    0.4574,   0.1076,    0.2484,    0.9025,    0.4016,    0.7120,    0.3126,    0.1408,    0.2856,    0.2732,    0.4368,    0.1904 ,0.0911, 0.3643, 0.3726, 0.1201,0.1118, 0.1242, 0.2049, 0.1387, 0.1138, 0.1242, 0.0952])

def func_powerlaw(x, a, b, c):
    return c + x**b * a


popt, pcov = curve_fit(func_powerlaw, x, y, maxfev=2000 )
#sol2 = curve_fit(func_powerlaw, x, y, p0 = np.asarray([-1,10**5,0]))

# These are the fitted options by python.
#array([-7.50259257e+01,  6.42232615e-04,  7.53522217e+01])

fig, ax = plt.subplots()
#plt.figure(figsize=(10, 5))
ax.plot(x, func_powerlaw(x, *popt), '--', color='k')
ax.scatter(x, y)
nb.axis_normal_plot(axis=ax)
nb.adjust_spines(ax, ['left', 'bottom'])
ax.set_xlabel('Distance from soma (um)')
ax.set_ylabel('Conductance (nS)')
#ax.legend()
plt.show()

fig.savefig('Figure_S3.pdf')
fig.savefig('Figure_S3.svg')
fig.savefig('Figure_S3.png')

print('Tutto pronto!')