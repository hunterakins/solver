"""
Description:
Test odeint on an oscillating ODE
d2f/dt2 = -f  
f0 = 1
dfdt0 = 0

y = [f, dfdt]
y0 = [1, 0]

dydt = [dfdt, -f] = [y[1], -y[0]]

Solution is cos(t)


Date:
7/11/20

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


from ode_solver import odeint

# f_dydx for osc.
import numba as nb
@nb.njit
def f_dydx(x, y):
    dydx = np.zeros(2)
    dydx[0] = y[1]
    dydx[1] = -y[0]
    return dydx


ystart = np.array([1, 0])
n = 2
x1 = 0.
x2 = 300 * np.pi
eps = 1e-14
yscale = np.zeros(n)
h1 = .01
hmin = 1e-10
max_steps = int(1e6)
kmax = int(1e3)
dx_sav = 1.1*(x2 - x1)/kmax


y, x_out, y_out, nok, nbad = odeint(ystart, f_dydx, n, x1, x2, eps, yscale, h1, hmin, max_steps, kmax, dx_sav)

print(x_out)
plt.figure()
plt.plot(x_out, y_out[0,:] -np.cos(x_out))
plt.show()



    
