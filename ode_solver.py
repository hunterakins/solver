"""
Description:
    Implementation of the Numerical Recipes Second Edition
    adaptive stepsize Runge-Kutta scheme 
    (section 16.2, pg. 719)
    in Python using numba for just-in-time compilation

Date:
    7/11/2023

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


def rkck(y, dydx, f_dydx, n, x, h):
    """
    Runge-Kutta Cash-Karp method to increment the output y by one stepsize
    from x to x+h
    Input 
    y - np 1d array 
        value of the multi-valued function at scalar value x
    dydx - np 1d array
        value of derivative dydx at scalar value x
    f_dydx - function(float, np array)
        the derivative of y with respect to x
        takes as input a point x, and a value of y
    n - int
        dimension of y
    x - float
        scalar value of x at beginning of interval
    h- float
        step size to increment x 
    Output
    yout - np 1d array
        value of function y at (x+h)
    yerr - np 1d array
        error estimate of y at (x+h)
    """

    """
    Cash-Karp coefficients
    """
    a2, a3, a4, a5, a6 = 0.2, 0.3, 0.6, 1.0, 0.875
    b21, b31, b32, b41, b42, b43, b51, b52, b53, b54 = 0.2, 3./40., 9./40., 0.3, -0.9, 1.2, -11./54., 2.5, -70./27., 35./27.
    b61, b62, b63, b64, b65 = 1631./55296., 175./512., 575./13824., 44275./110592., 253./4096.
    c1, c3, c4, c6 = 37./378., 250./621., 125./594., 512./1771.
    dc1, dc3, dc4, dc5, dc6 = c1-2825./27648., c3-18575./48384., c4-13525./55296., -277./14336., c6-0.25

    """
    store the derivative values at the various intermediate points in the algorithm
    """
    ak2 = np.zeros(n)
    ak3 = np.zeros(n)
    ak4 = np.zeros(n)
    ak5 = np.zeros(n)
    ak6 = np.zeros(n)

    """
    Take steps
    """
    ytmp = y + b21*h*dydx  # first step
    ak2 = f_dydx(x + a2*h, ytmp)
    ytmp = y + h*(b31*dydx + b32*ak2)  # second step
    ak3 = f_dydx(x + a3*h, ytmp)
    ytmp = y + h*(b41*dydx + b42*ak2 + b43*ak3)  # third step
    ak4 = f_dydx(x + a4*h, ytmp)
    ytmp = y + h*(b51*dydx + b52*ak2 + b53*ak3 + b54*ak4)  # fourth step
    ak5 = f_dydx(x + a5*h, ytmp)
    ytmp = y + h*(b61*dydx + b62*ak2 + b63*ak3 + b64*ak4 + b65*ak5)  # fifth step
    ak6 = f_dydx(x + a6*h, ytmp)

    """
    Get update y value
    """

    yout = y + h*(c1*dydx + c3*ak3 + c4*ak4 + c6*ak6)

    """
    Get error estimate as the difference between the fourth and fifth order method 
    (using the hardcoded coefficients)
    """
    yerr = h*(dc1*dydx + dc3*ak3 + dc4*ak4 + dc5*ak5 + dc6*ak6)

    return yout, yerr

def rkqs(y, dydx, f_dydx, n, x, htry, eps, yscal):
    """
    ``Runge-Kutta Quality-controlled Step ''
    Take a single quality-controlled step using Cash-Karp 
    Runge-Kutta scheme with error estimate
    The estimated truncated error will be less than eps*yscal 
    
    Input 
    y - np 1d array 
        value of the multi-valued function at scalar value x
    dydx - np 1d array
        value of derivative dydx at scalar value x
    f_dydx - function(float, np array)
        the derivative of y with respect to x
        takes as input a point x, and a value of y
    n - int
        dimension of y
    x - float
        scalar value of x at beginning of interval
    htry- float
        trial step size to increment x 
    eps - float
        desired accuracy (relative accuracy to yscal)

    Output
    yout - np 1d array
        value of function y at (x+hdid) 
        hdid is determined adaptively...
    hdid - float
        stepsize used in the update
    hnext - float
        estimated next stepsize to use
    """
    SAFETY = 0.9
    PGROW = -0.2
    PSHRINK = -0.25
    ERRCON = 1.89e-4

    h = htry
    while True:
        yout, yerr = rkck(y, dydx, f_dydx, n, x, h)
        errmax = np.max(np.abs(yerr/yscal))
        errmax /= eps
        if errmax <= 1.0:
            break
        """
        Error is above tolerance, so reduce stepsize
        """
        htmp = SAFETY*h*errmax**(PSHRINK)
        """
        Don't reduce stepsize by more than a factor of 10
        """
        if htmp > 0: # moving righte
            h = max(htmp, 0.1*h)
        else: # moving left
            h = min(htmp, 0.1*h)
        xnew = x + h
        if xnew == x:
            raise OverflowError("Stepsize underflow in rkqs")
    if errmax > ERRCON:
        hnext = SAFETY*h*errmax**(PGROW)
    else:
        hnext = 5.0*h
    hdid = h
    x += hdid
    return yout, hdid, hnext

def odeint(ystart, f_dydx, n, x1, x2, eps, yscale, h1, hmin, max_steps, kmax, dx_sav):
    """
    Integrate function y from x1 to x2 using adaptive stepsize Runge-Kutta
    Cash-Karp method with error estimate as in Numerical Recipes Second Edition 
    (Press et al. 1992)
    (Compare to odeint on page 721)

    Input - 
    ystart - np 1d array
        y at initial point x1
    f_dydx - function float, np 1d array
        the derivative of y with respect to x 
    n - int
        dimension of y
    x1 - float
        initial point x
    x2 - float
        final point x
    eps - float
        relative precision (truncation error bounded above by eps*yscale)
    yscale - np 1d array
        scale for the relative precision
        pass in an array of zeros if you wish to estimate it
        from the magnitude of y (as in Num. Rec.)
    h1 - float
        initial stepsize MAGNITUDE 
    hmin - float
        minimum stepsize 
    max_steps - int
        maximum number of integration steps
    kmax - int
        maximum number of intermediate steps to store
    dx_sav - float
        desired grid spacing of the saved intermediate points

    Output -
    yf - np 1d array
        final value of y
    y_out - np 2d array
        intermediate values of y
    x_out - np 1d array
        intermediate values of x
    nok - int
        number of good steps taken
    nbad - int
        number of bad steps taken (where h had to be updated)

    """
    TINY = 1e-30
    
    y = ystart.copy()
    x = x1
    h = h1 * np.sign(x2-x1) 
    nok = 0
    nbad = 0


    y_out = np.zeros((n, kmax))
    yf = np.zeros(n) # 1-d array for final point
    x_out = np.zeros(kmax)
    y_out[:, 0] = y.copy()
    xsav=  x - dx_sav*2.0 # xsav is the x value of last saved point...this initialization ensures that the first point is saved.
    kount = 0 # number of saved output values

    """
    Determine if adaptive yscale is needed
    """
    if not np.any(yscale):
        ADAP_YSCALE = True
    else:
        ADAP_YSCALE = False


    for i in range(max_steps):
        dydx = f_dydx(x, y)
        if ADAP_YSCALE == True:
            yscale = np.abs(y) + np.abs(dydx*h) + TINY
        if (kmax > 0) and (kount < kmax-1) and (np.abs(x-xsav) > np.abs(dx_sav)): # save this point
            y_out[:, kount] = y.copy()
            x_out[kount] = x
            kount += 1
            xsav = x
        if ((x + h - x2) * (x+ h - x1) > 0.0): # if step overshoots x2, reduce
            h = x2 - x
        """
        Update y
        """
        y, hdid, hnext = rkqs(y, dydx, f_dydx, n, x, h, eps, yscale)
        if hdid == h: # successful step
            nok += 1
        else:
            nbad += 1
        if ((x-x2)*(x2-x1) >= 0.0): # if we've reached x2, we're done
            if kmax > 0: # save final output...
                y_out[:, kount] = y.copy()
                x_out[kount] = x
                kount += 1
            return y, x_out[:kount], y_out[:,:kount], nok, nbad
        if np.abs(hnext) <= hmin:
            raise ValueError('Step size too small in odeint')
        h = hnext
        x += hdid
    raise ValueError('Too many steps in odeint')
        




    


    
