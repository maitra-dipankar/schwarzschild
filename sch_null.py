#!/usr/bin/env python

'''
Computes null geodesics. i.e. photon paths, in Schwarzschild spacetime.
The code then creates some basic visualizations of the computed null 
geodesics.

The code follows the conventions/prescriptions in the following two books:
    * Mathematical Theory of Black Holes, by Chandrasekhar.
    * Gravitation, by Misner, Thorne, & Wheeler.

To use the code the user has to specify the following inputs:
r0 = distance between the center of the black hole and the starting point of
     the light ray. This distance is given in units of gravitational radius 
     (GM/c^2) so that, for example, r0=6 means the starting location is 
     6GM/c^2 = 3 Schwarzschild radius away.

delta0 = angle that the light ray's path makes, with respect to the radially
         outward direction, at the starting location r0. delta0 has to be in
         radians. For example, r0=6 and delta0=0 means a light ray starting at
         6GM/c^2 and going directly away from the black hole. Similarly r0=6 
         and delta0 = pi means a light ray starting at 6GM/c^2 but going 
         directly towards the black hole. If r0=6 and delta0=pi/2 then light
         ray at 6GM/c^2 is moving perpendicular to the radia direction. If 
         (r0, delta0) = (6, pi/4) then the ray at 6GM/c^2 is making an angle of
         45-degrees with respect to the radially outward direction.

rMin = stop code if the distance between the ray and the center of the black
       hole becomes less than rMin. This distance is in units of gravitational
       radius (GM/c^2). If the user does not specify an rMin, then the code
       stops just outside the Schwarzschild event horizon (the default value
       of rMin is 2.000001).

rMax = stop code if the distance between the ray and the center of the black
       hole becomes larger than rMax. This distance is in units of
       gravitational radius (GM/c^2). The default value of rMax is 10.

npts = number of integration steps. Default is 10,000. If this is too small 
       then the code will stop before reaching rMin or rMax.


Code output:
r   = array of radial distances of the ray (in units of gravitational radius)
phi = array of azimuthal coordinates of the ray (in radian)
t   = array of Schwarzschild time coordinate of the ray (in units of GM/c^3)


Pre-requisites (beyond basic python):
    * numpy, matplotlib
    * Also needs the file cubic_realroots.py in the same location as this code.

To run code: 
    python sch_null.py

See detailed usage at the bottom of the code.
'''

import sys
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from cubic_realroots import depressedCubicDistinctRealRoots as cr

DEBUGSW = False #True

def rTP(bSq, M):                    # location of turning points
    '''
    When b^2 > 27 M, the turning points are the positive roots of
    R^3 - b^2 R + 2 M b^2 = 0
    '''
    roots = cr(-bSq, 2.*bSq*M)  # one -ve, two +ve roots, in ascending order
    positiveRoots = roots[1:]   # select only the positive roots

    return positiveRoots


def B(r, M):
    return r / np.sqrt(1 - 2*M/r)


def Bminus2 (r, M=1.):              # 1/B^2 from pg 674 of MTW
    return (1. - 2.*M/r)/r/r


def drdl(r, bminus2, sign, M):      # d(r)/d(lambda)
    drdlSq = bminus2 - Bminus2 (r, M)
    return sign*np.sqrt(drdlSq)


def dphidl(r):                      # d(phi)/d(lambda)
    return 1./r/r


def dtdl(r, b, M):                  # d(t)/d(lambda)
    return r/b/(r - 2.*M)


def integrateRadialOutwards (r0, rMax, npts, M=1., phi0=0., t0=0.):
    ''' Special case: radially outward till rMax'''
    if DEBUGSW:
        print('radially outward till rMax')

    r = np.linspace(r0, rMax,npts)
    phi = np.full(npts, phi0)

    t_0 = r0 + 2.*M * np.log(r0 - 2.*M)
    t  = r  + 2.*M * np.log(r  - 2.*M) - t_0

    return r, phi, t + t0


def integrateRadialInwards (r0, rMin, npts, M=1., phi0=0., t0=0.):
    ''' Special case: radially inward till max(rMin, 2M+0) '''
    if DEBUGSW:
        print('radially inward till greater of rMin or 2M+0')

    rmin = (2 + 1e-6)*M
    if rMin > rmin:
        rmin = rMin

    r = np.linspace(r0, rmin, npts)
    phi = np.full(npts, phi0)

    t_0 = r0 + 2.*M * np.log(r0 - 2.*M)
    t  = t_0 - (r  + 2.*M * np.log(r  - 2.*M))

    return r, phi, t + t0


def integrateFromPeriastron (r, phi, t, r0, rMax, M=1., phi0=0., t0=0., \
        dl=1e-2, npts=10000):
    '''
    Special case: r0>3M and delta0=pi/2. In this case r0 is periastron.
    '''
    if DEBUGSW:
        print('starting at periastron')

    # do the first step manually, using a small increase in r to 
    # compute resulting dl, dphi, and dt.
    dr = 1e-6*r[0]
    r[1] = r[0] + dr
    b = B(r0, M)
    bm2 = 1.0/b/b

    Bm2    = (r[1]-2*M)/r[1]/r[1]/r[1]
    dl0    = dr/np.sqrt(bm2 - Bm2)
    dphi   = dl0 * dphidl(r[0])
    phi[1] = phi[0] + dphi
    t[1]   = t[0]   + dl0 * dtdl(r[0], b, M)

    dy = r[1]*np.sin(dphi)
    r1_minus_r0 = np.sqrt(r[0]*r[0] + r[1]*r[1] - \
            2*r[0]*r[1]*np.cos(dphi))
    sindelta = dy/r1_minus_r0

    # now do the rest
    sgn=1
    for i in range(1,npts-1):
        rNew   = r[i]   + dl * drdl(r[i], bm2, sgn, M)
        phiNew = phi[i] + dl * dphidl(r[i])
        tNew   = t[i]   + dl * dtdl(r[i], b, M)
        if rNew > rMax:
            break
        r[i+1], phi[i+1], t[i+1] = rNew, phiNew, tNew

    if r[-1] < rMax:
        print('ran out of points before getting to rMax.')

    return 0 

   
def integrateFromApastron (r, phi, t, r0, M=1., phi0=0., t0=0., rMin=2., \
        dl=1e-2, npts=10000):
    '''
    Special case: r0<3M and delta0=pi/2. In this case r0 is apastron.
    '''
    if DEBUGSW:
        print('starting at apastron')

    # go to the second  hand, using a small decrease in r to 
    # compute resulting dl, dphi, and dt.
    dr = 1e-6*r[0]
    r[1] = r[0] - dr
    b = B(r0, M)
    bm2 = 1.0/b/b

    Bm2    = (r[1]-2*M)/r[1]/r[1]/r[1]
    dl0    = dr/np.sqrt(bm2 - Bm2)
    dphi   = dl0 * dphidl(r[0])
    phi[1] = phi[0] + dphi
    t[1]   = t[0]   + dl0 * dtdl(r[0], b, M)

    dy = r[1]*np.sin(dphi)
    r1_minus_r0 = np.sqrt(r[0]*r[0] + r[1]*r[1] - \
            2*r[0]*r[1]*np.cos(dphi))
    sindelta = dy/r1_minus_r0

    # now do the rest
    sgn=-1
    for i in range(1,npts-1):
        rNew   = r[i]   + dl * drdl(r[i], bm2, sgn, M)
        phiNew = phi[i] + dl * dphidl(r[i])
        tNew   = t[i]   + dl * dtdl(r[i], b, M)
        if rNew < rMin:
            break
        r[i+1], phi[i+1], t[i+1] = rNew, phiNew, tNew

    return 0 


def integrateNoTP (r, phi, t, r0, sin_delta0, sgn, M=1., phi0=0., t0=0., \
        rMin=0., rMax=10., dl=1e-2, npts=10000):
    '''
    Integrate orbits without any turning points, i.e., where the 
    sign of dr/dlambda doesn't change. 
    '''
    if DEBUGSW:
        print('using integrateNoTP function')

    b = B(r0, M) * sin_delta0

    bm2 = 1.0/b/b

    i=0
    for i in range(npts-1):
        rNew   = r[i]   + dl * drdl(r[i], bm2, sgn, M)
        phiNew = phi[i] + dl * dphidl(r[i])
        tNew   = t[i]   + dl * dtdl(r[i], b, M)
        if rNew < rMin or rNew > rMax:
            break
        r[i+1], phi[i+1], t[i+1] = rNew, phiNew, tNew

    if r[-1] < rMax:
        print('ran out of points before getting to rMax.')

    return 0 

def integrateSchGeodesic (r0, delta0, M=1., phi0=0., t0=0., \
        rMin=2 + 1e-6, rMax=10., npts=10000):
    '''
    Integrate geodesics in Schwarzschild metric. 
    Some of these have a turning point, i.e., where the sign of dr/dlambda 
    changes at peri/apastron. The peri/apastron points are given by the 
    positive real roots of R^3 - b^2 R + 2 M b^2 = 0
    '''

    # handle some special cases first

    # Special case: radially outward till rMax
    if np.isclose(delta0, 0.):
        r, phi, t = integrateRadialOutwards (r0, rMax, npts, M, phi0, t0)
        return r, phi, t

    # Special case: radially inward till max(rMin, 2M+0) 
    if np.isclose(delta0, np.pi):
        r, phi, t = integrateRadialInwards  (r0, rMin, npts, M, phi0, t0)
        return r, phi, t
 
    if np.isclose(delta0, np.pi/2):
        sindelta0 = 1.
        sgn=0.
    else:
        sindelta0 = np.sin(delta0)
        sgn = np.sign(np.pi/2 - delta0)

    dl = 1e-2
    r   = np.full(npts, np.nan)
    phi = np.full(npts, np.nan)
    t   = np.full(npts, np.nan)

    r[0]   = r0
    phi[0] = phi0
    t[0]   = t0

    # Special case: starting at periastron and going out till rMax
    if np.isclose(delta0, np.pi/2) and r0>3*M:
        ret = integrateFromPeriastron (r, phi, t, r0, rMax, M, phi0, t0, \
                npts=npts)
        return r, phi, t

    # Special case: starting at apastron and going in till rMin
    if np.isclose(delta0, np.pi/2) and r0<3*M:
        ret = integrateFromApastron (r, phi, t, r0, M=1., phi0=0., \
                t0=0., rMin=rMin)
        return r, phi, t


    # calculate impact parameter for non-radial geodesics
    b = B(r0, M) * sindelta0
    bm2 = 1.0/b/b

    b_bcrit = b / (np.sqrt(27)*M)
    if np.isclose(b_bcrit, 1.):
        b_bcrit = 1.

    if b_bcrit <= 1.0:
        ret = integrateNoTP (r, phi, t, r0, sindelta0, sgn, M, phi0, t0, \
                rMin, rMax, npts=npts)
        return r, phi, t

    apa, per = rTP(b*b, M)            # get apas/periastron for this b
    if DEBUGSW:
        print('apastron = ', apa, 'periastron = ', per)

    if np.isclose(r0, per) and np.isclose(delta0, np.pi/2):
        if DEBUGSW:
            print('at periastron and can only go away')

        sgn = 1.0
        ret = integrateNoTP (r, phi, t, r0, sindelta0, sgn, M, phi0, t0, \
                rMin, rMax, npts=npts)

    elif np.isclose(r0, apa) and np.isclose(delta0, np.pi/2):
        if DEBUGSW:
            print('at apastron and can only go in')

        sgn = -1.0
        ret = integrateNoTP (r, phi, t, r0, sindelta0, sgn, M, phi0, t0, \
                rMin, rMax, npts=npts)

    elif (r0 > per and sgn > 0) or (r0 < apa and sgn < 0):
        sgn = np.sign(np.cos(delta0))
        if DEBUGSW:
            print('noTP ... dr forever:', sgn)

        ret = integrateNoTP (r, phi, t, r0, sindelta0, sgn, M, phi0, t0, \
                rMin, rMax, npts=npts)

    elif r0 < apa and sgn > 0:
        if DEBUGSW:
            print('will get to apastron and then keep going in forever')

        i=0
        sign = 1
        for i in range(npts-1):
            rNew   = r[i]   + dl * drdl(r[i], bm2, sign, M)
            phiNew = phi[i] + dl * dphidl(r[i])
            tNew   = t[i]   + dl * dtdl(r[i], b, M)

            if np.isclose(rNew, apa):
                sign = -1

            if rNew < rMin or rNew > rMax:
                break
            r[i+1], phi[i+1], t[i+1] = rNew, phiNew, tNew

    elif r0 > per and sgn < 0:
        if DEBUGSW:
            print('will get to periastron and then keep going out forever')

        i=0
        sign = -1
        for i in range(npts-1):
            rNew   = r[i]   + dl * drdl(r[i], bm2, sign, M)
            phiNew = phi[i] + dl * dphidl(r[i])
            tNew   = t[i]   + dl * dtdl(r[i], b, M)

            if np.isclose(rNew, per):
                sign = 1

            if rNew < 0 or rNew > rMax:
                break
            r[i+1], phi[i+1], t[i+1] = rNew, phiNew, tNew


    else:
        print('this should never happen!')

    return r, phi, t 



# Chandra's escape cone formula (Chandra, pg.127 eq. (244) )
def avoidAngle (r):      
    nn = np.sqrt(0.5*r - 1)
    d1 = r/3 - 1
    d2 = np.sqrt(r/6 + 1)
    dd = d1*d2
    return np.pi - np.arctan2(nn, dd) # measured from radial direction



def plotRay(r, phi, t):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    rr = 1.05*np.nanmax(r)    # x/y range for plotting

    ehT = np.deg2rad(range(361))
    ehX = 2*np.cos(ehT)
    ehY = 2*np.sin(ehT)

    fig = plt.figure(figsize=(16,7))

    fig.suptitle('Ray tracing in Schwarzschild spacetime', \
            fontsize=16, fontweight='bold')
    gs1 = GridSpec(4, 16, left=-0.05, right=0.99, wspace=0.8, hspace=0.25)
    ax1 = fig.add_subplot(gs1[0:4, 0:8])
    ax2 = fig.add_subplot(gs1[0:2, 9:16])
    ax3 = fig.add_subplot(gs1[2:4, 9:16])

    ax1.plot(ehX, ehY, 'k-.', lw=1)     # Plot event horizon
    im = ax1.scatter(x, y, c=t, s=2, cmap=cm.turbo)
    cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(r'time [R$_{\rm g}$/c]', rotation=90, fontsize = 14)
    ax1.set_xlim(-rr, rr)
    ax1.set_ylim(-rr, rr)
    ax1.set_aspect('equal')
    ax1.set_xlabel(r'X [R$_{\rm g}$]', fontsize = 14)
    ax1.set_ylabel(r'Y [R$_{\rm g}$]', fontsize = 14)

    ax2.scatter(t, r, c=t, s=2, cmap=cm.turbo)
    ax2.set_ylabel(r'r [R$_{\rm g}$]', fontsize = 14)
    ax2.grid()

    ax3.scatter(t, phi, c=t, s=2, cmap=cm.turbo)
    ax3.set_xlabel(r'time [R$_{\rm g}$/c]', fontsize = 14)
    ax3.set_ylabel(r'$\phi$ [radians]', fontsize = 14)
    ax3.grid()

    plt.show()


def SchwarzschildLampPost (z, nrays):   
    # nrays from a lamppost at height z above BH
    delta0s_deg = np.linspace(0, 180, nrays)
    delta0s = np.deg2rad(delta0s_deg)
    #delta0s = np.pi*np.random.rand(nrays)

    rr = 3*z    # x/y range for plotting

    ehT = np.deg2rad(range(361))
    ehX, ehY = 2*np.cos(ehT), 2*np.sin(ehT)

    fig = plt.figure(figsize=(11,11))
    ax = fig.add_subplot()

    ax.set_xlim(-rr, rr)
    ax.set_ylim(-rr, rr)
    ax.set_aspect('equal')
    ax.set_xlabel(r'X [R$_{\rm g}$]', fontsize = 14)
    ax.set_ylabel(r'Y [R$_{\rm g}$]', fontsize = 14)

    ax.plot(ehX, ehY, 'k-.', lw=1)     # Plot event horizon

    for delta0 in delta0s:
        print('delta0 (deg) = %.3f' % np.rad2deg(delta0))
        r, phi, tt = integrateSchGeodesic (z, delta0, \
                rMin=2, rMax=5*z, npts=100000)
        y, x = r * np.cos(phi), r * np.sin(phi)
        ax.plot(+x, y, 'k-', lw=0.5)
        ax.plot(-x, y, 'k-', lw=0.5)

    plt.show()

    return 0



##############################################################

#
# Various test cases with [M, phi0, t0] = [1, 0, 0]
# uncomment each block and run code to see results
#

# radially outward ray
#r0, delta0, rmax = 2.5, 0, 12           
#myr, myphi, myt = integrateSchGeodesic (r0, delta0, rMax=rmax)


# radially inward ray
#r0, delta0, rmin = 6, np.pi, 2.01  
#myr, myphi, myt = integrateSchGeodesic (r0, delta0, rMin=rmin)


# ray starting at periastron
#r0, delta0, rmax, nPts = 6.0, np.pi/2, 30, 100000
#myr, myphi, myt = integrateSchGeodesic (r0, delta0, rMax=rmax, npts=nPts)


# ray starting at apastron
r0, delta0, rmin = 2.5, np.pi/2, 2.01
myr, myphi, myt = integrateSchGeodesic (r0, delta0, rMin=rmin)


# ray near bcrit@3M
#r0, delta0, nPts = 3., np.pi/2, 10000
#r0, delta0, nPts = 3., np.pi/2 + 0.0001, 10000
#r0, delta0, nPts = 3., np.pi/2 - 0.0001, 100000
#myr, myphi, myt = integrateSchGeodesic (r0, delta0, rMin=2, rMax=12, npts=nPts)


# ray approaching bcrit from inside 3M 
#r0     = 2.5
#delta0 = np.arcsin(np.sqrt(27-54/r0)/r0) 
#delta0 = np.arcsin(np.sqrt(27-54/r0)/r0) + 0.01 # reaches apastron, then fall onto BH
#delta0 = np.arcsin(np.sqrt(27-54/r0)/r0) - 0.01 # manages to escape to infinity
#myr, myphi, myt = integrateSchGeodesic (r0, delta0, rMin=2)


# ray approaching bcrit from outside 3M 
#r0, delta0 = 6, np.deg2rad(180-45)
#r0, delta0 = 6, np.deg2rad(180-45.1)
#r0, delta0 = 6, np.deg2rad(180-44.9)
#myr, myphi, myt = integrateSchGeodesic (r0, delta0, rMin=2, rMax=9)


# testing Chandra's escape cone formula
# a ray originating inside 3M but can't get out
#r0 = 2.5
#delta0 = avoidAngle(r0)
#myr, myphi, myt = integrateSchGeodesic (r0, delta0, rMin=2, rMax=9)
#myr, myphi, myt = integrateSchGeodesic (r0, delta0-0.0001, rMin=2, rMax=9)
#myr, myphi, myt = integrateSchGeodesic (r0, delta0+0.0001, rMin=2, rMax=9)


#r0, delta0 = 6, np.deg2rad(140)
#myr, myphi, myt = integrateSchGeodesic (r0, delta0, rMin=0.2, rMax=50)

plotRay(myr, myphi, myt)


# a bunch of rays from a lamppost at height z above the black hole
#ret = SchwarzschildLampPost (3.3, 31)

