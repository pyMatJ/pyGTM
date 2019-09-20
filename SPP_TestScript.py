#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Example file for the GTM package.
This file demonstrates the use of the new (20-09-2019) `System` methods 
for the computation of dispersion relations of surface modes in multilayer media. 

In this example we calculate the dispersion relation of Au/Air and Ag/Air SPPs.

The presence of surface mode is strongly linked to singularities in the 
Reflection coefficient. This can be traced back to a common denominator in all 
reflection/transmission coefficients, that has to be minimized. 

We start by setting up the two systems. 
Then, we evaluate the value of this denominator as a function of the in-plane 
wavevector value, using `System.calculate_matelem`. 
Since guided modes exist below the light cone, their in-plane wavevector 
can be a complex quantity. 
We thus use a regular grid for both real and imaginary part of the in-plane 
wavevector and look for a local minimum. The minimization is performed in 
the `System.calculate_eigen_wv` method.
Note that there is several local minima depending on the number of surface modes.

Then, the dispersion relation is obtained by sweeping the frequency and 
following a given local minimum in a step-wise manner, using the 
`System.disp_vs_f` method. 
"""

import numpy as np
import matplotlib.pyplot as plt
import GTM.TransferMatrix_PasslerAlgo as GTM
import GTM.Permittivities as mat


#%% System setup
c_const = 299792458 # m/s

eps_glass = lambda x: (1.5+1.0j*0.0)**2


## create a void system: substrate=superstrate=vac
SAu = GTM.System()
SAl = GTM.System()
# define the prism, airgap and SiC6H layers
Au = GTM.Layer(thickness = 1e-6, epsilon1=mat.eps_Au) # epsilon2 and epsilon3 default to epsilon1
Al = GTM.Layer(thickness = 1e-6, epsilon1=mat.eps_Ag) # epsilon2 and epsilon3 default to epsilon1
TopAir = GTM.Layer()
# start setting up the system
SAu.set_substrate(Au)
SAl.set_substrate(Al)
SAu.set_superstrate(TopAir) # not really necessary
SAl.set_superstrate(TopAir) # not really necessary

#%% Perform a 2D map of the disp. rel. to find initial guess 
fdisp = c_const/1e-6 # initial frequency for dispersion relation

### carefull choice in zr//zi since there can be several modes
zeta_r = np.linspace(0.9,2,50) ## real part of the wavevector. Avoid k=k0
zeta_i = np.geomspace(1e-6,5e-2,50) ## imaginary part of the wavevector
zrm, zim = np.meshgrid(zeta_r, zeta_i) ## grid

## 2D map of the denominator value to be minimized
AlphaDisp_Au = np.zeros(zrm.shape)
AlphaDisp_Al = np.zeros(zrm.shape)
for ii in range(zrm.size):
    print(ii/zrm.size) # progress 
    SAu.initialize_sys(fdisp) # not really necessary
    SAl.initialize_sys(fdisp)
    zeta_sys = zrm.flat[ii]+1.0j*zim.flat[ii]
    _ = SAu.calculate_GammaStar(fdisp, zeta_sys) # no interest in the System transfer matrix
    zeta0 = [zrm.flat[ii], zim.flat[ii]] # format the input
    alphaAu = SAu.calculate_matelem(zeta0, fdisp) # denominator value
    alphaAl = SAl.calculate_matelem(zeta0, fdisp) # denominator value
    AlphaDisp_Au.flat[ii] = alphaAu
    AlphaDisp_Al.flat[ii] = alphaAl

Au_min = divmod(AlphaDisp_Au.argmin(), AlphaDisp_Au.shape[1]) ## 2D indices of minimum value https://stackoverflow.com/questions/30180241
zeta0_Au = [zeta_r[Au_min[1]] , zeta_i[Au_min[0]]] # initial guess for minimization
resAu = SAu.calculate_eigen_wv(zeta0, fdisp) # minimization

Al_min = divmod(AlphaDisp_Al.argmin(), AlphaDisp_Al.shape[1]) ## 2D indices of minimum value https://stackoverflow.com/questions/30180241
zeta0_Al = [zeta_r[Al_min[1]] , zeta_i[Al_min[0]]] # initial guess for minimization
resAl = SAl.calculate_eigen_wv(zeta0, fdisp) # minimization

#%% plot 2D maps // no colorbar as the actual value has no real interest
fig2D = plt.figure(figsize=(6,4))
axAu = fig2D.add_subplot(121)
axAu.pcolormesh(zrm, zim, AlphaDisp_Au)
axAu.plot(zeta0_Au[0], zeta0_Au[1], '*g', ms=10)
axAu.plot(resAu.x[0], resAu.x[1], '*r', ms=10)
axAu.set_yscale('log')
axAu.set_xlabel('Real(k//)')
axAu.set_ylabel('Imag(k//)')


axAl = fig2D.add_subplot(122)
axAl.pcolormesh(zrm, zim, -np.log10(AlphaDisp_Al))
axAl.plot(zeta0_Al[0], zeta0_Al[1], '*g', ms=10)
axAl.plot(resAl.x[0], resAl.x[1], '*r', ms=10)
axAl.set_yscale('log')
axAl.set_xlabel('Real(k//)')
axAl.set_ylabel('Imag(k//)')

fig2D.tight_layout()
fig2D.show()

#%% frequency dependent dispersion relation
fv = np.linspace(c_const/1e-6, c_const/250e-9, 50) ## frequencies for disp rel

# Au system. Initial guess and calculation
zeta0 = [resAu.x[0], resAu.x[1]] 
zeta_disp_r_Au, zeta_disp_i_Au = SAu.disp_vs_f(fv, zeta0)
# Al system. Initial guess and calculation
zeta0 = [resAl.x[0], resAl.x[1]] 
zeta_disp_r_Al, zeta_disp_i_Al = SAl.disp_vs_f(fv, zeta0)
#
epsAudisp = np.array([mat.eps_Au(fj) for fj in fv])
epsAgdisp = np.array([mat.eps_Ag(fj) for fj in fv])
kspp_Au =  np.sqrt(epsAudisp/(1.0+epsAudisp)) ## Analytical expression
kspp_Ag =  np.sqrt(epsAgdisp/(1.0+epsAgdisp)) ## Analytical expression

#%% plot
figDisp = plt.figure()
axDisp = figDisp.add_subplot(111)
axDisp.plot(zeta_disp_r_Au, fv)
axDisp.plot(np.real(kspp_Au), fv, '--k')
axDisp.plot(zeta_disp_r_Al, fv)
axDisp.plot(np.real(kspp_Ag), fv, ':k')
axDisp.plot(np.ones(len(fv)), fv, 'b:') # light line in air (n==1)
axDisp.set_xlabel('Real(k//)')
axDisp.set_ylabel('Frequency (Hz)')
figDisp.show()

