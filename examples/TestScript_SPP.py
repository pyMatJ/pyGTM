# This file is part of the pyGTM module.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. 
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#     
# Copyright (C) Mathieu Jeannin 2019 2020 <math.jeannin@free.fr>.
"""
Example file for the pyGTM package.
This file demonstrates the use of the new (20-09-2019) `System` methods 
for the computation of dispersion relations of surface modes in multilayer media. 

In this example we calculate the dispersion relation of Au/Air and Ag/Air SPPs.

The presence of surface mode is strongly linked to singularities in the 
reflection coefficient. 
This can be traced back to a common denominator in all 
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

** warning ** This is strongly dependent on the minimization procedure.
Here, a sequential least-square programming is used through 
scipy.minimize(method='SLSQP'), with strong bounds. The guided mode is often 
a local minimum, and minimization algorithms tend to jump to global minima.

Once the correct wavevector is found, (requires a bit of trial and error)
the dispersion relation is obtained by sweeping the frequency and 
following the given local minimum in a step-wise manner, using the 
`System.disp_vs_f` method. 
"""

import numpy as np
import matplotlib.pyplot as plt
import GTM.GTMcore as GTM
import GTM.Permittivities as mat

#%% System setup
c_const = 299792458 # m/s

eps_glass = lambda x: (1.5+1.0j*0.0)**2


## create a void system: substrate=superstrate=vac
SAu = GTM.System()
SAl = GTM.System()
# define the layers
Au = GTM.Layer(thickness = 1e-6, epsilon1=mat.eps_Au) # epsilon2 and epsilon3 default to epsilon1
Al = GTM.Layer(thickness = 1e-6, epsilon1=mat.eps_Ag) # epsilon2 and epsilon3 default to epsilon1
TopAir = GTM.Layer()
# start setting up the system
SAu.set_substrate(Au)
SAl.set_substrate(Al)
SAu.set_superstrate(TopAir) # not really necessary
SAl.set_superstrate(TopAir) # not really necessary

#%% Perform a 2D map of the disp. rel. to find initial guess 
fdisp = c_const/200e-9 # initial frequency for dispersion relation

### carefull choice in zr//zi since there can be several modes
zeta_r = np.linspace(0.8,1.1,40) ## real part of the wavevector. Avoid k=k0
zeta_i = np.linspace(1.0e-4,2.0e-1,50) ## imaginary part of the wavevector
zrm, zim = np.meshgrid(zeta_r, zeta_i) ## grid

## 2D map of the denominator value to be minimized
AlphaDisp_Au = np.zeros(zrm.shape)
AlphaDisp_Al = np.zeros(zrm.shape)
for ii in range(zrm.size):
    print(ii/zrm.size) # progress 
    ####### taken care of in calculate_matelem: initialization and computation
    #S.initialize_sys(fdisp)
    #zeta_sys = zrm.flat[ii]+1.0j*zim.flat[ii]
    #_ = S.calculate_GammaStar(fdisp, zeta_sys) 
    #######
    zeta0 = [zrm.flat[ii], zim.flat[ii]] # format the input
    alphaAu = SAu.calculate_matelem(zeta0, fdisp) # denominator value to be minimized
    alphaAl = SAl.calculate_matelem(zeta0, fdisp) # denominator value to be minimized
    AlphaDisp_Au.flat[ii] = alphaAu 
    AlphaDisp_Al.flat[ii] = alphaAl

## bounds for the minimization algorithm
bounds = ((zeta_r.min(),zeta_r.max()),
          (zeta_i.min(), zeta_i.max()))

## 2D indices of minimum value https://stackoverflow.com/questions/30180241
Au_min = divmod(AlphaDisp_Au.argmin(), AlphaDisp_Au.shape[1]) 
zeta0_Au = [zeta_r[Au_min[1]] , zeta_i[Au_min[0]]] # initial guess for minimization
resAu = SAu.calculate_eigen_wv(zeta0, fdisp, bounds=bounds) # minimization

## 2D indices of minimum value https://stackoverflow.com/questions/30180241
Al_min = divmod(AlphaDisp_Al.argmin(), AlphaDisp_Al.shape[1]) 
zeta0_Al = [zeta_r[Al_min[1]] , zeta_i[Al_min[0]]] # initial guess for minimization
resAl = SAl.calculate_eigen_wv(zeta0, fdisp, bounds=bounds) # minimization

#%% plot 2D maps // the actual value has no real interest
# This is to check that a guided mode exists in the given bounds
# it should be approximated by zeta0 and found by S.calculate_eigen_wv.

fig2D = plt.figure(figsize=(6,4))
axAu = fig2D.add_subplot(121)
cAu = axAu.pcolormesh(zrm, zim, np.log10(AlphaDisp_Au))
axAu.plot(zeta0_Au[0], zeta0_Au[1], 'og', ms=8) ## numerical minimum as computed
axAu.plot(resAu.x[0], resAu.x[1], '*r', ms=10) ## result of the minimization algo.
axAu.set_yscale('log')
axAu.set_xlabel('Real(k//)')
axAu.set_ylabel('Imag(k//)')


axAl = fig2D.add_subplot(122)
axAl.pcolormesh(zrm, zim, np.log10(AlphaDisp_Al))
axAl.plot(zeta0_Al[0], zeta0_Al[1], 'og', ms=8) ## numerical minimum as computed
axAl.plot(resAl.x[0], resAl.x[1], '*r', ms=10) ## result of the minimization algo.
axAl.set_yscale('log')
axAl.set_xlabel('Real(k//)')
axAl.set_ylabel('Imag(k//)')

fig2D.colorbar(cAu)
fig2D.tight_layout()
fig2D.show()

#%% frequency dependent dispersion relation

## frequencies are swept **starting from the value carefully determined above**
fv = np.geomspace(c_const/200.0e-9, c_const/1e-6, 100) 
bounds = ((0.45, 1.8), ## Real(kspp/k0) bounds
          (0., 10.)) ## Imag(kspp/k0) bounds

# Au system. Initial guess and calculation
zeta0 = [resAu.x[0], resAu.x[1]] # initial guess is above
zeta_disp_r_Au, zeta_disp_i_Au = SAu.disp_vs_f(fv, zeta0, bounds=bounds)
## Al system. Initial guess and calculation
zeta0 = [resAl.x[0], resAl.x[1]] # initial guess is above
zeta_disp_r_Al, zeta_disp_i_Al = SAl.disp_vs_f(fv, zeta0, bounds=bounds)
##
epsAudisp = np.array([mat.eps_Au(fj) for fj in fv])
epsAgdisp = np.array([mat.eps_Ag(fj) for fj in fv])
kspp_Au =  np.sqrt(epsAudisp/(1.0+epsAudisp)) ## Analytical expression
kspp_Ag =  np.sqrt(epsAgdisp/(1.0+epsAgdisp)) ## Analytical expression


#%% plot of the dispersion relation

figDisp = plt.figure()
axDisp = figDisp.add_subplot(111)
axDisp.semilogy(zeta_disp_r_Au, fv,
                label='$k_{spp,Au}$ (GTM)')
axDisp.plot(np.real(kspp_Au), fv, '--k',
                label='$k_{spp,Au}$ (th.)')
axDisp.plot(zeta_disp_r_Al, fv,
            label='$k_{spp,Ag}$ (GTM)')
axDisp.plot(np.real(kspp_Ag), fv, ':k',
            label='$k_{spp,Ag}$ (th.)')
axDisp.plot(np.ones(len(fv)), fv, 'b:',
            label='light line n=1') # light line in air (n==1)
axDisp.plot([bounds[0][0],bounds[0][0]], [fv.min(), fv.max()], ':', 
            c='grey', label='bounds') # bounds
axDisp.plot([bounds[0][1],bounds[0][1]], [fv.min(), fv.max()], ':', c='grey') # bounds
#axDisp.set_xlim((0,2))
axDisp.set_xlabel('Real(k//)')
axDisp.set_ylabel('Frequency (Hz)')
axDisp.legend(frameon=False,
              handlelength=1.5, handletextpad=0.2, 
              bbox_to_anchor=(1.,0.5))
figDisp.tight_layout()
figDisp.show()

