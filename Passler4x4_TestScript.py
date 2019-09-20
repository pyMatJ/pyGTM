#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Example file for the GTM package.

It reproduces two specific configurations from the original paper of Passler and Paarmann
Passler, N. C. and Paarmann, A., JOSA B 34, 2128 (2017) doi.org/10.1364/JOSAB.34.002128
Fig. 2(a,b) with a 5.5um air gap 
Fig. 3(a,b) with a 3.5um air gap 
"""

import numpy as np
import matplotlib.pyplot as plt
import GTM.TransferMatrix_PasslerAlgo as GTM
import GTM.Permittivities as mat

#%%
c_const = 299792458 # m/s

## create a void system: substrate=superstrate=vac
S = GTM.System()
# define the prism, airgap and SiC6H layers
KRS5 = GTM.Layer(thickness = 1e-6, epsilon1=mat.eps_KRS5) # epsilon2 and epsilon3 default to epsilon1
AirGap = GTM.Layer(thickness = 5.5e-6) # no epsilon default to vacuum
SiC6H = GTM.Layer(thickness = 3e-6, epsilon1=mat.eps_SiC6Hx, # epsilon2 defaults to epsilon1
                  epsilon3 = mat.eps_SiC6Hz)

# start setting up the system
S.set_superstrate(KRS5)
S.set_substrate(SiC6H)
S.add_layer(AirGap)

# angle of incidence
thetain = np.deg2rad(30)

wnplot = np.arange(750, 1050, dtype=np.float) # wavenumber range of the plot in cm-1
fplot = wnplot*c_const*1e2 # corresponding frequency
dz = 100e-9 # spatial resolution


Rplot = np.zeros(len(wnplot))

Ex = []

for ii, fi in enumerate(fplot):
    S.initialize_sys(fi)
    zeta_sys = np.sin(thetain)*np.sqrt(S.superstrate.epsilon[0,0])
    Sys_Gamma = S.calculate_GammaStar(fi, zeta_sys)
    r, R, tfield, t, T = S.calculate_r_t()
    zplot, E_out, zn_plot = S.calculate_Efield(fi, zeta_sys, dz)
    Rplot[ii] = R[0]
    Ex.append(E_out[0,:])

#%% Plot the results
Exm = np.asarray(Ex) # make a 2D array from the electric field list

wnm, zm = np.meshgrid(wnplot,zplot*1e6)

fig2ab = plt.figure(figsize=(10,4))
axR = fig2ab.add_subplot(121)   
axR.plot(wnplot,Rplot, '+-')
axR.set_ylim([0,1.05])
axR.set_xlabel('wavenumber (cm$^{-1}$)')
axR.set_ylabel('Reflectivity')

axfield = fig2ab.add_subplot(122)
axc = axfield.pcolormesh(wnm, zm, np.abs(Exm.T),
                         vmin=0, vmax=5, shading='gouraud', cmap=plt.cm.gnuplot2)
for ii, zi in enumerate(zn_plot):
    axfield.plot([wnplot.min(),wnplot.max()],
                  [zi*1e6, zi*1e6], '--k')
axfield.invert_yaxis()
axfield.set_xlabel('wavenumber (cm$^{-1}$)')
axfield.set_ylabel('z-position ($\mu$m)')
axfield.set_ylim([8.5,0])

fig2ab.colorbar(axc)
fig2ab.tight_layout()
fig2ab.show()


#%% Add the GaN layer
S2 = GTM.System()
# define the new airgap and GaN layers
AirGap2 = GTM.Layer(thickness = 3.5e-6) # no epsilon default to vacuum
GaN = GTM.Layer(thickness = 2e-6, epsilon1=mat.eps_GaNx,
                epsilon3 = mat.eps_GaNz)
# start setting up the system
S2.set_superstrate(KRS5)
S2.set_substrate(SiC6H)
S2.add_layer(AirGap2)
S2.add_layer(GaN)

wnplot2 = np.arange(550, 1050) # wavenumber range of the plot in cm-1
fplot2 = wnplot2*c_const*1e2 # corresponding frequency

Rplot2 = np.zeros(len(wnplot2))
Ex2 = []
for ii, fi in enumerate(fplot2):
    S2.initialize_sys(fi)
    zeta_sys = np.sin(thetain)*np.sqrt(S.superstrate.epsilon[0,0])
    Sys_Gamma = S2.calculate_GammaStar(fi, zeta_sys)
    r, R, tfield, t, T = S2.calculate_r_t()
    zplot2, E_out, zn_plot2 = S2.calculate_Efield(fi, zeta_sys, dz)
    Rplot2[ii] = R[0]
    Ex2.append(E_out[0,:])
    
#%% Plot the results
Exm2 = np.asarray(Ex2) # make a 2D array from the electric field list

wnm2, zm2 = np.meshgrid(wnplot2,zplot2*1e6)

fig3ab = plt.figure(figsize=(10,4))
axR2 = fig3ab.add_subplot(121)   
axR2.plot(wnplot2,Rplot2, '+-')
axR2.set_ylim([0,1.05])
axR2.set_xlabel('wavenumber (cm$^{-1}$)')
axR2.set_ylabel('Reflectivity')

axfield2 = fig3ab.add_subplot(122)
axc2 = axfield2.pcolormesh(wnm2, zm2, np.abs(Exm2.T),
                           vmin=0, vmax=4, 
                           shading='gouraud', cmap=plt.cm.gnuplot2)
for ii, zi in enumerate(zn_plot2):
    axfield2.plot([wnplot2.min(),wnplot2.max()],
                  [zi*1e6, zi*1e6], '--k')
axfield2.invert_yaxis()
axfield2.set_xlabel('wavenumber (cm$^{-1}$)')
axfield2.set_ylabel('z-position ($\mu$m)')
axfield2.set_ylim([8.5,0])

fig3ab.colorbar(axc2)
fig3ab.tight_layout()
fig3ab.show()
