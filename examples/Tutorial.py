#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:02:49 2020

@author: mathieu
"""

import numpy as np
import matplotlib.pyplot as plt
from GTM.system import System
from GTM.layer import Layer
from GTM.structure import Structure
import GTM.permittivities as mat

#%% First example: Air/glass interface
epsGlass = lambda x : 1.5**2
Glass = Layer(epsilon1=epsGlass)
Air = Layer()

structure = Structure()

structure.set_superstrate(Air)
structure.set_substrate(Glass)

### Create an empty system
S = System(structure)

lbda = 600e-9 ## wavelength
c_const = 3e8 ## speed of light
f = c_const/lbda ## frequency
theta = np.deg2rad(np.linspace(0.0, 80.0, 500)) ## angle of incidence

Rplot = np.zeros((len(theta),2))  # intensity reflectivity (p- and s-pol)
Tplot = np.zeros((len(theta),2))  # intensity transmission (p- and s-pol)
for ii, thi in enumerate(theta):
    zeta_sys = np.sin(thi)*np.sqrt(structure.superstrate.epsilon[0, 0])  # in-plane wavevector
    Sys_Gamma = S.calculate_GammaStar(f, zeta_sys)  # Tranfer matrix of the system
    r, R, t, T = S.calculate_r_t(zeta_sys, Sys_Gamma)  # all reflection/transmission coefficients
    Rplot[ii, 0] = R[0]  # p-pol reflectivity
    Rplot[ii, 1] = R[1]  # s-pol reflectivity
    Tplot[ii, 0] = T[0]  # p-pol reflectivity
    Tplot[ii, 1] = T[1]  # s-pol reflectivity

plt.figure()
plt.plot(np.rad2deg(theta), Rplot[:,0], '-b', label=r'$R_p$')
plt.plot(np.rad2deg(theta), Rplot[:,1], '--b', label=r'$R_s$')
plt.plot(np.rad2deg(theta), Tplot[:,0], '-r', label=r'$T_p$')
plt.plot(np.rad2deg(theta), Tplot[:,1], '--r', label=r'$T_s$')
plt.xlabel('Angle of incidence (deg)')
plt.ylabel('Reflection/Transmission')
plt.legend()
plt.show()

#%%Second example: surface plasmon polariton

## Revert the substrate and superstrate
structure.set_superstrate(Glass)
structure.set_substrate(Air)
## define the Au layer
Au = Layer(thickness=50e-9, epsilon1=mat.eps_Au)
## Add the Au layer
structure.add_layer(Au)

Rplot = np.zeros((len(theta),2))  # intensity reflectivity (p- and s-pol)
Tplot = np.zeros((len(theta),2))  # intensity transmission (p- and s-pol)
for ii, thi in enumerate(theta):
    zeta_sys = np.sin(thi)*np.sqrt(structure.superstrate.epsilon[0, 0])  # in-plane wavevector
    Sys_Gamma = S.calculate_GammaStar(f, zeta_sys)  # Tranfer matrix of the system
    r, R, t, T = S.calculate_r_t(zeta_sys, Sys_Gamma)  # all reflection/transmission coefficients
    Rplot[ii,0] = R[0] # p-pol reflectivity
    Rplot[ii,1] = R[1] # s-pol reflectivity
    Tplot[ii,0] = T[0] # p-pol reflectivity
    Tplot[ii,1] = T[1] # s-pol reflectivity

plt.figure()
plt.plot(np.rad2deg(theta), Rplot[:, 0], '-b', label=r'$R_p$')
plt.plot(np.rad2deg(theta), Rplot[:, 1], '--b', label=r'$R_s$')
plt.plot(np.rad2deg(theta), Tplot[:, 0], '-r', label=r'$T_p$')
plt.plot(np.rad2deg(theta), Tplot[:, 1], '--r', label=r'$T_s$')
plt.xlabel('Angle of incidence (deg)')
plt.ylabel('Reflection/Transmission')
plt.legend()
plt.show()


#%% Electric field

thetaplot = np.deg2rad(45) ## momentum-matching angle
zeta_plot = np.sin(thetaplot)*np.sqrt(S.structure.superstrate.epsilon[0,0])
zplot, E_out, zn_plot = S.calculate_Efield(f, zeta_plot, dz=1e-9) # get the electric field

plt.figure()
## plot layer boundaries
yl = plt.gca().get_ylim()
for zi in zn_plot:
    plt.plot([zi, zi], [yl[0], yl[1]], ':k')
# z-component, p-polarized excitation
plt.plot(zplot, E_out[2,:], label='$E_z^p$')
# make it lisible
plt.text(-0.5e-6, -3.5, 'Glass')
plt.text(-1e-8, -3.5, 'Au')
plt.text(0.5e-6, -3.5, 'Air')
plt.xlabel('propagation axis z (m)')
plt.ylabel('Electric field (A.U.)')
plt.legend()
plt.show()