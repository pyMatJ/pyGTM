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
Example file for the pyGTM package

It reproduces the main result of section IV in Passler, Jeannin and Paarman
https://arxiv.org/abs/2002.03832
fig 2

It also demonstrates how to use the euler angles for the layers
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import GTM.GTMcore as GTM
import GTM.Permittivities as mat
from matplotlib import rcParams, rcParamsDefault

c_const = 3e8

#%% setup Simulation

## Parameters
fstart, fstop, fstep = 500e2, 1050e2, 2e2
f_sp = c_const*np.arange(fstart, fstop+fstep, fstep) ## frequency range (cm-1->Hz)
## **careful** angle of incidence is theta_in, 
## it is **not** the euler anle theta for the layers
theta_in = np.deg2rad(28)

## Note that Phi in the text is actually Euler angle psi
psi_start, psi_stop, psi_step = 0, 180, 2
psi_v = np.deg2rad(np.arange(psi_start, psi_stop+psi_step, psi_step)) # to get the last value
phi = 0.0
# the principal vertical axis of the layer is still z so theta=0
theta = 0.0

## physical dimensions
displaythick = 10e-6
tAir = 8e-6
tMoO3 = 1.2e-6
tAlN = 1.0e-6

## substrate, superstrate and layers, angles are left for later
KRS5 = GTM.Layer(thickness=displaythick,
                 epsilon1=mat.eps_KRS5)
SiC = GTM.Layer(thickness=displaythick, 
                epsilon1=mat.eps_SiCx, epsilon3=mat.eps_SiCz)
AirGap = GTM.Layer(thickness=tAir)
MoO3 = GTM.Layer(thickness=tMoO3, 
                 epsilon1=mat.eps_MoO3x, epsilon2=mat.eps_MoO3y, 
                 epsilon3=mat.eps_MoO3z)
AlN = GTM.Layer(thickness=tAlN,
                epsilon1=mat.eps_AlNx, epsilon3=mat.eps_AlNz)
## setup the system
S = GTM.System()
S.set_superstrate(KRS5)
S.set_substrate(SiC)
S.add_layer(AirGap)
S.add_layer(MoO3)
S.add_layer(AlN)


#%%
R = np.zeros((len(psi_v),len(f_sp)))
A = np.zeros((len(psi_v),len(f_sp),3))

for jj, psij in enumerate(psi_v):
    print(jj/len(psi_v))
    ## set the layers orientation
    S.substrate.set_euler(theta=theta, phi=phi, psi=psij)
    S.superstrate.set_euler(theta=theta, phi=phi, psi=0.0) # this guy stands still
    for L in S.layers:
        L.set_euler(theta=theta, phi=phi, psi=psij)
    # loop for frequency
    for ii, fi in enumerate(f_sp):
        S.initialize_sys(fi) # sets the epsilons w/ correct euler rotation
        zeta_sys = np.sin(theta_in)*np.sqrt(S.superstrate.epsilon[0,0]) # in-plane wavevector
        S.calculate_GammaStar(fi, zeta_sys) # main computation (discard output)
        r, R_loc, t, T_loc = S.calculate_r_t(zeta_sys) # calculate reflectivity
        R[jj,ii] = R_loc[0] # p-pol only
        zplot, E_out, H_out, zn_plot = S.calculate_Efield(fi, zeta_sys, 
                                                  magnetic=True)
        zMoO3 = np.abs(zplot-(tAir+tMoO3)).argmin()
        zAlN = np.abs(zplot-(tAir+tMoO3+tAlN)).argmin()
        zSiC = np.abs(zplot-(tAir+tMoO3+tAlN+displaythick)).argmin()
        S_loc, A_loc = S.calculate_Poynting_Absorption_vs_z(zplot, E_out, H_out, R_loc)
        A[jj,ii,0] = A_loc[0,zMoO3]
        A[jj,ii,1] = A_loc[0,zAlN]-A_loc[0,zMoO3]
        A[jj,ii,2] = A_loc[0,zSiC]-A_loc[0,zAlN]
        
#%% Make pretty plot
rcParams.update({'xtick.direction':'in','ytick.direction':'in'})

psim, wnm = np.meshgrid(np.rad2deg(psi_v), f_sp/c_const*1e-2)

gs = gridspec.GridSpec(1,4, hspace=0, wspace=0)

fig = plt.figure(figsize=(8,4))
axR = fig.add_subplot(gs[0])
axR.pcolormesh(psim, wnm, R.T, shading='gouraud',
               vmin=0, vmax=1)
axR.set_xlabel('Euler angle $\psi$ (deg)')
axR.set_ylabel('Wavenumber (cm$^{-1}$)')

axA_MoO3 = fig.add_subplot(gs[1])
axA_MoO3.pcolormesh(psim, wnm, A[:,:,0].T, shading='gouraud',
                    vmin=0, vmax=1)
axA_MoO3.set_yticklabels([])

axA_AlN = fig.add_subplot(gs[2],sharex = axR, sharey=axA_MoO3)
axA_AlN.pcolormesh(psim, wnm, A[:,:,1].T, shading='gouraud',
                   vmin=0, vmax=1)

axA_SiC = fig.add_subplot(gs[3],sharex = axR, sharey=axA_MoO3)
axA_SiC.pcolormesh(psim, wnm, A[:,:,2].T, shading='gouraud',
                   vmin=0, vmax=1)

fig.tight_layout()
fig.show()

rcParams.update(rcParamsDefault)

