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
# Copyright (C) Mathieu Jeannin 2019 2023 <mathieu.jeannin@c2n.upsaclay.fr>.
"""
@author: Mathieu Jeannin

Example file for the pyGTM package

It reproduces the main result of section IV in Passler, Jeannin and Paarman
https://arxiv.org/abs/2002.03832
fig 4
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import GTM.GTMcore as GTM
import GTM.Permittivities as mat
from matplotlib import rcParams, rcParamsDefault

### Physical constants
c_const = 3e8
eV = 1.6e-19
m0 = 9.11e-31 ## electron rest mass
meff = 0.202*m0
eps0 = 8.854e-12 ## vacuum permittivity

## Import the refractive index of the doped QW
fimp, eps1zz, eps2zz = np.genfromtxt('EpsQW_zz.txt', delimiter='\t',
                                     skip_header=1, unpack=True)
fimp, eps1plane, eps2plane = np.genfromtxt('EpsQW_plane.txt', delimiter='\t',
                                           skip_header=1, unpack=True)
def epsQW_zz(f):
    eps1 = np.interp(f,fimp, eps1zz)
    eps2 = np.interp(f,fimp, eps2zz)
    return eps1+1.0j*eps2

def epsQW_plane(f):
    eps1 = np.interp(f,fimp, eps1plane)
    eps2 = np.interp(f,fimp, eps2plane)
    return eps1+1.0j*eps2


#%% Setup simulation

###### General options
f_sp = np.linspace(c_const/11e-6, c_const/4e-6, 50) ## frequency points
theta = np.arange(0.0, 80.0, 5)/180.*np.pi ## angles

Nqw = 20 ## Number of QWs
tQW = 3e-9 ## thickness of the QW
tBarr = 10.0*1e-9 ## Barrier thickness
Lcav = 1e-6 ## total length of the cavity
LAR = Nqw*tQW+(Nqw+1)*tBarr ## length of the active region // superlattice
Lspac = Lcav-LAR ## length of the spacer
tmirr = 0.5e-6 ## mirror thickness

### Doped mirror: a bulky, doped semiconductor layer
DopMirr = 5e20*1e6 # doping in m-3
f_P_mirr = np.sqrt(DopMirr*eV**2/(meff*eps0*5.35))/(2.*np.pi) # plasma frequency
epsmirr = lambda x: mat.eps_drude(x, f_P_mirr, 1./500e-15, epsinf=5.35)

## Barriers have a permittivity interpolated from their constituents
epsBarrx = lambda x: 0.26*mat.eps_AlNx(x)+mat.eps_GaNx(x)*(1.0-0.26)
epsBarrz = lambda x: 0.26*mat.eps_AlNz(x)+mat.eps_GaNz(x)*(1.0-0.26)

#################### 
######### System
#### Layers definition
S_empty = GTM.System() ## empty cavity system
S_sl = GTM.System() ## superlattice system
QW = GTM.Layer(thickness=tQW, epsilon1=epsQW_plane,epsilon3=epsQW_zz) ## QW layer
barrier = GTM.Layer(thickness=tBarr, epsilon1=epsBarrx, epsilon3=epsBarrz) ## barrier layer
## GaN spacer filling the cavity with superlattice
GaNspacer_sl = GTM.Layer(thickness=Lspac, epsilon1=mat.eps_GaNx, epsilon3=mat.eps_GaNz) 
## GaN cavity empty of AR
GaNspacer_empty = GTM.Layer(thickness=Lcav, epsilon1=mat.eps_GaNx, epsilon3=mat.eps_GaNz) 
mirror = GTM.Layer(thickness=tmirr, epsilon1=epsmirr) ## doped GaN mirror

### superstrate is pure GaN
sup = GTM.Layer(thickness=0.5e-6, epsilon1=mat.eps_GaNx, epsilon3=mat.eps_GaNz)
# substrate is a gold mirror
sub = GTM.Layer(epsilon1=mat.eps_Au, thickness=200e-9)

##### building up both systems
## Empty system
S_empty.set_superstrate(sup)
S_empty.set_substrate(sub)
S_empty.add_layer(mirror)
S_empty.add_layer(GaNspacer_empty)

## superlattice system
S_sl.set_superstrate(sup)
S_sl.set_substrate(sub)
S_sl.add_layer(mirror)
S_sl.add_layer(GaNspacer_sl)
for nl in range(Nqw):
    S_sl.add_layer(barrier)
    S_sl.add_layer(QW)
S_sl.add_layer(barrier)

#%% Calculate reflectivity and absorption

R_empty = np.zeros((len(theta), len(f_sp)))
R_sl = np.zeros((len(theta), len(f_sp)))
A_lay_empty = np.zeros((len(theta), len(f_sp), 2))
A_lay_sl = np.zeros((len(theta), len(f_sp), 3))

for ii, thi in enumerate(theta):
    print(ii/len(theta))

    for jj, fj in enumerate(f_sp):

        for S in [S_empty, S_sl]:
            S.initialize_sys(fj) ## sets the epsilons in the layers
            zeta_sys = np.sin(thi)*np.sqrt(S.superstrate.epsilon[0,0]) ## in-plane wavevector
            Sys_Gamma = S.calculate_GammaStar(fj, zeta_sys) ## main transfer matrix calculation
            ## E field calculation is necessary to get layer resolved absorption
            ## no resolution or z-vector given: computation only at the layers interfaces
            zplot, E_out, H_out, zn_plot = S.calculate_Efield(fj, zeta_sys, 
                                                              magnetic=True)
        ## Interfaces of interest
        zmirr = np.abs(zplot-tmirr).argmin() ## doped mirror
        zspac = np.abs(zplot-(tmirr+Lspac)).argmin() ## spacer
        zsl = np.abs(zplot-(tmirr+Lspac+Nqw*tQW+Nqw*tBarr)).argmin() ## superlattice
        zAu = zplot.argmax() ## substrate
        
        r, R_loc, t, T_loc = S_empty.calculate_r_t(zeta_sys) ## reflectivity 
        R_empty[ii,jj] = R_loc[0] # p-pol only
        ## Poynting vector and absorption. **Requires R**
        S_loc, A_loc = S_empty.calculate_Poynting_Absorption_vs_z(zplot, E_out, H_out, R_loc)
        A_lay_empty[ii,jj,0] = A_loc[0,zmirr] # p-pol only
        A_lay_empty[ii,jj,1] = A_loc[0,zAu]-A_loc[0,zmirr] # p-pol only
        
        r, R_loc, t, T_loc = S_sl.calculate_r_t(zeta_sys) ## reflectivity
        R_sl[ii,jj] = R_loc[0]
        ## Poynting vector and absorption. **Requires R**
        S_loc, A_loc = S_sl.calculate_Poynting_Absorption_vs_z(zplot, E_out, H_out, R_loc)
        A_lay_sl[ii,jj,0] = A_loc[0,zmirr]
        A_lay_sl[ii,jj,1] = A_loc[0,zsl]-A_loc[0,zmirr]
        A_lay_sl[ii,jj,2] = A_loc[0,zAu]-A_loc[0,zsl]


#%% Make pretty plot
rcParams.update({'xtick.direction':'in','ytick.direction':'in'})

gs = gridspec.GridSpec(2,4, hspace=0, wspace=0)
thm, fm = np.meshgrid(theta,f_sp/1.0e12)


fig = plt.figure()
axR_empty = fig.add_subplot(gs[0,0])
axR_empty.pcolormesh(thm*180./np.pi, fm, R_empty.T,
                     vmin=0, vmax=1, shading='gouraud')
axR_empty.set_xticklabels([])
axR_empty.set_ylabel('Frequency (THz)')
axR_empty.text(0.1, 0.05, r'$R^p$', transform = axR_empty.transAxes)

axA_mirr_empty = fig.add_subplot(gs[0,1], sharex=axR_empty)
axA_mirr_empty.pcolormesh(thm*180./np.pi, fm, A_lay_empty[:,:,0].T,
                      vmin=0, vmax=1, shading='gouraud')
axA_mirr_empty.set_yticklabels([])
axA_mirr_empty.text(0.1, 0.05, r'$A^p_{mirror}$', transform = axA_mirr_empty.transAxes,
                    color='w')

axA_Au_empty = fig.add_subplot(gs[0,3], sharex=axR_empty)
axA_Au_empty.pcolormesh(thm*180./np.pi, fm, A_lay_empty[:,:,1].T,
                      vmin=0, vmax=1, shading='gouraud')
axA_Au_empty.set_yticklabels([])
axA_Au_empty.text(0.1, 0.05, r'$A^p_{Au}$', transform = axA_Au_empty.transAxes,
                  color='w')


axR_sl = fig.add_subplot(gs[1,0], sharey = axR_empty)
axR_sl.pcolormesh(thm*180./np.pi, fm, R_sl.T,
                   vmin=0, vmax=1, shading='gouraud')
axR_sl.set_xlabel(r'$\theta$ (deg)')
axR_sl.set_ylabel('Frequency (THz)')
axR_sl.set_xticks(np.arange(0,90,20))
axR_sl.text(0.1, 0.05, r'$R^p$', transform = axR_sl.transAxes)

axA_mirr_sl = fig.add_subplot(gs[1,1], sharex=axR_sl)
axA_mirr_sl.pcolormesh(thm*180./np.pi, fm, A_lay_sl[:,:,0].T,
                      vmin=0, vmax=1, shading='gouraud')
axA_mirr_sl.set_xlabel(r'$\theta$ (deg)')
axA_mirr_sl.set_yticklabels([])
axA_mirr_sl.text(0.1, 0.05, r'$A^p_{mirror}$', transform = axA_mirr_sl.transAxes,
                 color='w')

axA_sl_sl = fig.add_subplot(gs[1,2], sharex=axR_sl)
axA_sl_sl.pcolormesh(thm*180./np.pi, fm, A_lay_sl[:,:,1].T,
                      vmin=0, vmax=1, shading='gouraud')
axA_sl_sl.set_yticklabels([])
axA_sl_sl.set_xlabel(r'$\theta$ (deg)')
axA_sl_sl.text(0.1, 0.05, r'$A^p_{QWs}$', transform = axA_sl_sl.transAxes,
               color='w')

axA_Au_sl = fig.add_subplot(gs[1,3], sharex=axR_sl)
axA_Au_sl.pcolormesh(thm*180./np.pi, fm, A_lay_sl[:,:,2].T,
                      vmin=0, vmax=1, shading='gouraud')
axA_Au_sl.set_xlabel(r'$\theta$ (deg)')
axA_Au_sl.set_yticklabels([])
axA_Au_sl.text(0.1, 0.05, r'$A^p_{Au}$', transform = axA_Au_sl.transAxes,
               color='w')


fig.tight_layout()
fig.show()

# revert to usual setting
rcParams.update(rcParamsDefault)
