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
Define permittivity functions to be used in the GTM method.

Utility functions such as Drude or phonon models are defined at the end of 
the script. Some materials have different definitions according to the 
frequency range, you should always carefully check if they match your 
expectations !

All permittivities are expected to be calculated at frequency f in Hz.

**Change log:**
  *19-03-2020*:
   - Fixed various bugs in the definition of single resonance materials 
     (GaN, AlN, MoO3, InN, hBN) and added SiC without weak phonon modes
  *20-09-2019*:
   - Fixed database problems for some materials and some mistakes in the 
     permittivity functions. Commented out helper `print` functions. 
    
"""
import numpy as np
import os

## cache of values of epsilon loaded from file, so we load them only once
_eps_loaded = {}
## location of the package on the computer 
localpath = os.path.dirname(__file__)

c_const = 299792458 # m/s

def eps_KRS5(f):
    """
    Tabulated values for KRS5 material
    
    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)
    """
    wn_KRS5 = np.array([18518.5185185185,10000,6666.66666666667,5000,3333.33333333333,2500,2000,1666.66666666667,1428.57142857143,1250,1111.11111111111,1000,909.090909090909,833.333333333333,769.230769230769,714.285714285714,666.666666666667,625,588.235294117647,555.555555555556,526.315789473684,500,476.190476190476,454.545454545455,434.782608695652,416.666666666667,400,384.615384615385,370.370370370370,357.142857142857,344.827586206897,333.333333333333,322.580645161290,312.500000,303.030303030303,294.117647058824,285.714285714286,277.777777777778,270.270270270270,263.157894736842,256.410256410256,250])
    n_KRS5  = np.array([2.68059000,2.44620000,2.40774000,2.39498000,2.38574000,2.38204000,2.37979000,2.37797000,2.37627000,2.37452000,2.37267000,2.37069000,2.36854000,2.36622000,2.36371000,2.36101000,2.35812000,2.35502000,2.35173000,2.34822000,2.34451000,2.34058000,2.33643000,2.33206000,2.32746000,2.32264000,2.31758000,2.31229000,2.30676000,2.30098000,2.29495000,2.28867000,2.28212000,2.27531000,2.26823000,2.26087000,2.25322000,2.24528000,2.23705000,2.22850000,2.21965000,2.21047000])
    fn_KRS5 = wn_KRS5*c_const*1e2
    if (f<fn_KRS5.min()).any() or (f>fn_KRS5.max()).any():
        print('Beware: frequency out of tabulated data for KRS5')
    eps = np.interp(f,fn_KRS5,n_KRS5**2)
    return eps

def eps_SiCx(f): 
    """
    Silicon carbide (SiC) in-plane permittivity (x-y)

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)
    """
    return eps_SiC6Hx(f) ## same as below

def eps_SiCz(f): 
    """
    Silicon carbide (SiC) out-of-plane permittivity (z) without weak phonon modes

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)
    """
    wtSiCz = 783.67*c_const*1e2
    wlSiCz = 962.0*c_const*1e2
    epsinfSiCz = 6.78
    GtSiCz = 2.535*c_const*1e2*2
    GlSiCz = 2.535*c_const*1e2*2
    eps =  eps_1phonon(f,wtSiCz,wlSiCz,GtSiCz,GlSiCz,epsinfSiCz)
    return eps

def eps_SiC6Hx(f): 
    """
    6H-Silicon carbide (6H-SiC) in-plane permittivity (x-y)

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)
    """
    wtSiC = 794.78*c_const*1e2
    wlSiC = 967.93*c_const*1e2
    epsinfSiC = 6.56
    GtSiC = 2.535*c_const*1e2
    GlSiC = 2.535*c_const*1e2
    eps =  eps_1phonon(f,wtSiC,wlSiC,GtSiC,GlSiC,epsinfSiC)
    return eps

def eps_SiC6Hz(f):
    """
    6H-Silicon carbide (6H-SiC) out-of-plane permittivity (z) with weak phonon modes

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)
    """
    wtSiCz = 783.67*c_const*1e2
    wlSiCz = 962.0*c_const*1e2
    epsinfSiCz = 6.78
    GtSiCz = 2.535*c_const*1e2*2
    GlSiCz = 2.535*c_const*1e2*2
    wTw1 = 881.0*c_const*1e2
    wLw1 = 881.2*c_const*1e2
    gTw  = 1.3*c_const*1e2*2
    wTw2 = 886*c_const*1e2
    wLw2 = 886.1333*c_const*1e2
    eps =  eps_3phonon(f,wtSiCz,wlSiCz,GtSiCz,GlSiCz,wTw1,wLw1,wTw2,wLw2,gTw,epsinfSiCz)
    return eps

def eps_GaNx(f): 
    """
    Gallium nitride (GaN) ordinary axis permittivity (without excitons)

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)
    """
    wlGaN      = 742.1*c_const*1e2
    wtGaN     = 560.1*c_const*1e2
    GtGaN     = 4.0*c_const*1e2
    GlGaN     = 4.0*c_const*1e2
    epsinfGaN = 5.04
    eps =  eps_1phonon(f,wtGaN,wlGaN,GtGaN,GlGaN,epsinfGaN)
    return eps
def eps_GaNz(f): # GaN extraordinary epsilon
    """
    Gallium nitride (GaN) extraordinary axis permittivity (without excitons)

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)
    """
    wlGaNz     = 732.5*c_const*1e2
    wtGaNz     = 537*c_const*1e2
    GtGaNz     = 4.0*c_const*1e2
    GlGaNz     = 4.0*c_const*1e2
    epsinfGaNz = 5.01
    eps =  eps_1phonon(f,wtGaNz,wlGaNz,GtGaNz,GlGaNz,epsinfGaNz)
    return eps

def eps_AlNx(f):
    """
    Aluminium nitride (AlN) ordinary axis permittivity (without excitons)

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)
    """
    wlAlN     = 909.6*c_const*1e2
    wtAlN     = 667.2*c_const*1e2
    GtAlN     = 2.2*c_const*1e2
    epsinfAlN = 4.160
    eps = eps_Lorentz(f, wtAlN, wlAlN, GtAlN, epsinfAlN)
    return eps
def eps_AlNz(f):
    """
    Aliuminium nitride (AlN) extraordinary axis permittivity (without excitons)

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)
    """
    wlAlNz     = 888.9*c_const*1e2
    wtAlNz     = 608.5*c_const*1e2
    GtAlNz     = 2.2*c_const*1e2
    epsinfAlNz = 4.350
    eps = eps_Lorentz(f, wtAlNz, wlAlNz, GtAlNz, epsinfAlNz)
    return eps

def eps_hBNx(f):
    """
    Hexagonal boron nitride (hBN) ordinary axis permittivity (without excitons)

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)
    """
    wlhBN      = 1614*c_const*1e2
    wthBN      = 1360*c_const*1e2
    GthBN      = 7*c_const*1e2
    epsinfhBN  = 4.9
    eps = eps_Lorentz(f, wthBN, wlhBN, GthBN, epsinfhBN)
    return eps
def eps_hBNz(f):
    """
    Hexagonal boron nitride (hBN) extraordinary axis permittivity (without excitons)

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)
    """
    wlhBNz      = 811**c_const*1e2
    wthBNz      = 760*c_const*1e2
    GthBNz      = 1*c_const*1e2
    epsinfhBNz  = 2.95    
    eps = eps_Lorentz(f, wthBNz, wlhBNz, GthBNz, epsinfhBNz)
    return eps

def eps_InNx(f):
    """
    Indium nitride (InN) ordinary axis permittivity (without excitons)

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)
    """
    wlInN       = 593*c_const*1e2
    wtInN       = 476*c_const*1e2
    GtInN       = 4.4*c_const*1e2
    epsinfInN   = 8.45*c_const*1e2
    eps = eps_Lorentz(f, wtInN, wlInN, GtInN, epsinfInN)
    return eps
def eps_InNz(f):
    """
    Indium nitride (InN) extraordinary axis permittivity (without excitons)

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)
    """
    wlInNz      = 586*c_const*1e2
    wtInNz      = 447*c_const*1e2
    GtInNz       = 4.4*c_const*1e2
    epsinfInNz  = 8.35*c_const*1e2
    eps = eps_Lorentz(f, wtInNz, wlInNz, GtInNz, epsinfInNz)
    return eps

def eps_MoO3x(f):
    """
    Molybdenum oxide (MoO3) first axis permittivity

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)
    """
    wtMoO3x     = 820*c_const*1e2
    wlMoO3x     = 972*c_const*1e2
    GMoO3x      = 4*c_const*1e2
    epsinfMoO3x = 4.0
    eps = eps_Lorentz(f, wtMoO3x, wlMoO3x, GMoO3x, epsinfMoO3x)
    return eps
def eps_MoO3y(f):
    """
    Molybdenum oxide (MoO3) second axis permittivity

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)
    """
    wtMoO3y     = 545*c_const*1e2
    wlMoO3y     = 851*c_const*1e2
    GMoO3y      = 4*c_const*1e2
    epsinfMoO3y = 5.2
    eps = eps_Lorentz(f, wtMoO3y, wlMoO3y, GMoO3y, epsinfMoO3y)
    return eps
def eps_MoO3z(f):
    """
    Molybdenum oxide (MoO3) third axis permittivity

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)
    """
    wtMoO3z     = 958*c_const*1e2
    wlMoO3z     = 1004*c_const*1e2
    GMoO3z      = 2*c_const*1e2    
    epsinfMoO3z = 2.4
    eps = eps_Lorentz(f, wtMoO3z, wlMoO3z, GMoO3z, epsinfMoO3z)
    return eps

def eps_GaAs(f):
    """
    Gallium arsenide (GaAs) permittivity (without excitons)

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)
    """
    wlGaAs      = 292*c_const*1e2
    wtGaAs     = 268*c_const*1e2
    GtGaAs     = 0.8*c_const*1e2
    GlGaAs     = 0.8*c_const*1e2
    epsinfGaAs = 10.9
    eps =  eps_1phonon(f,wtGaAs,wlGaAs,GtGaAs,GlGaAs,epsinfGaAs)
    return eps

def eps_GaP(f):
    """
    Gallium phosphide (GaP) permittivity (without excitons)

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)
    """
    wlGaP      = 402*c_const*1e2
    wtGaP     = 366.3*c_const*1e2
    GtGaP     = 1.1*c_const*1e2
    GlGaP     = 1.1*c_const*1e2
    epsinfGaP = 9.1
    eps =  eps_1phonon(f,wtGaP,wlGaP,GtGaP,GlGaP,epsinfGaP)
    return eps

def eps_InAs(f):
    """
    Indium arsenide (InAs) permittivity (without excitons)

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)
    """
    wlInAs     = 243*c_const*1e2
    wtInAs     = 218*c_const*1e2
    GtInAs     = 2.5*c_const*1e2
    GlInAs     = 2.5*c_const*1e2
    epsinfInAs = 12.9
    eps =  eps_1phonon(f,wtInAs,wlInAs,GtInAs,GlInAs,epsinfInAs)
    return eps

def eps_Al2O3o(f):
    """
    Aluminium oxide (Al2O3) ordinary axis permittivity 

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)
    """
    # ordinary axis
    wTOo = np.array([385,442,569,635])*c_const*1e2
    wLOo = np.array([388,480,625,900])*c_const*1e2
    gTOo = np.array([3.3,3.1,4.7,5.0])*c_const*1e2
    gLOo = np.array([3.1,1.9,5.9,14.7])*c_const*1e2
    epsinfo = 3.077
    if isinstance(f, (int, float)):
        f = np.array([f])
    epso = epsinfo*np.ones(len(f))
    for ko in range(len(wTOo)):
        epso = epso*eps_1phonon(f,wTOo[ko], wLOo[ko], gTOo[ko], gLOo[ko], 1.0)
    return epso
def eps_Al2O3e(f):
    """
    Aluminium oxide (Al2O3) extraordinary axis permittivity 

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)
    """
    # extraordinary axis
    wTOe = np.array([400,583])*c_const*1e2
    wLOe = np.array([512,871])*c_const*1e2
    gTOe = np.array([5.3,1.1])*c_const*1e2
    gLOe = np.array([3.0,15.4])*c_const*1e2
    epsinfe = 3.072
    if isinstance(f, (int, float)):
        f = np.array([f])
    epse = epsinfe*np.ones(len(f))
    for ke in range(len(wTOe)):
        epse = epse*eps_1phonon(f,wTOe[ke], wLOe[ke], gTOe[ke], gLOe[ke], 1.0)
    return epse

def eps_BaTiO3(f):
    """
    Barium titanate (BaTiO3) permittivity 

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)
    """
    wTOo = np.array([22.043,179.053,247.65,496.055])*c_const*1e2
    wLOo = np.array([50.8132,183.666,466.771,723.765])*c_const*1e2
    gTOo = np.array([47.1166,3.39954,155.258,45.1082])*c_const*1e2
    gLOo = np.array([184.371,12.7144,14.6996,39.0983])*c_const*1e2
    epsinfo = 2.356
    if isinstance(f, (int, float)):
        f = np.array([f])
    epso = epsinfo*np.ones(len(f))
    for ko in range(len(wTOo)):
        epso = epso*eps_1phonon(f,wTOo[ko], wLOo[ko], gTOo[ko], gLOo[ko], 1.0)
    return epso

def eps_SiN(f):
    """
    Low-stress silicon nitride (SiN) permittivity in the far and mid-infrared.

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)

    Typically for SiN_x PECVD deposited layers, from 
    `Cataldo et al., Optics Letters 37, 4200 (2017) <https://arxiv.org/pdf/1209.2987.pdf>`_
    """
    omega = f*(2*np.pi)
    epsj = np.array([7.582+0.0*1.0j,6.754+0.3759*1.0j,6.601+0.0041*1.0j,5.43+0.1179*1.0j,
                     4.601+0.2073*1.0j,4.562+0.0124*1.0j],dtype=complex)
    omegaj = np.array([13.913,15.053,24.521,26.44,31.724])*2*np.pi
    gammaj = np.array([5.810,6.436,2.751,3.482,5.948])*2*np.pi
    alphaj = np.array([0.0001,0.3427,0.0006,0.0002,0.008])
    gammapj = np.array([gammaj[ii]*np.exp(-alphaj[ii]*((omegaj[ii]**2-omega**2)/(omega*gammaj[ii]))**2) for ii in range(len(gammaj))])
    SumTerm = 0
    for ii in range(len(gammaj)):
        SumTerm += (epsj[ii]-epsj[ii+1])*omegaj[ii]**2/(omegaj[ii]**2-omega**2-1.0j*omega*gammapj[ii])
    eps = epsj[-1]+SumTerm
    return eps

def eps_Au(f):
    """
    Gold (Au) permittivity. 

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)

    **Attention**, two models are used: a simple Drude model with parameters 
    from `Derkachova et al., Plasmonics 11, 941 (2016) (open access) 
    <http://doi.org/10.1007/s11468-015-0128-7>`_ 
    or the tabulated data from Jonhson and Christy 
    (`refractiveindex.info <https://refractiveindex.info/?shelf=main&book=Au&page=Johnson>`_)
    You should check carefully if this works out for you. 
    """
    fmin_JC = c_const/1.93e-6
    fmax_JC = c_const/0.188e-6
    if (f>fmin_JC) and (f<fmax_JC):
        #print('Using Jonhson and Christy data for Au')
        epsAufile = os.path.join(localpath,'MaterialData/Au_Johnson_nk.csv')
        if epsAufile not in _eps_loaded:
            #print('Loaded Au permittivity, Johnson and Christy')
            _eps_loaded[epsAufile] = np.genfromtxt(epsAufile, delimiter='\t',
                                       skip_header=1, unpack=True)
        lbda = _eps_loaded[epsAufile][0,:]*1e-6
        n = _eps_loaded[epsAufile][1,:]
        k = _eps_loaded[epsAufile][2,:]
        eps1 = n**2-k**2
        eps2 = 2*n*k
        floc = c_const/lbda
        epsr = np.interp(f, floc[::-1], eps1[::-1])
        epsi = np.interp(f, floc[::-1], eps2[::-1])
        return epsr+1.0j*epsi
    else:
        #print('Using a simple Drude model for Au')
        ## Parameters from Derkachova et al., Plasmonics 11, 941 (2016)
        # 10.1007/s11468-015-0128-7
        fp = 2.183e15
        gammap = 1.7410e13
        epsinf = 9.84
        return eps_drude(f,fp,gammap, epsinf)

def eps_Ag(f):
    """
    Silver (Ag) permittivity. 

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)

    **Attention**, two models are used: a simple Drude model or tabulated data. 
    Both are taken from `Yang et al., Phys. Rev. B 91, 235137 (2015) 
    <http://doi.org/10.1103/PhysRevB.91.235137>`_
    You should check carefully if this works out for you. 
    """
    fmin = c_const/1.93e-6
    fmax = c_const/0.188e-6
    if (f>fmin) and (f<fmax):
        #print('Using Yang data for Ag')
        epsAgfile = os.path.join(localpath,'MaterialData/Ag_Yang_nk.csv')
        if epsAgfile not in _eps_loaded:
            #print('Loaded Ag permittivity, Yang')
            _eps_loaded[epsAgfile] = np.genfromtxt(epsAgfile, delimiter='\t',
                                       skip_header=1, unpack=True)
        lbda = _eps_loaded[epsAgfile][0,:]*1e-6
        n = _eps_loaded[epsAgfile][1,:]
        k = _eps_loaded[epsAgfile][2,:]
        eps1 = n**2-k**2
        eps2 = 2*n*k
        floc = c_const/lbda
        epsr = np.interp(f, floc[::-1], eps1[::-1])
        epsi = np.interp(f, floc[::-1], eps2[::-1])
        return epsr+1.0j*epsi
    else:
        #print('Using a simple Drude model for Ag')
        # from the same paper, simple Drude model (not extended)
        fp = 2.152e15
        gammap = 1/17e-15 # 17 fs
        epsinf = 5
        return eps_drude(f,fp,gammap, epsinf)
    
def eps_BaF2(f):
    """
    Barium fluoride (BaF2) refractive index.

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)

    Originally from `Querry, Contractor Report CRDEC-CR-88009 (1987) 
    <https://apps.dtic.mil/docs/citations/ADA192210>`_
    downloaded from `RefractiveIndex.info <http://refractiveindex.info>`_ (2019)
    """
    fmin = c_const/166.6e-6
    fmax = c_const/0.22e-6
    if (f<fmin) or (f>fmax):
        print('Beware: frequency out of tabulated data for BaF2')
    epsBaF2file = os.path.join(localpath,'MaterialData/BaF2_Querry_nk.csv')
    if epsBaF2file not in _eps_loaded:
        #print('Loaded BaF2 permittivity')
        _eps_loaded[epsBaF2file] = np.genfromtxt(epsBaF2file, delimiter='\t',
                                   skip_header=1, unpack=True)
    lbda = _eps_loaded[epsBaF2file][0,:]*1e-6
    n = _eps_loaded[epsBaF2file][1,:]
    k = _eps_loaded[epsBaF2file][2,:]
    eps1 = n**2-k**2
    eps2 = 2*n*k
    floc = c_const/lbda
    epsr = np.interp(f, floc[::-1], eps1[::-1])
    epsi = np.interp(f, floc[::-1], eps2[::-1])
    eps = epsr+1.0j*epsi
    return eps

def eps_CaF2(f):
    """
    Calcium fluoride (CaF2) refractive index.

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)

    Originally from `Malitson, Applied Optics 2, 1103 (1963) 
    <https://doi.org/10.1364/AO.2.001103>`_
    downloaded from `RefractiveIndex.info <http://refractiveindex.info>`_ (2019)
    """
    fmin = c_const/166.6e-6
    fmax = c_const/0.22e-6
    if (f<fmin) or (f>fmax):
        print('Beware: frequency out of tabulated data for CaF2')
    epsCaF2file = os.path.join(localpath,'MaterialData/CaF2_Malitson_n.csv')
    if epsCaF2file not in _eps_loaded:
        #print('Loaded CaF2 permittivity')
        _eps_loaded[epsCaF2file] = np.genfromtxt(epsCaF2file, delimiter='\t',
                                   skip_header=1, unpack=True)
    lbda = _eps_loaded[epsCaF2file][0,:]*1e-6
    n = _eps_loaded[epsCaF2file][1,:]
    k = _eps_loaded[epsCaF2file][2,:]
    eps1 = n**2-k**2
    eps2 = 2*n*k
    floc = c_const/lbda
    epsr = np.interp(f, floc[::-1], eps1[::-1])
    epsi = np.interp(f, floc[::-1], eps2[::-1])
    eps = epsr+1.0j*epsi
    return eps


def eps_SiO2(f):
    """
    Silicium dioxide (SiO2) refractive index.

    :param array f: frequency (array or float)
    :return: permittivity (float or len(f)-array)

    Originally from Popova, Opt. Spectrosc. 33, 444 (1972)
    and and Malitson, J. Opt. Soc. Am. 55, 1205 (1965)
    downloaded from `RefractiveIndex.info <http://refractiveindex.info>`_ (2019)
    """
    fmin_P = c_const/50e-6
    fmax_P = c_const/7e-6
    fmin_M = c_const/7e-6
    fmax_M = c_const/0.21e-6
    if (f<fmin_P) or (f>fmax_M):
        print('Beware: frequency out of tabulated data for SiO2')
    epsSiO2file_P = os.path.join(localpath,'MaterialData/SiO2_Popova_nk.csv')
    epsSiO2file_M = os.path.join(localpath,'MaterialData/SiO2_Malitson_n.csv')
    if f>fmin_M:
        epsSiO2file = epsSiO2file_M
    else:
        epsSiO2file = epsSiO2file_P
    if epsSiO2file not in _eps_loaded:
        #print('Loaded SiO2 permittivity')
        _eps_loaded[epsSiO2file] = np.genfromtxt(epsSiO2file, delimiter='\t',
                                   skip_header=1, unpack=True)
    lbda = _eps_loaded[epsSiO2file][0,:]*1e-6
    n = _eps_loaded[epsSiO2file][1,:]
    if epsSiO2file == epsSiO2file_P:
        k = _eps_loaded[epsSiO2file][2,:]
    else:
        k = np.zeros(len(n))
    eps1 = n**2-k**2
    eps2 = 2*n*k
    floc = c_const/lbda
    epsr = np.interp(f, floc[::-1], eps1[::-1])
    epsi = np.interp(f, floc[::-1], eps2[::-1])
    eps = epsr+1.0j*epsi
    return eps
    
#%% generic material functions

def eps_drude(f, fp, gammap, epsinf=1.0):
    """
    Drude-like permittivity
    
    :param array f: frequency (Hz)
    :param float fp: plasma frequency (Hz)
    :param float gammap: mean collision rate (Hz)
    :param float epsinf: high frequency permittivity
    :return: complex permittivity
    """
    eps = epsinf-fp**2/(f**2+1.0j*gammap*f)
    return eps

def eps_Lorentz(f, fT, fL, gammaT, eps_inf):
    """
    Single Lorentz oscillator model
    
    :param array f: frequency (Hz)
    :param float fT: frequency of the transverse phonon
    :param float fL: frequency of the longitudinal phonon
    :param float gammaT: damping rate of the transverse phonon
    :param float epsinf: high-frequency permittivity
    """
    return eps_1phonon(f, fT, fL, gammaT, gammaT, eps_inf)
    
def eps_1phonon(f, fT, fL, gammaT, gammaL, eps_inf):
    """
    1 phonon permittivity model
    
    :param array f: frequency (Hz)
    :param float fT: frequency of the transverse phonon
    :param float fL: frequency of the longitudinal phonon
    :param float gammaT: damping rate of the transverse phonon
    :param float gammaL: damping rate of the longitudinal phonon    
    :param float epsinf: high-frequency permittivity
    :return: complex permittivity
    """
    eps = eps_inf*(f**2-fL**2+1.0j*gammaL*f)/(f**2-fT**2+1.0j*gammaT*f)
    return eps

def eps_3phonon(f, fT, fL, gammaT, gammaL, 
                fT1, fL1, fT2, fL2,
                gammaTw, eps_inf):
    """
    3 phonons permittivity model
    
    :param array f: frequency (Hz)
    :param float fT: frequency of the transverse phonon
    :param float fL: frequency of the longitudinal phonon
    :param float gammaT: relaxation rate of the transverse phonon
    :param float gammaL: relaxation rate of the longitudinal phonon    
    :param float fT1: frequency of the weak transverse phonon 1
    :param float fL1: frequency of the weak longitudinal phonon 1
    :param float fT2: frequency of the weak transverse phonon 2
    :param float fL2: frequency of the weak longitudinal phonon 2
    :param float gammaTw: damping of the transverse phonon
    :param float epsinf: high-frequency permittivity
    :return: complex permittivity
    """
    eps = (f**2-fL**2)/(f**2-fT**2+1.0j*gammaT*f)
    eps += (fT1**2-fL1**2)/(f**2-fT1**2+1.0j*gammaTw*f)
    eps += (fT2**2-fL2**2)/(f**2-fT2**2+1.0j*gammaTw*f)
    eps = eps_inf*(1.0+eps)
    return eps

