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
# Copyright (C) Mathieu Jeannin 2019-2021
# <mathieu.jeannin@c2n.upsaclay.fr>
# <math.jeannin@free.fr>.

"""
This module implements the generalized 4x4 transfer matrix (GTM) method
poposed in `Passler, N. C. and Paarmann, A., JOSA B 34, 2128 (2017)
<http://doi.org/10.1364/JOSAB.34.002128>`_
and corrected in
`JOSA B 36, 3246 (2019) <http://doi.org/10.1364/JOSAB.36.003246>`_,
as well as the layer-resolved absorption proposed in
`Passler, Jeannin and Paarman <https://arxiv.org/abs/2002.03832>`_.
This code uses inputs from D. Dietze's FSRStools library
https://github.com/ddietze/FSRStools

Please cite the relevant associated publications if you use this code. 

Author: 
    - Mathieu Jeannin mathieu.jeannin@c2n.upsaclay.fr  math.jeannin@free.fr (permanent)

Affiliations: 
    - Laboratoire de Physique de l'Ecole Normale Superieure (2019)
    - Centre de Nanosciences et Nanotechnologies (2020-2021)

Layers are represented by the :py:class:`Layer` class that holds all parameters
describing the optical properties of a single layer.
The optical system is assembled using the :py:class:`System` class.


**Change log:**

    *15-10-2021*:

        - Fixed rounding error bug in lag.eig() causing the program to crash randomly
        for negligibly small imaginary parts of the wavevectors

        - Corrected a sign error in gamma32 that lead to field discontinuities

    *19-03-2020*:

        - Adapted the code to compute the layer-resolved absorption as proposed
          by Passler et al. (https://arxiv.org/abs/2002.03832), using
          :py:func:`System.calculate_Poynting_Absorption_vs_z`.

        - Include the correct calculation of intensity transmission coefficients
          in :py:func:`System.calculate_r_t`.
          **This BREAKS compatibility** with the previous definition of the function.

        - Corrected bugs in :py:func:`System.calculate_Efield`
          and added magnetic field option

        - Adapted :py:func:`System.calculate_Efield` to allow hand-defined,
          irregular grid and a shorthand to compute only at layers interfaces. 

    *20-09-2019*:
        - Added functions in the :py:class:`System` class to compute in-plane
          wavevector of guided modes and dispersion relation for such guided surface modes.
          This is *highly prospective* as it depends on the robustness of the minimization
          procedure (or the lack of thereoff)
"""

import numpy as np
import numpy.linalg as lag
#from scipy.optimize import minimize
from .helpers import exact_inv

c_const = 299792458  # m/s
eps0 = 8.854e-12  # vacuum permittivity
qsd_thr = 1e-10  # threshold for wavevector comparison
zero_thr = 1e-10  # threshold for eigenvalue comparison to zero


class System:
    """
    System class.

    Attributes
    ----------
    theta : float
            Angle of incidence, in radians
    substrate : Layer
            The substrate layer. Defaults to vacuum (empty layer instance)
    superstrate : Layer
            The superstrate layer, defaults to vacuum (empty layer instance)
    layers : list of layers
            list of the layers in the system

    Notes
    -----
    Layers can be added and removed (not inserted).

    The whole system's transfer matrix is computed using :py:func:`calculate_GammaStar`,
    which calls :py:func:`Layer.update` for each layer.
    General reflection and transmission coefficient functions are given, they require prior
    execution of :py:func:`calculate_GammaStar`.
    The electric fields can be visualized in the case of incident plane wave
    using :py:func:`calculate_Efield`

    """
    def __init__(self, structure):

        self.structure = structure

    # from Layer
    def calculate_layer_matrices(self, layer, zeta):
        """
        Calculate the principal matrices necessary for the GTM algorithm.

        Parameters
        ----------
        zeta : complex
             In-plane reduced wavevector kx/k0 in the system.

        Returns
        -------
             None

        Notes
        -----
        Note that zeta is conserved through the whole system and set externaly
        using the angle of incidence and `System.superstrate.epsilon[0,0]` value

        Requires prior execution of :py:func:`calculate_epsilon`

        """

        M = np.zeros((6, 6), dtype=np.complex128)  # constitutive relations
        a = np.zeros((6, 6), dtype=np.complex128)
        S = np.zeros((4, 4), dtype=np.complex128)
        Delta = np.zeros((4, 4), dtype=np.complex128)

        # Constitutive matrix (see e.g. eqn (4))
        M[0:3, 0:3] = layer.epsilon.copy()
        M[3:6, 3:6] = layer.mu*np.identity(3)

        # from eqn (10)
        b = M[2, 2]*M[5, 5] - M[2, 5]*M[5, 2]

        # a matrix from eqn (9)
        a[2, 0] = (M[5, 0]*M[2, 5] - M[2, 0]*M[5, 5])/b
        a[2, 1] = ((M[5, 1]-zeta)*M[2, 5] - M[2, 1]*M[5, 5])/b
        a[2, 3] = (M[5, 3]*M[2, 5] - M[2, 3]*M[5, 5])/b
        a[2, 4] = (M[5, 4]*M[2, 5] - (M[2, 4]+zeta)*M[5, 5])/b
        a[5, 0] = (M[5, 2]*M[2, 0] - M[2, 2]*M[5, 0])/b
        a[5, 1] = (M[5, 2]*M[2, 1] - M[2, 2]*(M[5, 1]-zeta))/b
        a[5, 3] = (M[5, 2]*M[2, 3] - M[2, 2]*M[5, 3])/b
        a[5, 4] = (M[5, 2]*(M[2, 4]+zeta) - M[2, 2]*M[5, 4])/b

        # S Matrix (Don't know where it comes from since Delta is just S re-ordered)
        # Note that after this only Delta is used
        S[0, 0] = M[0, 0] + M[0, 2]*a[2, 0] + M[0, 5]*a[5, 0]
        S[0, 1] = M[0, 1] + M[0, 2]*a[2, 1] + M[0, 5]*a[5, 1]
        S[0, 2] = M[0, 3] + M[0, 2]*a[2, 3] + M[0, 5]*a[5, 3]
        S[0, 3] = M[0, 4] + M[0, 2]*a[2, 4] + M[0, 5]*a[5, 4]
        S[1, 0] = M[1, 0] + M[1, 2]*a[2, 0] + (M[1, 5]-zeta)*a[5, 0]
        S[1, 1] = M[1, 1] + M[1, 2]*a[2, 1] + (M[1, 5]-zeta)*a[5, 1]
        S[1, 2] = M[1, 3] + M[1, 2]*a[2, 3] + (M[1, 5]-zeta)*a[5, 3]
        S[1, 3] = M[1, 4] + M[1, 2]*a[2, 4] + (M[1, 5]-zeta)*a[5, 4]
        S[2, 0] = M[3, 0] + M[3, 2]*a[2, 0] + M[3, 5]*a[5, 0]
        S[2, 1] = M[3, 1] + M[3, 2]*a[2, 1] + M[3, 5]*a[5, 1]
        S[2, 2] = M[3, 3] + M[3, 2]*a[2, 3] + M[3, 5]*a[5, 3]
        S[2, 3] = M[3, 4] + M[3, 2]*a[2, 4] + M[3, 5]*a[5, 4]
        S[3, 0] = M[4, 0] + (M[4, 2]+zeta)*a[2, 0] + M[4, 5]*a[5, 0]
        S[3, 1] = M[4, 1] + (M[4, 2]+zeta)*a[2, 1] + M[4, 5]*a[5, 1]
        S[3, 2] = M[4, 3] + (M[4, 2]+zeta)*a[2, 3] + M[4, 5]*a[5, 3]
        S[3, 3] = M[4, 4] + (M[4, 2]+zeta)*a[2, 4] + M[4, 5]*a[5, 4]

        # Delta Matrix from eqn (8)
        Delta[0, 0] = S[3, 0]
        Delta[0, 1] = S[3, 3]
        Delta[0, 2] = S[3, 1]
        Delta[0, 3] = - S[3, 2]
        Delta[1, 0] = S[0, 0]
        Delta[1, 1] = S[0, 3]
        Delta[1, 2] = S[0, 1]
        Delta[1, 3] = - S[0, 2]
        Delta[2, 0] = -S[2, 0]
        Delta[2, 1] = -S[2, 3]
        Delta[2, 2] = -S[2, 1]
        Delta[2, 3] = S[2, 2]
        Delta[3, 0] = S[1, 0]
        Delta[3, 1] = S[1, 3]
        Delta[3, 2] = S[1, 1]
        Delta[3, 3] = -S[1, 2]

        return M, a, b, S, Delta

    def calculate_layer_q(self, layer, zeta):
        """
        Calculates the 4 out-of-plane wavevectors for the current layer.

        Returns
        -------
        None

        Notes
        -----
        From this we also get the Poynting vectors.
        Wavevectors are sorted according to (trans-p, trans-s, refl-p, refl-s)
        Birefringence is determined according to a threshold value `qsd_thr` 
        set at the beginning of the script.
        """

        M, a, b, S, Delta = self.calculate_layer_matrices(layer, zeta)

        qs = np.zeros(4, dtype=np.complex128)  # out of plane wavevector
        Py = np.zeros((3, 4), dtype=np.complex128)  # Poyting vector
        # Stores the Berreman modes, used for birefringent layers
        Berreman = np.zeros((4, 3), dtype=np.complex128)
        Berreman_unsorted = np.zeros((4, 3), dtype=np.complex128)

        Delta_loc = np.zeros((4, 4), dtype=np.complex128)
        transmode = np.zeros((2), dtype=np.int64)
        reflmode = np.zeros((2), dtype=np.int64)

        Delta_loc = Delta.copy()
        # eigenvals // eigenvects as of eqn (11)
        qsunsorted, psiunsorted = lag.eig(Delta_loc)
        # remove extremely small real/imaginary parts that are due to
        # numerical inaccuracy
        for km in range(4):
            if ((np.abs(np.imag(qsunsorted[km])) > 0) and
                    (np.abs(np.imag(qsunsorted[km])) < zero_thr)):
                qsunsorted[km] = np.real(qsunsorted[km]) + 0.0j
            if ((np.abs(np.real(qsunsorted[km])) > 0) and
                    (np.abs(np.real(qsunsorted[km])) < zero_thr)):
                qsunsorted[km] = 0.0 + 1.0j*np.imag(qsunsorted[km])
        for comp in range(4):
            if ((np.abs(np.real(psiunsorted[km][comp])) > 0) and
                    (np.abs(np.real(psiunsorted[km][comp])) < zero_thr)):
                psiunsorted[km][comp] = 0.0 + 1.0j*np.imag(psiunsorted[km][comp])
            if ((np.abs(np.imag(psiunsorted[km][comp])) > 0) and
                    (np.abs(np.imag(psiunsorted[km][comp])) < zero_thr)):
                psiunsorted[km][comp] = np.real(psiunsorted[km][comp]) + 0.0j

        kt = 0
        kr = 0
        # sort berremann qi's according to (12)
        if any(np.abs(np.imag(qsunsorted))):
            for km in range(0, 4):
                if np.imag(qsunsorted[km]) >= 0:
                    transmode[kt] = km
                    kt += 1
                else:
                    reflmode[kr] = km
                    kr += 1
        else:
            for km in range(0, 4):
                if np.real(qsunsorted[km]) > 0:
                    transmode[kt] = km
                    kt += 1
                else:
                    reflmode[kr] = km
                    kr += 1
        # Calculate the Poyting vector for each Psi using (16-18)
        for km in range(0, 4):
            Ex = psiunsorted[0, km]
            Ey = psiunsorted[2, km]
            Hx = -psiunsorted[3, km]
            Hy = psiunsorted[1, km]
            # from eqn (17)
            Ez = a[2, 0]*Ex + a[2, 1]*Ey + a[2, 3]*Hx + a[2, 4]*Hy
            # from eqn (18)
            Hz = a[5, 0]*Ex + a[5, 1]*Ey + a[5, 3]*Hx + a[5, 4]*Hy
            # and from (16)
            Py[0, km] = Ey*Hz-Ez*Hy
            Py[1, km] = Ez*Hx-Ex*Hz
            Py[2, km] = Ex*Hy-Ey*Hx
            # Berreman modes (unsorted) in case they are needed later (birefringence)
            Berreman_unsorted[km, 0] = Ex
            Berreman_unsorted[km, 1] = Ey
            Berreman_unsorted[km, 2] = Ez
        # check Cp using either the Poynting vector for birefringent
        # materials or the electric field vector for non-birefringent
        # media to sort the modes

        # first calculate Cp for transmitted waves
        Cp_t1 = np.abs(Py[0, transmode[0]])**2/(np.abs(Py[0, transmode[0]])**2
                                                + np.abs(Py[1, transmode[0]])**2)
        Cp_t2 = np.abs(Py[0, transmode[1]])**2/(np.abs(Py[0, transmode[1]])**2
                                                + np.abs(Py[1, transmode[1]])**2)

        if np.abs(Cp_t1-Cp_t2) > qsd_thr:  # birefringence
            # sets _useBerreman fo the calculation of gamma matrix below
            layer._useBerreman = True
            if Cp_t2 > Cp_t1:
                transmode = np.flip(transmode, 0)  # flip the two values
            # then calculate for reflected waves if necessary
            Cp_r1 = np.abs(Py[0, reflmode[1]])**2/(np.abs(Py[0, reflmode[1]])**2
                                                   + np.abs(Py[1, reflmode[1]])**2)
            Cp_r2 = np.abs(Py[0, reflmode[0]])**2/(np.abs(Py[0, reflmode[0]])**2
                                                   + np.abs(Py[1, reflmode[0]])**2)
            if Cp_r1 > Cp_r2:
                reflmode = np.flip(reflmode, 0)  # flip the two values

        else:  # No birefringence, use the Electric field s-pol/p-pol
            Cp_te1 = np.abs(psiunsorted[0, transmode[1]])**2/(np.abs(psiunsorted[0, transmode[1]])**2
                                                              + np.abs(psiunsorted[2, transmode[1]])**2)
            Cp_te2 = np.abs(psiunsorted[0, transmode[0]])**2/(np.abs(psiunsorted[0, transmode[0]])**2
                                                              + np.abs(psiunsorted[2, transmode[0]])**2)
            if Cp_te1>Cp_te2:
                transmode = np.flip(transmode,0) ## flip the two values
            Cp_re1 = np.abs(psiunsorted[0, reflmode[1]])**2/(np.abs(psiunsorted[0, reflmode[1]])**2
                                                             + np.abs(psiunsorted[2, reflmode[1]])**2)
            Cp_re2 = np.abs(psiunsorted[0, reflmode[0]])**2/(np.abs(psiunsorted[0, reflmode[0]])**2
                                                             + np.abs(psiunsorted[2, reflmode[0]])**2)
            if Cp_re1>Cp_re2:
                reflmode = np.flip(reflmode, 0)  # flip the two values

        # finaly store the sorted version
        # q is (trans-p, trans-s, refl-p, refl-s)
        qs[0] = qsunsorted[transmode[0]]
        qs[1] = qsunsorted[transmode[1]]
        qs[2] = qsunsorted[reflmode[0]]
        qs[3] = qsunsorted[reflmode[1]]
        Py_temp = Py.copy()
        Py[:, 0] = Py_temp[:, transmode[0]]
        Py[:, 1] = Py_temp[:, transmode[1]]
        Py[:, 2] = Py_temp[:, reflmode[0]]
        Py[:, 3] = Py_temp[:, reflmode[1]]
        # Store the (sorted) Berreman modes
        Berreman[0] = Berreman_unsorted[transmode[0], :]
        Berreman[1] = Berreman_unsorted[transmode[1], :]
        Berreman[2] = Berreman_unsorted[reflmode[0], :]
        Berreman[3] = Berreman_unsorted[reflmode[1], :]

        return qs, Py, Berreman

    def calculate_layer_gamma(self, layer, zeta):
        """
        Calculate the gamma matrix

        Parameters
        ----------
        zeta : complex
             in-plane reduced wavevector kx/k0

        Returns
        -------
        None
        """
        qs, Py, Berreman = self.calculate_layer_q(layer, zeta)

        gamma = np.zeros((4, 3), dtype=np.complex128)

        # this whole function is eqn (20)
        gamma[0, 0] = 1.0 + 0.0j
        gamma[1, 1] = 1.0 + 0.0j
        gamma[3, 1] = 1.0 + 0.0j
        gamma[2, 0] = -1.0 + 0.0j

        # convenience definition of the repetitive factor
        mu_eps33_zeta2 = (layer.mu*layer.epsilon[2, 2]-zeta**2)

        if np.abs(qs[0]-qs[1]) < qsd_thr:
            gamma12 = 0.0 + 0.0j

            gamma13 = -(layer.mu*layer.epsilon[2, 0]+zeta*qs[0])/mu_eps33_zeta2

            gamma21 = 0.0 + 0.0j

            gamma23 = -layer.mu*layer.epsilon[2, 1]/mu_eps33_zeta2
        else:
            gamma12_num = layer.mu*layer.epsilon[1, 2]*(layer.mu*layer.epsilon[2, 0]+zeta*qs[0])
            gamma12_num = gamma12_num - layer.mu*layer.epsilon[1, 0]*mu_eps33_zeta2
            gamma12_denom = mu_eps33_zeta2*(layer.mu*layer.epsilon[1, 1]-zeta**2-qs[0]**2)
            gamma12_denom = gamma12_denom - layer.mu**2*layer.epsilon[1, 2]*layer.epsilon[2, 1]
            gamma12 = gamma12_num/gamma12_denom
            if np.isnan(gamma12):
                gamma12 = 0.0 + 0.0j

            gamma13 = -(layer.mu*layer.epsilon[2, 0]+zeta*qs[0])
            gamma13 = gamma13-layer.mu*layer.epsilon[2, 1]*gamma12
            gamma13 = gamma13/mu_eps33_zeta2

            if np.isnan(gamma13):
                gamma13 = -(layer.mu*layer.epsilon[2, 0]+zeta*qs[0])
                gamma13 = gamma13/mu_eps33_zeta2

            gamma21_num = layer.mu*layer.epsilon[2, 1]*(layer.mu*layer.epsilon[0, 2]+zeta*qs[1])
            gamma21_num = gamma21_num-layer.mu*layer.epsilon[0, 1]*mu_eps33_zeta2
            gamma21_denom = mu_eps33_zeta2*(layer.mu*layer.epsilon[0, 0]-qs[1]**2)
            gamma21_denom = gamma21_denom-(layer.mu*layer.epsilon[0, 2]+zeta*qs[1])*(layer.mu*layer.epsilon[2, 0]+zeta*qs[1])
            gamma21 = gamma21_num/gamma21_denom
            if np.isnan(gamma21):
                gamma21 = 0.0+0.0j

            gamma23 = -(layer.mu*layer.epsilon[2, 0] +zeta*qs[1])*gamma21-layer.mu*layer.epsilon[2, 1]
            gamma23 = gamma23/mu_eps33_zeta2
            if np.isnan(gamma23):
                gamma23 = -layer.mu*layer.epsilon[2, 1]/mu_eps33_zeta2

        if np.abs(qs[2]-qs[3]) < qsd_thr:
            gamma32 = 0.0 + 0.0j
            gamma33 = (layer.mu*layer.epsilon[2, 0]+zeta*qs[2])/mu_eps33_zeta2
            gamma41 = 0.0 + 0.0j
            gamma43 = -layer.mu*layer.epsilon[2, 1]/mu_eps33_zeta2
        else:
            gamma32_num = layer.mu*layer.epsilon[1, 0]*mu_eps33_zeta2
            gamma32_num = gamma32_num-layer.mu*layer.epsilon[1, 2]*(layer.mu*layer.epsilon[2, 0]+zeta*qs[2])
            gamma32_denom = mu_eps33_zeta2*(layer.mu*layer.epsilon[1, 1]-zeta**2-qs[2]**2)
            gamma32_denom = gamma32_denom-layer.mu**2*layer.epsilon[1, 2]*layer.epsilon[2, 1]
            gamma32 = gamma32_num/gamma32_denom
            if np.isnan(gamma32):
                gamma32 = 0.0 + 0.0j

            gamma33 = layer.mu*layer.epsilon[2, 0] + zeta*qs[2]
            gamma33 = gamma33 + layer.mu*layer.epsilon[2, 1]*gamma32
            gamma33 = gamma33/mu_eps33_zeta2
            if np.isnan(gamma33):
                gamma33 = (layer.mu*layer.epsilon[2, 0] + zeta*qs[2])/mu_eps33_zeta2

            gamma41_num = layer.mu*layer.epsilon[2, 1]*(layer.mu*layer.epsilon[0, 2]+zeta*qs[3])
            gamma41_num = gamma41_num - layer.mu*layer.epsilon[0, 1]*mu_eps33_zeta2
            gamma41_denom = mu_eps33_zeta2*(layer.mu*layer.epsilon[0, 0]-qs[3]**2)
            gamma41_denom = gamma41_denom - (layer.mu*layer.epsilon[0, 2]
                                             + zeta*qs[3])*(layer.mu*layer.epsilon[2 ,0]+zeta*qs[3])
            gamma41 = gamma41_num/gamma41_denom
            if np.isnan(gamma41):
                gamma41 = 0.0 + 0.0j

            gamma43 = -(layer.mu*layer.epsilon[2, 0]+zeta*qs[3])*gamma41
            gamma43 = gamma43-layer.mu*layer.epsilon[2, 1]
            gamma43 = gamma43/mu_eps33_zeta2
            if np.isnan(gamma43):
                gamma43 = -layer.mu*layer.epsilon[2, 1]/mu_eps33_zeta2

        # gamma field vectors should be normalized to avoid any birefringence problems
        # use double square bracket notation to ensure correct array shape
        gamma1 = np.array([[gamma[0, 0], gamma12, gamma13]],dtype=np.complex128)
        gamma2 = np.array([[gamma21, gamma[1, 1], gamma23]],dtype=np.complex128)
        gamma3 = np.array([[gamma[2, 0], gamma32, gamma33]],dtype=np.complex128)
        gamma4 = np.array([[gamma41, gamma[3, 1], gamma43]],dtype=np.complex128)

        # Regular case, no birefringence, we keep the Xu fields
        gamma[0, :] = gamma1/lag.norm(gamma1)
        gamma[1, :] = gamma2/lag.norm(gamma2)
        gamma[2, :] = gamma3/lag.norm(gamma3)
        gamma[3, :] = gamma4/lag.norm(gamma4)

        # In case of birefringence, use Berreman fields
        if layer.useBerreman:
            for ki in range(4):
                # normalize them first
                Berreman[ki] = Berreman[ki]/lag.norm(Berreman[ki])

            print('replaced gamma by Berreman')
            gamma = Berreman

        return gamma, qs

    def calculate_layer_transfer_matrix(self, layer, f, zeta):
        """
        Compute the transfer matrix of the whole layer :math:`T_i=A_iP_iA_i^{-1}`

        Parameters
        ----------
        f : float
            frequency (in Hz)
        zeta : complex
               reduced in-plane wavevector kx/k0
        Returns
        -------
        None

        """

        layer.calculate_epsilon(f)

        gamma, qs = self.calculate_layer_gamma(layer, zeta)

        Ai = np.zeros((4, 4), dtype=np.complex128)
        Ki = np.zeros((4, 4), dtype=np.complex128)
        Ti = np.zeros((4, 4), dtype=np.complex128)  # Layer transfer matrix

        # eqn(22)
        Ai[0, :] = gamma[:, 0].copy()
        Ai[1, :] = gamma[:, 1].copy()

        Ai[2, :] = (qs*gamma[:, 0]-zeta*gamma[:, 2])/layer.mu
        Ai[3, :] = qs*gamma[:, 1]/layer.mu

        for ii in range(4):
            # looks a lot like eqn (25). Why is K not Pi ?
            Ki[ii, ii] = np.exp(-1.0j*(2.0*np.pi*f*qs[ii]*layer.thick)/c_const)

        Ai_inv = exact_inv(Ai.copy())
        # eqn (26)
        Ti = np.matmul(Ai, np.matmul(Ki, Ai_inv))

        return Ai, Ki, Ai_inv, Ti

    def calculate_GammaStar(self, f, zeta_sys):
        """
        Calculate the whole system's transfer matrix.

        Parameters
        -----------
        f : float
            Frequency (Hz)
        zeta_sys : complex
            In-plane wavevector kx/k0

        Returns
        -------
        GammaStar: 4x4 complex matrix
                   System transfer matrix :math:`\Gamma^{*}`
        """

        Ai_super, Ki_super, Ai_inv_super, T_super = self.calculate_layer_transfer_matrix(
            self.structure.superstrate, f, zeta_sys)
        Ai_sub, Ki_sub, Ai_inv_sub, T_sub = self.calculate_layer_transfer_matrix(
            self.structure.substrate, f, zeta_sys)

        Delta1234 = np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1]])

        Gamma = np.zeros(4, dtype=np.complex128)
        GammaStar = np.zeros(4, dtype=np.complex128)

        Tloc = np.identity(4, dtype=np.complex128)

        for ii in range(len(self.structure.layers))[::-1]:
            Ai, Ki, Ai_inv, T_ii = self.calculate_layer_transfer_matrix(
                self.structure.layers[ii], f, zeta_sys)
            Tloc = np.matmul(T_ii, Tloc)

        Gamma = np.matmul(Ai_inv_super, np.matmul(Tloc, Ai_sub))
        GammaStar = np.matmul(exact_inv(Delta1234),
                              np.matmul(Gamma, Delta1234))

        return GammaStar

    def calculate_r_t(self, zeta_sys, GammaStar):
        """ Calculate various field and intensity reflection and transmission
        coefficients, as well as the 4-valued vector of transmitted field.

        Parameters
        -----------
        zeta_sys : complex
            Incident in-plane wavevector
        Returns
        -------
        r_out : len(4)-array
                Complex *field* reflection coefficients r_out=([rpp,rps,rss,rsp])
        R_out : len(4)-array
                Real *intensity* reflection coefficients R_out=([Rpp,Rss,Rsp,Tps])
        t_out : len(4)-array
                Complex *field* transmition coefficients t=([tpp, tps, tsp, tss])
        T_out : len(4)-array
                Real *intensity* transmition coefficients T_out=([Tp,Ts]) (mode-inselective)

        Notes
        -----
        **IMPORTANT**
        ..version 19-03-2020:
        All intensity coefficients are now well defined. Transmission is defined 
        mode-independently. It could be defined mode-dependently for non-birefringent
        substrates in future versions.
        The new definition of this function **BREAKS compatibility** with the previous
        one.

        ..version 13-09-2019:
        Note that the field reflectivity and transmission coefficients
        r and t are well defined. The intensity reflection coefficient is also correct.
        However, the intensity transmission coefficients T are ill-defined so far.
        This will be corrected upon future publication of the correct intensity coefficients.

        Note also the different ordering of the coefficients, for consistency w/ Passler's matlab code

        """

        # common denominator for all coefficients
        Denom = GammaStar[0, 0]*GammaStar[2, 2]-GammaStar[0, 2]*GammaStar[2, 0]
        # field reflection coefficients
        rpp = GammaStar[1, 0]*GammaStar[2, 2]-GammaStar[1, 2]*GammaStar[2, 0]
        rpp = np.nan_to_num(rpp/Denom)

        rss = GammaStar[0, 0]*GammaStar[3, 2]-GammaStar[3, 0]*GammaStar[0, 2]
        rss = np.nan_to_num(rss/Denom)

        rps = GammaStar[3, 0]*GammaStar[2, 2]-GammaStar[3, 2]*GammaStar[2, 0]
        rps = np.nan_to_num(rps/Denom)

        rsp = GammaStar[0, 0]*GammaStar[1, 2]-GammaStar[1, 0]*GammaStar[0, 2]
        rsp = np.nan_to_num(rsp/Denom)

        # Intensity reflection coefficients are just square moduli
        Rpp = np.abs(rpp)**2
        Rss = np.abs(rss)**2
        Rps = np.abs(rps)**2
        Rsp = np.abs(rsp)**2
        r_out = np.array([rpp, rps, rss, rsp])  # order matching Passler Matlab code
        R_out = np.array([Rpp, Rss, Rsp, Rps])  # order matching Passler Matlab code

        # field transmission coefficients
        # t_field = np.zeros(4, dtype=np.complex128)
        t_out = np.zeros(4, dtype=np.complex128)
        tpp = np.nan_to_num(GammaStar[2, 2]/Denom)
        tss = np.nan_to_num(GammaStar[0, 0]/Denom)
        tps = np.nan_to_num(-GammaStar[2, 0]/Denom)
        tsp = np.nan_to_num(-GammaStar[0, 2]/Denom)
        t_out = np.array([tpp, tps, tsp, tss])
        # t_field = np.array([tpp, tps, tsp, tss])

        # Intensity transmission requires Poyting vector analysis
        # N.B: could be done mode-dependentely later
        # start with the superstrate
        # Incident fields are either p or s polarized
        ksup = np.zeros((4, 3), dtype=np.complex128)  # wavevector in superstrate
        ksup[:, 0] = zeta_sys

        gamma_sup, qs_sup = self.calculate_layer_gamma(self.structure.superstrate, zeta_sys)
        gamma_sub, qs_sub = self.calculate_layer_gamma(self.structure.substrate, zeta_sys)

        for ii, qi in enumerate(qs_sup):
            ksup[ii,2] = qi
        ksup = ksup/c_const  # omega simplifies in the H field formula
        Einc_pin = gamma_sup[0, :]  # p-pol incident electric field
        Einc_sin = gamma_sup[1, :]  # s-pol incident electric field
        # Poyting vector in superstrate (incident, p-in and s-in)
        Sinc_pin = 0.5*np.real(np.cross(Einc_pin, np.conj(np.cross(ksup[0, :], Einc_pin))))
        Sinc_sin = 0.5*np.real(np.cross(Einc_sin, np.conj(np.cross(ksup[1, :], Einc_sin))))

        # Substrate Poyting vector
        # Outgoing fields (eqn 17)
        Eout_pin = t_out[0]*gamma_sub[0,:]+t_out[1]*gamma_sub[1,:]  # p-in, p or s out
        Eout_sin = t_out[2]*gamma_sub[0,:]+t_out[3]*gamma_sub[1,:]  # s-in, p or s out
        ksub = np.zeros((4,3), dtype=np.complex128)
        ksub[:, 0] = zeta_sys
        for ii, qi in enumerate(qs_sub):
            ksub[ii, 2] = qi
        ksub = ksub/c_const  # omega simplifies in the H field formula

        # outgoing Poyting vectors, 2 formulations
        Sout_pin = 0.5*np.real(np.cross(Eout_pin, np.conj(np.cross(ksub[0, :], Eout_pin))))
        Sout_sin = 0.5*np.real(np.cross(Eout_sin, np.conj(np.cross(ksub[1, :], Eout_sin))))
        # Intensity transmission coefficients are only the z-component of S !
        T_pp = (Sout_pin[2]/Sinc_pin[2])  # z-component only
        T_ss = (Sout_sin[2]/Sinc_sin[2])  # z-component only

        T_out = np.array([T_pp, T_ss])

        return r_out, R_out, t_out, T_out

    def calculate_Efield(self, f, zeta_sys, z_vect=None, x=0.0,
                         magnetic=False, dz=None):
        """
        Calculate the electric field profiles for both s-pol and p-pol excitation.

        Parameters
        ----------
        f : float
            frequency (Hz)
        zeta_sys : complex
            in-plane normalized wavevector kx/k0
        z_vect : 1Darray
            Coordinates at which the calculation is done. if None, the layers boundaries are used.
        x : float or 1D array
            x-coordinates for (future) 2D plot of the electric field. Not yet implemented
        magnetic : bool
            Boolean to skip or compute the magnetic field vector
        dz : float (optional)
            Space resolution along propagation (z) axis. Superseed z_vect

        Returns
        --------
        z : 1Darray
            1D array of z-coordinates according to dz
        E_out : (len(z),3)-Array
            Total electric field in the structure
        H_out (opt): (len(z),3)-Array
            Total magnetic field in the structure
        zn : list
            Positions of the different interfaces

        Notes
        -----
        ..Version 19-03-2020:
            changed keywords to add z_vect
            z_vect is used for either minimal computation (using get_layers_boundaries)
            or hand-defined z-positions (e.g. irregular spacing for improved resolution)
            if dz is given, a regular grid is used.
            A sketch of the definition of all fields and algorithm is supplied in the module,
            to better get a grasp on where Fft and Fbk are defined.
        ..Version 28-01-2020:
            Added Magnetic field keyword to save time.
            Poyting and absorption defined in a separate function
        ..Version 06-01-2020:
            Added Magnetic field and Poyting vector.
        ..Version 13-09-2019:
            the 2D field profile is not implemented yet. x should be left to default

        """

        GammaStar = self.calculate_GammaStar(f, zeta_sys)
        # r_out, R_out, t_field, t_out, T_out = self.calculate_r_t()
        r_out, R_out, t, T = self.calculate_r_t(zeta_sys, GammaStar)

        # Nb of layers
        laynum = len(self.structure.layers)
        zn = np.zeros(laynum+2)  # superstrate+layers+substrate

        # 4-components field tensor at the front and
        # back interfaces of the layer
        # correspond to E0 and E1
        # defined by (37*)
        # E0 (E^(p/o)_t, E^(s/e)_t, E^(p/o)_r, E^(s/e)_r)
        # twice for p-pol in and s-pol in
        F_ft = np.zeros((laynum+2, 8), dtype=np.complex128)
        # E1 (E^(p/o)_t, E^(s/e)_t, E^(p/o)_r, E^(s/e)_r)
        # twice for p-pol in and s-pol in
        F_bk = np.zeros((laynum+2, 8), dtype=np.complex128)

        zn[-1] = 0.0  # initially with the substrate

        # First step of the algorithm starts from the top of the substrate
        # a sketch is provided to better visualize the steps
        # red quantities in sketch
        # (37*) with p-pol excitation
        F_ft[-1, 0] = t[0]  # t_pp
        F_ft[-1, 1] = t[1]  # t_ps
        # (37*) with s-pol excitation
        F_ft[-1, 4] = t[2]  # t_sp
        F_ft[-1, 5] = t[3]  # t_ss

        # propagate to the "end" of the substrate
        # F_bk[-1] for plot purpose (see Fig. 1.(a))

        Ai, Ki_sub, _, _ = self.calculate_layer_transfer_matrix(
            self.structure.substrate, f, zeta_sys)
        F_bk[-1, :4] = np.matmul(exact_inv(Ki_sub), F_ft[-1, :4])
        F_bk[-1, 4:] = np.matmul(exact_inv(Ki_sub), F_ft[-1, 4:])

        if laynum > 0:
            # First layer is a special case to handle System.substrate
            # purple quantities in sketch
            zn[-2] = zn[-1]-self.structure.substrate.thick

            Aim1, Kim1, _, _ = self.calculate_layer_transfer_matrix(
                self.structure.layers[-1], f, zeta_sys)

            Li = np.matmul(exact_inv(Aim1), Ai)
            F_bk[-2, :4] = np.matmul(Li, F_ft[-1, :4])
            F_bk[-2, 4:] = np.matmul(Li, F_ft[-1, 4:])
            F_ft[-2, :4] = np.matmul(Kim1, F_bk[-2, :4])
            F_ft[-2, 4:] = np.matmul(Kim1, F_bk[-2, 4:])

            # From here we start recursively computing the fields
            # blue quantities in sketch
            for kl in range(1, laynum)[::-1]:
                # subtract the thickness (building thickness array backwards)
                zn[kl] = zn[kl+1]-self.structure.layers[kl].thick

                Aim1, Kim1, _, _ = self.calculate_layer_transfer_matrix(
                    self.structure.layers[kl-1], f, zeta_sys)
                Ai, _, _, _ = self.calculate_layer_transfer_matrix(
                    self.structure.layers[kl], f, zeta_sys)

                Li = np.matmul(exact_inv(Aim1), Ai)
                # F_ft == E0  //  F_bk == E1
                F_bk[kl, :4] = np.matmul(Li, F_ft[kl+1, :4])
                F_bk[kl, 4:] = np.matmul(Li, F_ft[kl+1, 4:])
                F_ft[kl, :4] = np.matmul(Kim1, F_bk[kl, :4])
                F_ft[kl, 4:] = np.matmul(Kim1, F_bk[kl, 4:])

            zn[0] = zn[1]-self.structure.layers[0].thick

            Aim1, Ki_sup, _, _ = self.calculate_layer_transfer_matrix(
                self.structure.superstrate, f, zeta_sys)
            Ai, _, _, _ = self.calculate_layer_transfer_matrix(
                self.structure.layers[0], f, zeta_sys)

            Li = np.matmul(exact_inv(Aim1), Ai)
            # F_ft == E0  //  F_bk == E1
            F_bk[0, :4] = np.matmul(Li, F_ft[1, :4])
            F_bk[0, 4:] = np.matmul(Li, F_ft[1, 4:])
            F_ft[0, :4] = np.matmul(Ki_sup, F_bk[0, :4])
            F_ft[0, 4:] = np.matmul(Ki_sup, F_bk[0, 4:])
        else:
            zn[0] = -self.structure.substrate.thick
            Ai, Ki_sub, _, _ = self.calculate_layer_transfer_matrix(
                self.structure.substrate, f, zeta_sys)
            Aim1, Ki_sup, _, _ = self.calculate_layer_transfer_matrix(
                self.structure.superstrate, f, zeta_sys)
            Li = np.matmul(exact_inv(Aim1), Ai)
            # F_ft == E0  //  F_bk == E1
            F_bk[0, :4] = np.matmul(Li, F_ft[1, :4])
            F_bk[0, 4:] = np.matmul(Li, F_ft[1, 4:])
            F_ft[0, :4] = np.matmul(Ki_sup, F_bk[0, :4])
            F_ft[0, 4:] = np.matmul(Ki_sup, F_bk[0, 4:])

        # shift everything so that incident boundary is at z=0
        zn = zn-zn[0]

        # define the spatial points where the computation is performed
        if dz is None:
            # print('No dz given, \n')
            if z_vect is None:
                # print('Resorting to minimal computation on boundaries')
                z = self.structure.get_layers_boundaries()
            else:
                print('using manually given z-vector')
                z = z_vect
        else:
            # print('using dz=%.2e'%(dz))
            z = np.arange(-self.structure.superstrate.thick, zn[-1], dz)

        # 2x4 component field tensor E_prop propagated from front surface
        Eprop = np.empty((8), dtype=np.complex128)
        # 4-component field tensor F_tens for each direction and polarization
        F_tens = np.zeros((24, len(z)), dtype=np.complex128)
        if magnetic is True:
            H_tens = np.zeros((24, len(z)), dtype=np.complex128)
        # final component electric field E_out = (E_x, Ey, Ez)
        # for p-pol and s-pol excitation
        E_out = np.zeros((6, len(z)), dtype=np.complex128)
        if magnetic is True:
            H_out = np.zeros((6, len(z)), dtype=np.complex128)
        # Elementary propagation
        dKiz = np.zeros((4, 4), dtype=np.complex128)

        # starting from the superstrate:
        current_layer = 0
        L = self.structure.superstrate
        for ii, zc in enumerate(z):  # enumerates returns a tuple (index, value)

            if zc > zn[current_layer]:
                # change the layer
                # important to count here until laynum+1 to get the correct zn
                # in the substrate for dKiz
                current_layer += 1

                if current_layer == laynum+1:  # reached substrate
                    L = self.structure.substrate
                else:
                    L = self.structure.layers[current_layer-1]

            gamma, qs = self.calculate_layer_gamma(L, zeta_sys)

            for kk in range(4):
                # use the conjugate of the K matrix => exp(+1.0j...)
                dKiz[kk, kk] = np.exp(1.0j*(2.0*np.pi*f*qs[kk]*(zc-zn[current_layer]))/c_const)

            # Eprop propagated from front surface to back of next layer
            # n.b: unclear why using F_bk and not F_ft works... but it works !
            Eprop[:4] = np.matmul(dKiz, F_bk[current_layer, :4])
            Eprop[4:] = np.matmul(dKiz, F_bk[current_layer, 4:])

            # wave vector for each mode in layer L 
            k_lay = np.zeros((4, 3), dtype=np.complex128)
            k_lay[:, 0] = zeta_sys
            for jj, qj in enumerate(qs):
                k_lay[jj, 2] = qj
            # no normalization by c_const eases the visualization of H
            # k_lay = k_lay/(c_const) ## omega simplifies in the H field formula

            # p-pol in 
            # forward, o/p
            F_tens[:3, ii] = Eprop[0]*gamma[0, :]
            if magnetic is True:
                H_tens[:3, ii] = (1./L.mu)*np.cross(k_lay[0, :], F_tens[:3, ii])
            # forward, e/s
            F_tens[3:6, ii] = Eprop[1]*gamma[1, :]
            if magnetic is True:
                H_tens[3:6, ii] = (1./L.mu)*np.cross(k_lay[1, :], F_tens[3:6, ii])
            # backward, o/p
            F_tens[6:9, ii] = Eprop[2]*gamma[2, :]
            if magnetic is True:
                H_tens[6:9, ii] = (1./L.mu)*np.cross(k_lay[2, :], F_tens[6:9, ii])
            # backward, e/s
            F_tens[9:12, ii] = Eprop[3]*gamma[3, :]
            if magnetic is True:
                H_tens[9:12, ii] = (1./L.mu)*np.cross(k_lay[3, :], F_tens[9:12, ii])
            # s-pol in 
            # forward, o/p
            F_tens[12:15, ii] = Eprop[4]*gamma[0, :]
            if magnetic is True:
                H_tens[12:15, ii] = (1./L.mu)*np.cross(k_lay[0, :], F_tens[12:15, ii])
            # forward, e/s
            F_tens[15:18, ii] = Eprop[5]*gamma[1, :]
            if magnetic is True:
                H_tens[15:18, ii] = (1./L.mu)*np.cross(k_lay[1, :], F_tens[15:18, ii])
            # backward, o/p
            F_tens[18:21, ii] = Eprop[6]*gamma[2, :]
            if magnetic is True:
                H_tens[18:21, ii] = (1./L.mu)*np.cross(k_lay[2, :], F_tens[18:21, ii])
            # backward, e/s
            F_tens[21:, ii] = Eprop[7]*gamma[3, :]
            if magnetic is True:
                H_tens[21:, ii] = (1./L.mu)*np.cross(k_lay[3, :], F_tens[21:, ii])

            # Total electric field (note that sign flip for
            # backward propagation is already in gamma)
            # p in 
            E_out[:3, ii] = F_tens[:3, ii]+F_tens[3:6, ii]+F_tens[6:9, ii]+F_tens[9:12, ii]
            if magnetic is True:
                H_out[:3, ii] = H_tens[:3, ii]+H_tens[3:6, ii]+H_tens[6:9, ii]+H_tens[9:12, ii]
            # s in
            E_out[3:6, ii] = F_tens[12:15, ii]+F_tens[15:18, ii]+F_tens[18:21, ii]+F_tens[21:, ii]
            if magnetic is True:
                H_out[3:6, ii] = H_tens[12:15, ii]+H_tens[15:18, ii]+H_tens[18:21, ii]+H_tens[21:, ii]
        if magnetic is True:
            return z, E_out, H_out, zn[:-1]  # last interface is useless, substrate=infinite
        else:
            return z, E_out, zn[:-1]  # last interface is useless, substrate=infinite

    def calculate_Poynting_Absorption_vs_z(self, z, E, H, R):
        """
        Calculate the z-dependent Poynting vector and cumulated absorption.

        Parameters
        ----------
        z : 1Darray
            Spatial coordinate for the fields
        E : 1Darray
            6-components Electric field vector (p- or s- in) along z
        H : 1Darray
            6-components Magnetic field vector (p- or s- in) along z
        R : len(4)-array
            Reflectivity from :py:func:`calculate_r_t`
        S_out : 6xlen(z) array
            6 components (p//s) Poyting vector along z
        A_out : 2xlen(z)
            2 components (p//s) absorption along z
        """
        S_out = np.zeros((6, len(z)))  # Poynting vector
        A_out = np.zeros((2, len(z)))  # z-dependent absorption

        # S=0.5*Re(ExB)
        S_out[:3, :] = 0.5*np.real(np.cross(E[:3, :], np.conj(H[:3, :]),
                                 axisa=0, axisb=0, axisc=0))
        S_out[3:6, :] = 0.5*np.real(np.cross(E[3:6, :], np.conj(H[3:6, :]),
                                 axisa=0, axisb=0, axisc=0))

        z1 = np.abs(z).argmin()+1  # index where z>0, first interface
        Tp_z = S_out[2, :]/S_out[2, 0]*(1.0-(R[0]+R[2]))  # layer-resolved transmittance p-pol
        Ts_z = S_out[5, :]/S_out[5,0]*(1.0-(R[1]+R[3]))  # layer-resolved transmittance s-pol
        A_out[0, z1:] = 1.0-(R[0]+R[2])-Tp_z[z1:]
        A_out[1, z1:] = 1.0-(R[1]+R[3])-Ts_z[z1:]

        return S_out, A_out

    # def calculate_matelem(self, zeta0, f):
    #     """
    #     Calculate the common denominator of all reflexion/transmission coefficients.

    #     Parameters
    #     ----------
    #     zeta0 : 2-tuple
    #         Tuple [zeta_r, zeta_i] of real and imaginary part of the wavevector
    #     f : float
    #         frequency (in Hz)

    #     Returns
    #     -------
    #     matelem : complex
    #         Matrix element to minimize for dispersion relation (absolute value)

    #     Notes
    #     -----
    #     Returns the relevant quantity to find waveguide modes according
    #     to Davis' paper on multilayers (scalar model
    #     http://doi.org/10.1016/j.optcom.2008.09.043)
    #     and then Yeh (4X4 formalism http://doi.org/10.1016/0039-6028(80)90293-9).

    #     """
    #     self.initialize_sys(f)
    #     zeta_sys = zeta0[0]+1.0j*zeta0[1]
    #     self.calculate_GammaStar(f, zeta_sys)
    #     matelem = self.GammaStar[0,0]*self.GammaStar[2,2]-self.GammaStar[0,2]*self.GammaStar[2,0]
    #     return np.abs(matelem)

    # def calculate_eigen_wv(self, zeta0, f, bounds=None):
    #     """
    #     Calculate the eigenmode in-plane wavevector that shows guiding along the plane.

    #     Parameters
    #     ----------
    #     zeta0 : 2-tuple
    #         Initial guess for the minimization procedure
    #     f : float
    #         Frequency (in Hz)
    #     bounds : list (optional)
    #         list of 2-tuple containing (lower, upper) bound for each parameter

    #     Returns
    #     -------
    #     res : OptimizeResult
    #         Result of the minimization procedure. Eigenvalue is the list res.x

    #     Notes
    #     -----
    #     Based on the idea that guided mode := an output field exists with no input field
    #     This is **strongly** dependant on the minimization procedure and thus
    #     has to be consistently and carefully checked.

    #     """
    #     res = minimize(self.calculate_matelem, zeta0, args=(f),
    #                    method='SLSQP', bounds=bounds)
    #     return res

    # def disp_vs_f(self, fv, zeta0, bounds=None):
    #     """
    #     Performs a frequency dependent search of the eigenwavevector for a guided mode
    #     to get the dispersion relation of a surface mode.

    #     Provided a reasonable initial guess for the first frequency point, we 
    #     use the eigen_wv from the above method and follow its value as a function 
    #     of frequency in a stepping manner.

    #     Parameters
    #     -----------
    #     fv : 1Darray
    #         Array of frequencies
    #     zeta0 : 2-tuple
    #         Initial guess for the minimization
    #     bounds : list
    #         list of 2-tuple containing (lower, upper) bound for each parameter

    #     Returns
    #     -------
    #     zeta_disp_r : 1Darray (complex)
    #         Array of real part of the in-plane wavevector
    #     zeta_disp_i : 1Darray (complex)
    #         Array of imaginary part of the in-plane wavevector
    #     """
    #     zeta_disp_r = np.zeros(len(fv))
    #     zeta_disp_i = np.zeros(len(fv))
    #     zeta_disp_r[-1] = zeta0[0]
    #     zeta_disp_i[-1] = zeta0[1]
    #     print('Solving for dispersion relation: \n')
    #     for ii, fi in enumerate(fv):
    #         print(ii/len(fv))
    #         zetaguess = [zeta_disp_r[ii-1], zeta_disp_i[ii-1]]
    #         res = self.calculate_eigen_wv(zetaguess, fi, bounds=bounds)
    #         zeta_disp_r[ii] = res.x[0]
    #         zeta_disp_i[ii] = res.x[1]
    #     return zeta_disp_r, zeta_disp_i
