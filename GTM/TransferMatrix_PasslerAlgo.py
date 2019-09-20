# -*- coding: utf-8 -*-
"""
This program implements the generalized 4x4 transfer matrix (GTM) method poposed in 
Passler, N. C. and Paarmann, A., JOSA B 34, 2128 (2017) doi.org/10.1364/JOSAB.34.002128
and corrected in <DOI to be inserted>, with inputs from D. Dietze's FSRStools code
https://github.com/ddietze/FSRStools

Layers are represented by the :py:class:`Layer` class that holds all parameters 
describing the optical properties of a single layer. 
The optical system is assembled using the :py:class:`System` class.

Please cite the associated publications if you use this code. 

author: Mathieu Jeannin <math.jeannin@free.fr> (permanent)
affiliation: Laboratoire de Physique de l'Ecole Normale Superieure (2019)

**Change log:**

*20-09-2019*:
    - Added functions in the `System` class to compute in-plane wavevector of guided modes
    and dispersion relation for such guided surface modes
    ** Highly propespective**

    
..
This file is part of the GTM module.

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    Copyright (C) Mathieu Jeannin 2019 <math.jeannin@free.fr>.
    
"""
######## general utilities

import numpy as np
import numpy.linalg as lag
from scipy.optimize import minimize

c_const = 299792458 # m/s
eps0 = 8.854e-12 ## vacuum permittivity
qsd_thr = 1e-10 ### threshold for wavevector comparison


def vacuum_eps(f):
    try:
        return np.ones(len(f))
    except:
        return 1.0+1.0j*0
    
    
def exact_inv(M):
    """Compute the 'exact' inverse of a 4x4 matrix using the analytical result. 
    
    This should give a higher precision and speed at a reduced noise.

    :param matrix M: 4x4 Matrix.
    :returns: Inverse of this matrix or Moore-Penrose approximation if matrix cannot be inverted.

    .. seealso:: http://www.cg.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/teche23.html
    
    From D.Dietze code https://github.com/ddietze/FSRStools
    """
    assert M.shape == (4, 4)

    # the following equations use algebraic indexing; transpose input matrix to get indexing right
    A = M.T
    detA = A[0, 0] * A[1, 1] * A[2, 2] * A[3, 3] + A[0, 0] * A[1, 2] * A[2, 3] * A[3, 1] + A[0, 0] * A[1, 3] * A[2, 1] * A[3, 2]
    detA = detA + A[0, 1] * A[1, 0] * A[2, 3] * A[3, 2] + A[0, 1] * A[1, 2] * A[2, 0] * A[3, 3] + A[0, 1] * A[1, 3] * A[2, 2] * A[3, 0]
    detA = detA + A[0, 2] * A[1, 0] * A[2, 1] * A[3, 3] + A[0, 2] * A[1, 1] * A[2, 3] * A[3, 0] + A[0, 2] * A[1, 3] * A[2, 0] * A[3, 1]
    detA = detA + A[0, 3] * A[1, 0] * A[2, 2] * A[3, 1] + A[0, 3] * A[1, 1] * A[2, 0] * A[3, 2] + A[0, 3] * A[1, 2] * A[2, 1] * A[3, 0]

    detA = detA - A[0, 0] * A[1, 1] * A[2, 3] * A[3, 2] - A[0, 0] * A[1, 2] * A[2, 1] * A[3, 3] - A[0, 0] * A[1, 3] * A[2, 2] * A[3, 1]
    detA = detA - A[0, 1] * A[1, 0] * A[2, 2] * A[3, 3] - A[0, 1] * A[1, 2] * A[2, 3] * A[3, 0] - A[0, 1] * A[1, 3] * A[2, 0] * A[3, 2]
    detA = detA - A[0, 2] * A[1, 0] * A[2, 3] * A[3, 1] - A[0, 2] * A[1, 1] * A[2, 0] * A[3, 3] - A[0, 2] * A[1, 3] * A[2, 1] * A[3, 0]
    detA = detA - A[0, 3] * A[1, 0] * A[2, 1] * A[3, 2] - A[0, 3] * A[1, 1] * A[2, 2] * A[3, 0] - A[0, 3] * A[1, 2] * A[2, 0] * A[3, 1]

    if detA == 0:
        return np.linalg.pinv(M)

    B = np.zeros(A.shape, dtype=np.complex128)
    B[0, 0] = A[1, 1] * A[2, 2] * A[3, 3] + A[1, 2] * A[2, 3] * A[3, 1] + A[1, 3] * A[2, 1] * A[3, 2] - A[1, 1] * A[2, 3] * A[3, 2] - A[1, 2] * A[2, 1] * A[3, 3] - A[1, 3] * A[2, 2] * A[3, 1]
    B[0, 1] = A[0, 1] * A[2, 3] * A[3, 2] + A[0, 2] * A[2, 1] * A[3, 3] + A[0, 3] * A[2, 2] * A[3, 1] - A[0, 1] * A[2, 2] * A[3, 3] - A[0, 2] * A[2, 3] * A[3, 1] - A[0, 3] * A[2, 1] * A[3, 2]
    B[0, 2] = A[0, 1] * A[1, 2] * A[3, 3] + A[0, 2] * A[1, 3] * A[3, 1] + A[0, 3] * A[1, 1] * A[3, 2] - A[0, 1] * A[1, 3] * A[3, 2] - A[0, 2] * A[1, 1] * A[3, 3] - A[0, 3] * A[1, 2] * A[3, 1]
    B[0, 3] = A[0, 1] * A[1, 3] * A[2, 2] + A[0, 2] * A[1, 1] * A[2, 3] + A[0, 3] * A[1, 2] * A[2, 1] - A[0, 1] * A[1, 2] * A[2, 3] - A[0, 2] * A[1, 3] * A[2, 1] - A[0, 3] * A[1, 1] * A[2, 2]

    B[1, 0] = A[1, 0] * A[2, 3] * A[3, 2] + A[1, 2] * A[2, 0] * A[3, 3] + A[1, 3] * A[2, 2] * A[3, 0] - A[1, 0] * A[2, 2] * A[3, 3] - A[1, 2] * A[2, 3] * A[3, 0] - A[1, 3] * A[2, 0] * A[3, 2]
    B[1, 1] = A[0, 0] * A[2, 2] * A[3, 3] + A[0, 2] * A[2, 3] * A[3, 0] + A[0, 3] * A[2, 0] * A[3, 2] - A[0, 0] * A[2, 3] * A[3, 2] - A[0, 2] * A[2, 0] * A[3, 3] - A[0, 3] * A[2, 2] * A[3, 0]
    B[1, 2] = A[0, 0] * A[1, 3] * A[3, 2] + A[0, 2] * A[1, 0] * A[3, 3] + A[0, 3] * A[1, 2] * A[3, 0] - A[0, 0] * A[1, 2] * A[3, 3] - A[0, 2] * A[1, 3] * A[3, 0] - A[0, 3] * A[1, 0] * A[3, 2]
    B[1, 3] = A[0, 0] * A[1, 2] * A[2, 3] + A[0, 2] * A[1, 3] * A[2, 0] + A[0, 3] * A[1, 0] * A[2, 2] - A[0, 0] * A[1, 3] * A[2, 2] - A[0, 2] * A[1, 0] * A[2, 3] - A[0, 3] * A[1, 2] * A[2, 0]

    B[2, 0] = A[1, 0] * A[2, 1] * A[3, 3] + A[1, 1] * A[2, 3] * A[3, 0] + A[1, 3] * A[2, 0] * A[3, 1] - A[1, 0] * A[2, 3] * A[3, 1] - A[1, 1] * A[2, 0] * A[3, 3] - A[1, 3] * A[2, 1] * A[3, 0]
    B[2, 1] = A[0, 0] * A[2, 3] * A[3, 1] + A[0, 1] * A[2, 0] * A[3, 3] + A[0, 3] * A[2, 1] * A[3, 0] - A[0, 0] * A[2, 1] * A[3, 3] - A[0, 1] * A[2, 3] * A[3, 0] - A[0, 3] * A[2, 0] * A[3, 1]
    B[2, 2] = A[0, 0] * A[1, 1] * A[3, 3] + A[0, 1] * A[1, 3] * A[3, 0] + A[0, 3] * A[1, 0] * A[3, 1] - A[0, 0] * A[1, 3] * A[3, 1] - A[0, 1] * A[1, 0] * A[3, 3] - A[0, 3] * A[1, 1] * A[3, 0]
    B[2, 3] = A[0, 0] * A[1, 3] * A[2, 1] + A[0, 1] * A[1, 0] * A[2, 3] + A[0, 3] * A[1, 1] * A[2, 0] - A[0, 0] * A[1, 1] * A[2, 3] - A[0, 1] * A[1, 3] * A[2, 0] - A[0, 3] * A[1, 0] * A[2, 1]

    B[3, 0] = A[1, 0] * A[2, 2] * A[3, 1] + A[1, 1] * A[2, 0] * A[3, 2] + A[1, 2] * A[2, 1] * A[3, 0] - A[1, 0] * A[2, 1] * A[3, 2] - A[1, 1] * A[2, 2] * A[3, 0] - A[1, 2] * A[2, 0] * A[3, 1]
    B[3, 1] = A[0, 0] * A[2, 1] * A[3, 2] + A[0, 1] * A[2, 2] * A[3, 0] + A[0, 2] * A[2, 0] * A[3, 1] - A[0, 0] * A[2, 2] * A[3, 1] - A[0, 1] * A[2, 0] * A[3, 2] - A[0, 2] * A[2, 1] * A[3, 0]
    B[3, 2] = A[0, 0] * A[1, 2] * A[3, 1] + A[0, 1] * A[1, 0] * A[3, 2] + A[0, 2] * A[1, 1] * A[3, 0] - A[0, 0] * A[1, 1] * A[3, 2] - A[0, 1] * A[1, 2] * A[3, 0] - A[0, 2] * A[1, 0] * A[3, 1]
    B[3, 3] = A[0, 0] * A[1, 1] * A[2, 2] + A[0, 1] * A[1, 2] * A[2, 0] + A[0, 2] * A[1, 0] * A[2, 1] - A[0, 0] * A[1, 2] * A[2, 1] - A[0, 1] * A[1, 0] * A[2, 2] - A[0, 2] * A[1, 1] * A[2, 0]

    return B.T / detA


#%%
###########################
#### The Layer Class   ####
###########################
class Layer:
    """
    Layer class. An instance is a single layer:
        
    :param float thickness: thickness of the layer in m
    :param function epsilon1: function epsilon(frequency) for the first axis. If none, defaults to vacuum.
    :param function epsilon2: function epsilon(frequency) for the second axis. If none, defaults to epsilon1.
    :param function epsilon3: function epsilon(frequency) for the third axis. If none, defaults to epsilon1.
    :param float theta: Euler angle theta (colatitude)
    :param float phi: Euler angle phi
    :param float psi: Euler angle psi 
    """

    def __init__(self, thickness=1.0e-6, epsilon1=None, epsilon2=None, epsilon3=None,
                                         theta=0, phi=0, psi=0): 
        
        self.epsilon = np.identity(3, dtype=np.complex128)
        self.mu = 1.0 ### mu=1 for now
        ## epsilon is a 3x3 matrix of permittivity at a given frequency
        
        ### initialization of all important quantities
        self.M = np.zeros((6, 6), dtype=np.complex128) ## constitutive relations
        self.a = np.zeros((6, 6), dtype=np.complex128) ##
        self.S = np.zeros((4, 4), dtype=np.complex128) ##
        self.Delta = np.zeros((4, 4), dtype=np.complex128) ##
        self.qs = np.zeros(4, dtype=np.complex128) ## out of plane wavevector
        self.Py = np.zeros((3,4), dtype=np.complex128) ## Poyting vector
        self.gamma = np.zeros((4, 3), dtype=np.complex128) ##
        self.Ai = np.zeros((4, 4), dtype=np.complex128) ##
        self.Ki = np.zeros((4, 4), dtype=np.complex128) ##
        self.Ti = np.zeros((4, 4), dtype=np.complex128) ## Layer transfer matrix

      
        self.euler = np.identity(3, dtype=np.complex128) ## rotation matrix
        
        self.set_thickness(thickness) ## set the thickness, 1um by default
        self.set_epsilon(epsilon1, epsilon2, epsilon3) # set epsilon, vacuum by default
        self.set_euler(theta, phi, psi) ## set orientation of crystal axis w/ respect to the lab frame
         

    def set_thickness(self, thickness):
        self.thick = thickness
        
    def set_epsilon(self, epsilon1=None, epsilon2=None, epsilon3=None):
        """
        Sets the dielectric functions for the three main axis.
        
        Each epsilon_i function returns the dielectric constant along axis i as 
        a function of the frequency f in Hz.
        
        epsilon1 defaults to 1.0
        epsilon2 and epsilon3 default to epsilon1: if None, a homogeneous material is assumed
        """
        if epsilon1==None:
            self.epsilon1_f = vacuum_eps
        else:
            self.epsilon1_f = epsilon1
        
        if epsilon2 == None:
            self.epsilon2_f = self.epsilon1_f
        else:
            self.epsilon2_f = epsilon2
        if epsilon3 == None:
            self.epsilon3_f = self.epsilon1_f
        else:
            self.epsilon3_f = epsilon3
    

    def calculate_epsilon(self, f):
        """ 
        Sets the value of epsilon in the (rotated) lab frame. 
        
        The values are set according to the epsilon_fi (i=1..3) functions 
        defined using the 'set_epsilon' method, at the given frequency f. 
        The rotation w/ respect to the lab frame is computed using the Euler angles.
        
        ** Use only explicitely if you don't use the `update` function **
        """
        epsilon_xstal = np.zeros((3,3), dtype=np.complex128)
        epsilon_xstal[0,0] = self.epsilon1_f(f)
        epsilon_xstal[1,1] = self.epsilon2_f(f)
        epsilon_xstal[2,2] = self.epsilon3_f(f)
        self.epsilon = np.matmul(lag.pinv(self.euler), np.matmul(epsilon_xstal,self.euler))
        self.epsilon = epsilon_xstal.copy()
        return self.epsilon.copy()
    
    
    def set_euler(self,theta,phi,psi):
        """
        Sets the values for the Euler rotations angles. 
        :param float theta: Euler angle theta (colatitude)
        :param float phi: Euler angle phi
        :param float psi: Euler angle psi 
        """
        self.theta = theta
        self.phi = phi
        self.psi = psi
        # euler matrix for rotation of dielectric tensor
        self.euler[0, 0] = np.cos(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.sin(psi)
        self.euler[0, 1] = -np.sin(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.cos(psi)
        self.euler[0, 2] = np.sin(theta) * np.sin(phi)
        self.euler[1, 0] = np.cos(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.sin(psi)
        self.euler[1, 1] = -np.sin(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.cos(psi)
        self.euler[1, 2] = -np.sin(theta) * np.cos(phi)
        self.euler[2, 0] = np.sin(theta) * np.sin(psi)
        self.euler[2, 1] = np.sin(theta) * np.cos(psi)
        self.euler[2, 2] = np.cos(theta)

        
    def calculate_matrices(self, zeta):
        """
        Calculate the principal matrices necessary for the GTM algorithm.
        
        :param complex128 zeta: in-place reduced wavevector kx/k0 in the system. 
        
        Note that zeta is conserved through the whole system and set externaly
        using the angle of incidence and System.superstrate.epsilon[0,0] value
        
        ** Requires prior execution of `calculate_epsilon` **
        
        """
        ## Constitutive matrix (see e.g. eqn (4))
        self.M[0:3, 0:3] = self.epsilon.copy()
        self.M[3:6, 3:6] = self.mu*np.identity(3)
        
        ## from eqn (10)
        b = self.M[2,2]*self.M[5,5] - self.M[2,5]*self.M[5,2]
        
        ## a matrix from eqn (9)
        self.a[2,0] = (self.M[5,0]*self.M[2,5] - self.M[2,0]*self.M[5,5])/b
        self.a[2,1] = ((self.M[5,1]-zeta)*self.M[2,5] - self.M[2,1]*self.M[5,5])/b
        self.a[2,3] = (self.M[5,3]*self.M[2,5] - self.M[2,3]*self.M[5,5])/b
        self.a[2,4] = (self.M[5,4]*self.M[2,5] - (self.M[2,4]+zeta)*self.M[5,5])/b
        self.a[5,0] = (self.M[5,2]*self.M[2,0] - self.M[2,2]*self.M[5,0])/b
        self.a[5,1] = (self.M[5,2]*self.M[2,1] - self.M[2,2]*(self.M[5,1]-zeta))/b
        self.a[5,3] = (self.M[5,2]*self.M[2,3] - self.M[2,2]*self.M[5,3])/b
        self.a[5,4] = (self.M[5,2]*(self.M[2,4]+zeta) - self.M[2,2]*self.M[5,4])/b
        
        ## S Matrix (Don't know where it comes from since Delta is just S re-ordered)
        ## Note that after this only Delta is used
        ### S[3,ii] was wrong in matlab code (M{kl}(4,6,:) should be M{kl}(5,6,:))
        self.S[0,0] = self.M[0,0] + self.M[0,2]*self.a[2,0] + self.M[0,5]*self.a[5,0];
        self.S[0,1] = self.M[0,1] + self.M[0,2]*self.a[2,1] + self.M[0,5]*self.a[5,1];
        self.S[0,2] = self.M[0,3] + self.M[0,2]*self.a[2,3] + self.M[0,5]*self.a[5,3];
        self.S[0,3] = self.M[0,4] + self.M[0,2]*self.a[2,4] + self.M[0,5]*self.a[5,4];
        self.S[1,0] = self.M[1,0] + self.M[1,2]*self.a[2,0] + (self.M[1,5]-zeta)*self.a[5,0];
        self.S[1,1] = self.M[1,1] + self.M[1,2]*self.a[2,1] + (self.M[1,5]-zeta)*self.a[5,1];
        self.S[1,2] = self.M[1,3] + self.M[1,2]*self.a[2,3] + (self.M[1,5]-zeta)*self.a[5,3];
        self.S[1,3] = self.M[1,4] + self.M[1,2]*self.a[2,4] + (self.M[1,5]-zeta)*self.a[5,4];
        self.S[2,0] = self.M[3,0] + self.M[3,2]*self.a[2,0] + self.M[3,5]*self.a[5,0];
        self.S[2,1] = self.M[3,1] + self.M[3,2]*self.a[2,1] + self.M[3,5]*self.a[5,1];
        self.S[2,2] = self.M[3,3] + self.M[3,2]*self.a[2,3] + self.M[3,5]*self.a[5,3];
        self.S[2,3] = self.M[3,4] + self.M[3,2]*self.a[2,4] + self.M[3,5]*self.a[5,4];
        self.S[3,0] = self.M[4,0] + (self.M[4,2]+zeta)*self.a[2,0] + self.M[4,5]*self.a[5,0];
        self.S[3,1] = self.M[4,1] + (self.M[4,2]+zeta)*self.a[2,1] + self.M[4,5]*self.a[5,1];
        self.S[3,2] = self.M[4,3] + (self.M[4,2]+zeta)*self.a[2,3] + self.M[4,5]*self.a[5,3];
        self.S[3,3] = self.M[4,4] + (self.M[4,2]+zeta)*self.a[2,4] + self.M[4,5]*self.a[5,4];
        
        
        ## Delta Matrix from eqn (8)
        self.Delta[0,0] = self.S[3,0]
        self.Delta[0,1] = self.S[3,3]
        self.Delta[0,2] = self.S[3,1]
        self.Delta[0,3] = - self.S[3,2]
        self.Delta[1,0] = self.S[0,0]
        self.Delta[1,1] = self.S[0,3]
        self.Delta[1,2] = self.S[0,1]
        self.Delta[1,3] = - self.S[0,2]
        self.Delta[2,0] = -self.S[2,0]
        self.Delta[2,1] = -self.S[2,3]
        self.Delta[2,2] = -self.S[2,1]
        self.Delta[2,3] = self.S[2,2]
        self.Delta[3,0] = self.S[1,0]
        self.Delta[3,1] = self.S[1,3]
        self.Delta[3,2] = self.S[1,1]
        self.Delta[3,3] = -self.S[1,2]        
        
    def calculate_q(self):
        """
        This function calculates the 4 out-of-plane wavevectors for the current layer. 
        
        From this we also get the Poynting vectors. 
        Wavevectors are sorted according to (trans-p, trans-s, refl-p, refl-s)
        Birefringence is determined according to a threshold value `qsd_thr` set at the beginning of the script. 
        """
        Delta_loc = np.zeros((4,4), dtype=np.complex128)
        transmode = np.zeros((2), dtype=np.int)
        reflmode = np.zeros((2), dtype=np.int)
        
        Delta_loc = self.Delta.copy()
        ## eigenvals // eigenvects as of eqn (11)
        qsunsorted, psiunsorted = lag.eig(Delta_loc)
        
        kt = 0 
        kr = 0;
        ## sort berremann qi's according to (12)
        if any(np.abs(np.imag(qsunsorted))):
            for km in range(0,4):
                if np.imag(qsunsorted[km])>=0 :
                    transmode[kt] = km
                    kt = kt + 1
                else:
                    reflmode[kr] = km
                    kr = kr +1
        else:
            for km in range(0,4):
                if np.real(qsunsorted[km])>0 :
                    transmode[kt] = km
                    kt = kt + 1
                else:
                    reflmode[kr] = km
                    kr = kr +1
        ## Calculate the Poyting vector for each Psi using (16-18)
        for km in range(0,4):
            Ex = psiunsorted[0,km] 
            Ey = psiunsorted[2,km]
            Hx = -psiunsorted[3,km]
            Hy = psiunsorted[1,km]
            ## from eqn (17)
            Ez = self.a[2,0]*Ex + self.a[2,1]*Ey + self.a[2,3]*Hx + self.a[2,4]*Hy
            # from eqn (18)
            Hz = self.a[5,0]*Ex + self.a[5,1]*Ey + self.a[5,3]*Hx + self.a[5,4]*Hy
            ## and from (16)
            self.Py[0,km] = Ey*Hz-Ez*Hy
            self.Py[1,km] = Ez*Hx-Ex*Hz
            self.Py[2,km] = Ex*Hy-Ey*Hx
            
        ## check Cp using either the Poynting vector for birefringent
        ## materials or the electric field vector for non-birefringent
        ## media to sort the modes       
        
        ## first calculate Cp for transmitted waves
        Cp_t1 = np.abs(self.Py[0,transmode[0]])**2/(np.abs(self.Py[0,transmode[0]])**2+np.abs(self.Py[1,transmode[0]])**2)
        Cp_t2 = np.abs(self.Py[0,transmode[1]])**2/(np.abs(self.Py[0,transmode[1]])**2+np.abs(self.Py[1,transmode[1]])**2)
        
        if np.abs(Cp_t1-Cp_t2) > qsd_thr: ## birefringence
            if Cp_t2>Cp_t1:
                transmode = np.flip(transmode,0) ## flip the two values
            ## then calculate for reflected waves if necessary
            Cp_r1 = np.abs(self.Py[0,reflmode[1]])**2/(np.abs(self.Py[0,reflmode[1]])**2+np.abs(self.Py[1,reflmode[1]])**2)
            Cp_r2 = np.abs(self.Py[0,reflmode[0]])**2/(np.abs(self.Py[0,reflmode[0]])**2+np.abs(self.Py[1,reflmode[0]])**2)
            if Cp_r1>Cp_r2:
                reflmode = np.flip(reflmode,0) ## flip the two values
        
        else:     ### No birefringence, use the Electric field s-pol/p-pol
            Cp_te1 = np.abs(psiunsorted[0,transmode[1]])**2/(np.abs(psiunsorted[0,transmode[1]])**2+np.abs(psiunsorted[2,transmode[1]])**2)
            Cp_te2 = np.abs(psiunsorted[0,transmode[0]])**2/(np.abs(psiunsorted[0,transmode[0]])**2+np.abs(psiunsorted[2,transmode[0]])**2)
            if Cp_te1>Cp_te2:
                transmode = np.flip(transmode,0) ## flip the two values
            Cp_re1 = np.abs(psiunsorted[0,reflmode[1]])**2/(np.abs(psiunsorted[0,reflmode[1]])**2+np.abs(psiunsorted[2,reflmode[1]])**2)  
            Cp_re2 = np.abs(psiunsorted[0,reflmode[0]])**2/(np.abs(psiunsorted[0,reflmode[0]])**2+np.abs(psiunsorted[2,reflmode[0]])**2)  
            if Cp_re1>Cp_re2:
                reflmode = np.flip(reflmode,0) ## flip the two values
        
        ## finaly store the sorted version        
        ####### q is (trans-p, trans-s, refl-p, refl-s)
        self.qs[0] = qsunsorted[transmode[0]]
        self.qs[1] = qsunsorted[transmode[1]]
        self.qs[2] = qsunsorted[reflmode[0]]
        self.qs[3] = qsunsorted[reflmode[1]]
        Py_temp = self.Py.copy()      
        self.Py[:,0] = Py_temp[:,transmode[0]]
        self.Py[:,1] = Py_temp[:,transmode[1]]
        self.Py[:,2] = Py_temp[:,reflmode[0]]
        self.Py[:,3] = Py_temp[:,reflmode[1]]
        
    def calculate_gamma(self, zeta):
        """
        Calculate the gamma matrix
        
        :param complex zeta: in-plane reduced wavevector kx/k0
        """
        ### this whole function is eqn (20)
        self.gamma[0,0] = 1.0 + 0.0j
        self.gamma[1,1] = 1.0 + 0.0j
        self.gamma[3,1] = 1.0 + 0.0j
        self.gamma[2,0] = -1.0 + 0.0j
        
        if np.abs(self.qs[0]-self.qs[1])<qsd_thr:
            gamma12 = 0.0 + 0.0j
            
            gamma13 = -(self.mu*self.epsilon[2,0]+zeta*self.qs[0])
            gamma13 = gamma13/(self.mu*self.epsilon[2,2]-zeta**2)
            
            gamma21 = 0.0 + 0.0j
            
            gamma23 = -self.mu*self.epsilon[2,1]
            gamma23 = gamma23/(self.mu*self.epsilon[2,2]-zeta**2)
        
        else:
            gamma12_num = self.mu*self.epsilon[1,2]*(self.mu*self.epsilon[2,0]+zeta*self.qs[0])
            gamma12_num = gamma12_num - self.mu*self.epsilon[1,0]*(self.mu*self.epsilon[2,2]-zeta**2)
            gamma12_denom = (self.mu*self.epsilon[2,2]-zeta**2)*(self.mu*self.epsilon[1,1]-zeta**2-self.qs[0]**2)
            gamma12_denom = gamma12_denom - self.mu**2*self.epsilon[1,2]*self.epsilon[2,1]
            gamma12 = gamma12_num/gamma12_denom
            if np.isnan(gamma12):
                gamma12 = 0.0 + 0.0j
            
            gamma13 = -(self.mu*self.epsilon[2,0]+zeta*self.qs[0])
            gamma13 = gamma13-self.mu*self.epsilon[2,1]*gamma12 #### gamma12 factor missing in (20) but present in ref [13]
            gamma13 = gamma13/(self.mu*self.epsilon[2,2]-zeta**2)
            
            if np.isnan(gamma13):
                gamma13 = -(self.mu*self.epsilon[2,0]+zeta*self.qs[0])
                gamma13 = gamma13/(self.mu*self.epsilon[2,2]-zeta**2)

            gamma21_num = self.mu*self.epsilon[2,1]*(self.mu*self.epsilon[0,2]+zeta*self.qs[1])
            gamma21_num = gamma21_num-self.mu*self.epsilon[0,1]*(self.mu*self.epsilon[2,2]-zeta**2)
            gamma21_denom = (self.mu*self.epsilon[2,2]-zeta**2)*(self.mu*self.epsilon[0,0]-self.qs[1]**2)
            gamma21_denom = gamma21_denom-(self.mu*self.epsilon[0,2]+zeta*self.qs[1])*(self.mu*self.epsilon[2,0]+zeta*self.qs[1])
            gamma21 = gamma21_num/gamma21_denom
            if np.isnan(gamma21):
                gamma21 = 0.0+0.0j
                
            gamma23 = -(self.mu*self.epsilon[2,0] +zeta*self.qs[1])*gamma21-self.mu*self.epsilon[2,1]
            gamma23 = gamma23/(self.mu*self.epsilon[2,2]-zeta**2)
            if np.isnan(gamma23):
                gamma23 = -self.mu*self.epsilon[2,1]/(self.mu*self.epsilon[2,2]-zeta**2)

        if np.abs(self.qs[2]-self.qs[3])<qsd_thr:
            gamma32 = 0.0 + 0.0j
            gamma33 = (self.mu*self.epsilon[2,0]+zeta*self.qs[2])/(self.mu*self.epsilon[2,2]-zeta**2)
            gamma41 = 0.0 + 0.0j
            gamma43 = -self.mu*self.epsilon[2,1]/(self.mu*self.epsilon[2,2]-zeta**2)
        
        else:
            gamma32_num = self.mu*self.epsilon[1,0]*(self.mu*self.epsilon[2,2]+zeta**2)
            gamma32_num = gamma32_num-self.mu*self.epsilon[1,2]*(self.mu*self.epsilon[2,0]+zeta*self.qs[2])
            gamma32_denom = (self.mu*self.epsilon[2,2]-zeta**2)*(self.mu*self.epsilon[1,1]-zeta**2-self.qs[2]**2)
            gamma32_denom = gamma32_denom-self.mu**2*self.epsilon[1,2]*self.epsilon[2,1]
            gamma32 = gamma32_num/gamma32_denom
            if np.isnan(gamma32):
                gamma32 = 0.0 + 0.0j
            
            gamma33 = self.mu*self.epsilon[2,0] + zeta*self.qs[2]
            gamma33 = gamma33 + self.mu*self.epsilon[2,1]*gamma32 #### gamma32 factor missing in (20) but present in ref [13]
            gamma33 = gamma33/(self.mu*self.epsilon[2,2]-zeta**2)
            if np.isnan(gamma33):
                gamma33 = (self.mu*self.epsilon[2,0] + zeta*self.qs[2])/(self.mu*self.epsilon[2,2]-zeta**2)
            
            gamma41_num = self.mu*self.epsilon[2,1]*(self.mu*self.epsilon[0,2]+zeta*self.qs[3])
            gamma41_num = gamma41_num - self.mu*self.epsilon[0,1]*(self.mu*self.epsilon[2,2]-zeta**2)
            gamma41_denom = (self.mu*self.epsilon[2,2]-zeta**2)*(self.mu*self.epsilon[0,0]-self.qs[3]**2)
            gamma41_denom = gamma41_denom - (self.mu*self.epsilon[0,2]+zeta*self.qs[3])*(self.mu*self.epsilon[2,0]+zeta*self.qs[3])
            gamma41 = gamma41_num/gamma41_denom
            if np.isnan(gamma41):
                gamma41 = 0.0 + 0.0j
                
            gamma43 = -(self.mu*self.epsilon[2,0]+zeta*self.qs[3])*gamma41
            gamma43 = gamma43-self.mu*self.epsilon[2,1]
            gamma43 = gamma43/(self.mu*self.epsilon[2,2]-zeta**2)
            if np.isnan(gamma43):
                gamma43 = -self.mu*self.epsilon[2,1]/(self.mu*self.epsilon[2,2]-zeta**2)
            
        self.gamma[0,1] = gamma12
        self.gamma[0,2] = gamma13
        self.gamma[1,0] = gamma21
        self.gamma[1,2] = gamma23
        self.gamma[2,1] = gamma32
        self.gamma[2,2] = gamma33
        self.gamma[3,0] = gamma41
        self.gamma[3,2] = gamma43

    def calculate_transfer_matrix(self, f, zeta):
        """
        Compute the transfer matrix of the whole layer T=APA^{-1}
        
        :param float f: frequency 
        :param complex zeta: reduced in-plane wavevector kx/k0
        """
        ## eqn(22)
        self.Ai[0,:] = self.gamma[:,0].copy()
        self.Ai[1,:] = self.gamma[:,1].copy()
        
        self.Ai[2,:] = (self.qs*self.gamma[:,0]-zeta*self.gamma[:,2])/self.mu
        self.Ai[3,:] = self.qs*self.gamma[:,1]/self.mu
        
        for ii in range(4):
            ## looks a lot like eqn (25). Why is K not Pi ?
            self.Ki[ii,ii] = np.exp(-1.0j*(2.0*np.pi*f*self.qs[ii]*self.thick)/c_const)
        
        Aim1 = exact_inv(self.Ai.copy())
        ## eqn (26)
        self.Ti = np.matmul(self.Ai,np.matmul(self.Ki,Aim1))
        

    def update(self, f, zeta):
        """Shortcut to recalculate all layer properties.

        :param float zeta: in-plane propagation vector (reduced)
        :param float f: Frequency value.
        :returns: matrices Ai, Ki, Ai^{-1} and Ti
        """
        
        self.calculate_epsilon(f)
        self.calculate_matrices(zeta)
        self.calculate_q()
        self.calculate_gamma(zeta)
        self.calculate_transfer_matrix(f, zeta)
        Ai_inv = exact_inv(self.Ai.copy())
        
        return[self.Ai.copy(), self.Ki.copy(), Ai_inv.copy(), self.Ti.copy()]
        


#%%
###########################
#### The System Class  ####
###########################


class System:
    """
    System class. An instance is an optical system with substrate, superstrate and layers.
    
    :param float theta: angle of incidence, in radians
    :param layer substrate: the substrate layer. defaults to vacuum (empty layer instance)
    :param layer superstrate: the superstrate layer, defaults to vacuum (empty layer instance)
    :param list layers: list of the layers
    
    Layers can be added and removed (not inserted). 
    
    The whole system's transfer matrix is computed using calculate_GammaStar, 
    which calls layer.update() for each layer.
    General reflection and transmission coeffs. functions are given, they require prior 
    execution of calculate_GammaStar.
    The electric fields can be visualized in the case of incident plane wave
    using calculate_Efield
    
    **Note that the transmission coefficients are still ill-defined in this version.
    This will be corrected**
    """
    def __init__(self, substrate=None, superstrate=None, layers=[]):#,
                 #theta=0.0, phi=0.0, psi=0.0):
        
        self.layers=[]
        if len(layers)>0:
            self.layers=layers
        
        ## system transfer matrix
        self.Gamma = np.zeros((4,4), dtype=np.complex128)
        self.GammaStar = np.zeros((4,4), dtype=np.complex128)
               
        if substrate is not None:
            self.substrate = substrate
        else:
            self.substrate=Layer() ## should default to 1µm of vacuum
        if superstrate is not None:
            self.superstrate = superstrate
        else:
            self.superstrate=Layer() ## should default to 1µm of vacuum

    def set_substrate(self,sub):
        """Set the substrate
        """
        self.substrate=sub
        
    def set_superstrate(self,sup):
        """Set the superstrate
        """
        self.superstrate=sup
    
    def get_all_layers(self):
        """Returns the list of all layers in the system
        """
        return self.layers
    
    def get_layer(self,pos):
        """Get the layer at a given position
        """
        return self.layers[pos]
    def get_superstrate(self):
        """Returns the System's superstrate
        """
        return self.superstrate
    def get_substrate(self):
        """Returns the System's substrate
        """
        return self.substrate
    
    def add_layer(self,layer):
        """Add a layer instance.
        Note that the layers are added **from superstrate to substrate** order.
        Light is incident from the superstrate.
        
        .. note:: This function adds a reference to L to the list. So if you are adding the same layer several times, be aware that if you change something for one of them, it changes all of them.
        """
        self.layers.append(layer)
    def del_layer(self,pos):
        """Remove a layer at given position. Does nothing for invalid position.
        :param integer pos: index of layer to be removed.
        """
        if pos >= 0 and pos < len(self.layers):
            self.layers.pop(pos)


    def initialize_sys(self, f):
        """Sets the values of epsilon at given frequency, allowing to define zeta out of the class
        :param float f: frequency (Hz)
        """
        self.superstrate.calculate_epsilon(f)
        self.substrate.calculate_epsilon(f)
        for li in self.layers:
            li.calculate_epsilon(f)
            
            
    def calculate_GammaStar(self,f, zeta_sys):
        """
        Calculate the whole system's transfer matrix GammaStar
        
        :param float f: frequency (Hz)
        :param complex zeta_sys: in-plane wavevector kx/k0
        :returns: System transfer matrix np.array((4,4), dtype=np.complex128)
        """
        Ai_super, Ki_super, Ai_inv_super, T_super = self.superstrate.update(f, zeta_sys)
        Ai_sub, Ki_sub, Ai_inv_sub, T_sub = self.substrate.update(f, zeta_sys)
       
        Delta1234 = np.array([[1,0,0,0],
                              [0,0,1,0],
                              [0,1,0,0],
                              [0,0,0,1]])
        
        
        Gamma = np.zeros(4, dtype=np.complex128)
        GammaStar = np.zeros(4, dtype=np.complex128)
        Tloc = np.identity(4, dtype=np.complex128)
        
        for ii in range(len(self.layers))[::-1]:
            Ai, Ki, Ai_inv, T_ii = self.layers[ii].update(f, zeta_sys)
            Tloc = np.matmul(T_ii,Tloc)
        
        Gamma = np.matmul(Ai_inv_super,np.matmul(Tloc,Ai_sub))
        GammaStar = np.matmul(exact_inv(Delta1234),np.matmul(Gamma,Delta1234))
        
        self.Gamma = Gamma.copy()
        self.GammaStar = GammaStar.copy()
        return self.GammaStar.copy()
        
    
    def calculate_r_t(self):
        """ Calculate various field and intensity reflection and transmission coefficients, as well as the 4-valued vector of transmitted field.
        
        **IMPORTANT** 
        ..version 13-09-2019
        Note tha the field reflectivity and transmission coefficients 
        r and t are well defined. The intensity reflection coefficient is also correct. 
        However, the intensity transmission coefficients T are ill-defined so far. 
        This will be corrected upon future publication of the correct intensity coefficients.
        
        Note also the different ordering of the coefficients, for consistency w/ Passler's matlab code
        :returns: Complex *field* reflection coefficients r_out=([rpp,rps,rss,rsp])
        :returns: Real *intensity* reflection coefficients R_out=([Rpp,Rss,Rsp,Tps])
        :returns: Complex transmitted field t_field=([tpp, tps, tsp, tss])
        :returns: Complex *field* transmition coefficients t_out = ([toutpp, toutps, toutss, toutsp])
        :returns: Real *intensity* transmition coefficients T_out=([Tpp,Tss,Tsp,Tps])
        
        """
        # common denominator for all coefficients
        Denom = self.GammaStar[0,0]*self.GammaStar[2,2]-self.GammaStar[0,2]*self.GammaStar[2,0]
        
        # field reflection coefficients
        rpp = self.GammaStar[1,0]*self.GammaStar[2,2]-self.GammaStar[1,2]*self.GammaStar[2,0]
        rpp = np.nan_to_num(rpp/Denom)
        
        rss = self.GammaStar[0,0]*self.GammaStar[3,2]-self.GammaStar[3,0]*self.GammaStar[0,2]
        rss = np.nan_to_num(rss/Denom)
        
        rps = self.GammaStar[3, 0]*self.GammaStar[2,2]-self.GammaStar[3,2]*self.GammaStar[2,0]
        rps = np.nan_to_num(rps/Denom)
        
        rsp = self.GammaStar[0,0]*self.GammaStar[1,2]-self.GammaStar[1,0]*self.GammaStar[0,2]
        rsp = np.nan_to_num(rsp/Denom)
        
        # Intensity reflection coefficients are just square moduli
        Rpp = np.abs(rpp)**2
        Rss = np.abs(rss)**2
        Rps = np.abs(rps)**2
        Rsp = np.abs(rsp)**2
        r_out = np.array([rpp,rps,rss,rsp]) ## order matching Passler Matlab code
        R_out = np.array([Rpp,Rss,Rsp,Rps]) ## order matching Passler Matlab code

        # field transmission coefficients
        t_field = np.zeros(4, dtype=np.complex128)
        tpp = np.nan_to_num(self.GammaStar[2,2]/Denom)
        tss = np.nan_to_num(self.GammaStar[0,0]/Denom)
        tps = np.nan_to_num(-self.GammaStar[2,0]/Denom)
        tsp = np.nan_to_num(-self.GammaStar[0,2]/Denom)
        t_field = np.array([tpp, tps, tsp, tss])
        
        # toutlm is the m-polarized transmitted field from l-polarized excitation
        # as defined in the first two equations of (E1)
        toutpp = np.zeros(3, dtype=np.complex128)
        toutps = np.zeros(3, dtype=np.complex128)
        toutss = np.zeros(3, dtype=np.complex128)
        toutsp = np.zeros(3, dtype=np.complex128)
        
        toutpp = self.substrate.gamma[0,:]*tpp # gamma[0]->p-pol in substrate
        toutps = self.substrate.gamma[1,:]*tps # gamma[1]->s-pol in substrate
        toutss = self.substrate.gamma[1,:]*tss # gamma[1]->s-pol in substrate
        toutsp = self.substrate.gamma[0,:]*tsp # gamma[0]->s-pol in substrate
        
        ### Power transmission coefficient has a wavevector mismatch to account for
        ## remember that qs is set as (trans-p, trans-s, refl-p, refl-s)
        ## this is currently not working for TM case (to be fixed)
        Tpp = np.abs(tpp)**2*np.real(self.substrate.qs[0]/self.superstrate.qs[0])
        Tss = np.abs(tss)**2*np.real(self.substrate.qs[1]/self.superstrate.qs[1])
        Tps = np.abs(tps)**2*np.real(self.substrate.qs[0]/self.superstrate.qs[1])
        Tsp = np.abs(tsp)**2*np.real(self.substrate.qs[1]/self.superstrate.qs[0])
        
        t_out = np.array([toutpp, toutps, toutss, toutsp])
        T_out = np.array([Tpp, Tss, Tps, Tsp])
        
        return r_out, R_out, t_field, t_out, T_out
       
        
    def calculate_Efield(self, f, zeta_sys, dz, x=0.0):
        """
        Calculate the electric field profiles for both s-pol and p-pol excitation.
        
        ..Version 13-09-2019:
            the 2D field profile is not implemented yet. x should be left to default
        
        :param float f: frequency (Hz)
        :param complex zeta_sys: in-plane normalized wavevector kx/k0
        :param float dz: space resolution along propagation (z) axis
        :param array x: x-coordinates for 2D plot of the electric field.
        :returns: 1D array of z-coordinates according to dz
        :returns: (len(z),3)-Array E_out of total electric field in the structure
        :returns: list zn of the positions of the different interfaces
        """

        self.calculate_GammaStar(f, zeta_sys)
        r_out, R_out, t_field, t_out, T_out = self.calculate_r_t()

        ## Nb of layers
        laynum = len(self.layers)
        zn = np.zeros(laynum+2) ## superstrate+layers+substrate
        
        ## 4-components field tensor at the front and back interfaces of the layer
        ## correspond to E0 and E1
        ## defined by (37*)
        # E0 (E^(p/o)_t, E^(s/e)_t, E^(p/o)_r, E^(s/e)_r) twice for p-pol in and s-pol in
        F_ft = np.zeros((laynum+2,8), dtype=np.complex128) 
        # E1 (E^(p/o)_t, E^(s/e)_t, E^(p/o)_r, E^(s/e)_r) twice for p-pol in and s-pol in
        F_bk = np.zeros((laynum+2,8), dtype=np.complex128) 
        
        zn[-1] = 0.0 ## initially with the substrate
        
        ## (37*) with p-pol excitation
        F_ft[-1,0] = t_field[0] # tpp
        F_ft[-1,1] = t_field[1] # t_ps
        ## (37*) with s-pol excitation
        F_ft[-1,4] = t_field[2] # t_sp
        F_ft[-1,5] = t_field[3] # t_ss
        
        # F_bk[-1] for plot purpose (see Fig. 1.(a))
        F_bk[-1,:4] = np.matmul(exact_inv(self.substrate.Ki), F_ft[-1,:4])
        F_bk[-1,4:] = np.matmul(exact_inv(self.substrate.Ki), F_ft[-1,4:])
        
        if laynum>0:
            
            zn[-2] = zn[-1]-self.substrate.thick
            Aim1 = self.layers[-1].Ai
            Ai = self.substrate.Ai
            Li = np.matmul(exact_inv(Aim1),Ai)
            F_bk[-2,:4] = np.matmul(Li, F_ft[-1,:4])
            F_bk[-2,4:] = np.matmul(Li, F_ft[-1,4:])
            F_ft[-2,:4] = np.matmul(self.layers[-1].Ki, F_bk[-2,:4])
            F_ft[-2,4:] = np.matmul(self.layers[-1].Ki, F_bk[-2,4:])
            
            for kl in range(1,laynum)[::-1]:
                ### subtract the thickness (building thickness array backwards)    
                zn[kl] = zn[kl+1]-self.layers[kl].thick
                Aim1 = self.layers[kl-1].Ai
                Ai = self.layers[kl].Ai
                Li = np.matmul(exact_inv(Aim1),Ai)
                # F_ft == E0  //  F_bk == E1
                F_bk[kl,:4] = np.matmul(Li,F_ft[kl+1,:4])
                F_bk[kl,4:] = np.matmul(Li,F_ft[kl+1,4:])
                F_ft[kl,:4] = np.matmul(self.layers[kl].Ki,F_bk[kl,:4])
                F_ft[kl,4:] = np.matmul(self.layers[kl].Ki,F_bk[kl,4:])
           
            zn[0] = zn[1]-self.layers[0].thick
            Aim1 = self.superstrate.Ai
            Ai = self.layers[0].Ai
            Li = np.matmul(exact_inv(Aim1),Ai)
            # F_ft == E0  //  F_bk == E1
            F_bk[0,:4] = np.matmul(Li,F_ft[1,:4])
            F_bk[0,4:] = np.matmul(Li,F_ft[1,4:])
            F_ft[0,:4] = np.matmul(self.superstrate.Ki,F_bk[0,:4])
            F_ft[0,4:] = np.matmul(self.superstrate.Ki,F_bk[0,4:])
            
        else:
            zn[0] = -self.substrate.thick
            Aim1 = self.substrate.Ai
            Ai = self.superstrate.Ai
            Li = np.matmul(exact_inv(Aim1),Ai)
            # F_ft == E0  //  F_bk == E1
            F_bk[0,:4] = np.matmul(Li, F_ft[1,:4])
            F_bk[0,4:] = np.matmul(Li, F_ft[1,4:])
            F_ft[0,:4] = np.matmul(self.superstrate.Ki, F_bk[0,:4])
            F_ft[0,4:] = np.matmul(self.superstrate.Ki, F_bk[0,4:])
        
        ### shift everything so that incident boundary is at z=0
        zn = zn-zn[0]
        
        z = np.arange(-self.superstrate.thick, zn[-1], dz)
        
        # 2x4 component field tensor E_prop propagated from front surface
        Eprop = np.empty((8), dtype=np.complex128)
        # 4-component field tensor F_tens for each direction and polarization
        F_tens = np.zeros((24,len(z)), dtype=np.complex128)
        # final component electric field E_out = (E_x, Ey, Ez)
        # for p-pol and s-pol excitation
        E_out = np.zeros((6,len(z)), dtype=np.complex128)
        ### Elementary propagation
        dKiz = np.zeros((4,4), dtype=np.complex128)

        ## starting from the superstrate:
        current_layer = 0
        L = self.superstrate
        for ii, zc in enumerate(z): ## enumerates returns a tuple (index, value)
            
            if zc>zn[current_layer]:
                # change the layer
                # important to count here until laynum+1 to get the correct zn
                # in the substrate for dKiz
                
                current_layer += 1
                
                if current_layer == laynum+1: ## reached substrate
                    L = self.substrate
                else:
                    L = self.layers[current_layer-1]
                                
            for kk in range(4):
                # use the conjugate of the K matrix => exp(+1.0j...)
                dKiz[kk,kk] = np.exp(1.0j*(2.0*np.pi*f*L.qs[kk]*(zc-zn[current_layer]))/c_const)
            
            #### Eprop propagated from front surface to back of next layer
            # n.b: unclear why using F_bk and not F_ft works... but it works !
            Eprop[:4] = np.matmul(dKiz,F_bk[current_layer,:4])
            Eprop[4:] = np.matmul(dKiz,F_bk[current_layer,4:])

            ## p-pol in 
            # forward, o/p
            F_tens[:3,ii] = Eprop[0]*L.gamma[0,:]
            # forward, e/s
            F_tens[3:6,ii] = Eprop[1]*L.gamma[1,:]
            # backward, o/p
            F_tens[6:9,ii] = Eprop[2]*L.gamma[2,:]
            # backward, e/s
            F_tens[9:12,ii] = Eprop[3]*L.gamma[3,:]
            ## s-pol in 
            # forward, o/p
            F_tens[12:15,ii] = Eprop[4]*L.gamma[0,:]
            # forward, e/s
            F_tens[15:18,ii] = Eprop[5]*L.gamma[1,:]
            # backward, o/p
            F_tens[18:21,ii] = Eprop[6]*L.gamma[2,:]
            # backward, e/s
            F_tens[21:,ii] = Eprop[7]*L.gamma[3,:]
            
            ### Total electric field (note that sign flip for 
            ### backward propagation is already in gamma)
            # p in 
            E_out[:3,ii] = F_tens[:3,ii]+F_tens[3:6,ii]+F_tens[6:9,ii]+F_tens[9:12,ii]
            # s in
            E_out[3:6,ii] = F_tens[12:15,ii]+F_tens[15:18,ii]+F_tens[18:21,ii]+F_tens[21:,ii]
            
        return z, E_out, zn[:-1] #last interface is useless, substrate=infinite
    
    def calculate_matelem(self, zeta0, f):
        """
        Returns the relevant quantity to find waveguide modes according 
        to Davis' paper on multilayers (scalar model doi.org/10.1016/j.optcom.2008.09.043)
        and then Yeh (4X4 formalism doi.org/10.1016/0039-6028(80)90293-9).
        
        :param 2-tuple zeta0: Tuple [zeta_r, zeta_i] of real and imaginary part of the wavevector
        :param float f: frequency
        :returns: matrix element to minimize for dispersion relation (absolute value)
        """
        self.initialize_sys(f)
        zeta_sys = zeta0[0]+1.0j*zeta0[1]
        self.calculate_GammaStar(f, zeta_sys)
        matelem = self.GammaStar[0,0]*self.GammaStar[2,2]-self.GammaStar[0,2]*self.GammaStar[2,0]
        return np.abs(matelem)
    
    def calculate_eigen_wv(self, zeta0, f):
        """
        Get the eigenmode in-plane wavevector that shows guiding along the plane.
        Based on the idea that guided mode := no input field but output field OK
        :param 2-tuple zeta0: initial guess for the minimization procedure
        :param float f: frequency
        
        :returns: result of the minimization procedure. Eigenvalue is the list res.x
        """
        res = minimize(self.calculate_matelem, zeta0, args=(f), method='Nelder-Mead')
        return res

    def disp_vs_f(self, fv, zeta0):
        """
        Performs a frequency dependent search of the eigenwavevector for a guided mode
        to get the dispersion relation of a surface mode.
        
        Provided a reasonable initial guess for the first frequency point, we 
        use the eigen_wv from the above method and follow its value as a function 
        of frequency in a stepping manner.
        
        :param array fv: array of frequencies
        :param 2-tuple zeta0: initial guess
        
        :returns: array of real part of the in-plane wavevector
        :returns: array of imagniary part of the in-plane wavevector
        """
        zeta_disp_r = np.zeros(len(fv))
        zeta_disp_i = np.zeros(len(fv))
        zeta_disp_r[-1] = zeta0[0]
        zeta_disp_i[-1] = zeta0[1]
        print('Solving for dispersion relation: \n')
        for ii, fi in enumerate(fv):
            print(ii/len(fv))
            zetaguess = [zeta_disp_r[ii-1], zeta_disp_i[ii-1]]
            res = self.calculate_eigen_wv(zetaguess, fi)
            zeta_disp_r[ii] = res.x[0]
            zeta_disp_i[ii] = res.x[1]
        return zeta_disp_r, zeta_disp_i
