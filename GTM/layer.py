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


import numpy as np
import numpy.linalg as lag
from .permittivities import vacuum_eps

class Layer:
    """
    Layer class. An instance is a single layer:

    Attributes
    -----------
    thickness : float
              thickness of the layer in m
    epsilon1 : complex function
             function epsilon(frequency) for the first axis.
             If none, defaults to vacuum.
    epsilon2 : complex function
             function epsilon(frequency) for the second axis.
             If none, defaults to epsilon1.
    epsilon3 : complex function
             function epsilon(frequency) for the third axis.
             If none, defaults to epsilon1.
    theta : float
          Euler angle theta (colatitude) in rad
    phi : float
        Euler angle phi in rad
    psi : float
        Euler angle psi in rad

    Notes
    -----
    If instanciated with defaults values, it generates a 1um
    thick layer of air.
    Properties can be checked/changed dynamically using the
    corresponding get/set methods.
    """

    def __init__(self, thickness=1.0e-6, epsilon1=None, epsilon2=None,
                 epsilon3=None, theta=0, phi=0, psi=0):

        self.epsilon = np.identity(3, dtype=np.complex128)
        self.mu = 1.0  # mu=1 for now
        # epsilon is a 3x3 matrix of permittivity at a given frequency

        # initialization of all important quantities
        # Boolean to replace Xu's eigenvectors by Berreman's in case of Birefringence
        self.useBerreman = False

        self.euler = np.identity(3, dtype=np.complex128)  # rotation matrix

        self.set_thickness(thickness)  # set the thickness, 1um by default
        self.set_epsilon(epsilon1, epsilon2, epsilon3)  # set epsilon, vacuum by default
        self.set_euler(theta, phi, psi)  # set orientation of crystal axis w/ respect to the lab frame

    def set_thickness(self, thickness):
        """
        Sets the layer thickness

        Parameters
        ----------
        thickness : float
                  the layer thickness (in m)

        Returns
        -------
               None
        """
        self.thick = thickness

    def set_epsilon(self, epsilon1=vacuum_eps, epsilon2=None, epsilon3=None):
        """
        Sets the dielectric functions for the three main axis.

        Parameters
        -----------
        epsilon1 : complex function
                 function epsilon(frequency) for the first axis. If none,
                 defaults to :py:func:`vacuum_eps`
        epsilon2 : complex function
                 function epsilon(frequency) for the second axis. If none,
                 defaults to epsilon1.
        epsilon3 : complex function
                 function epsilon(frequency) for the third axis. If none,
                 defaults to epsilon1.
        func epsilon1: function returning the first (xx) component of the
                complex permittivity tensor in the crystal frame.

        Returns
        -------
               None

        Notes
        ------
        Each *epsilon_i* function returns the dielectric constant along
        axis i as a function of the frequency f in Hz.

        If no function is given for epsilon1, it defaults to
        :py:func:`vacuum_eps` (1.0 everywhere).
        epsilon2 and epsilon3 default to epsilon1: if None, a homogeneous
        material is assumed
        """

        if epsilon1 is None:
            self.epsilon1_f = vacuum_eps
        else:
            self.epsilon1_f = epsilon1

        if epsilon2 is None:
            self.epsilon2_f = self.epsilon1_f
        else:
            self.epsilon2_f = epsilon2
        if epsilon3 is None:
            self.epsilon3_f = self.epsilon1_f
        else:
            self.epsilon3_f = epsilon3

    def calculate_epsilon(self, f):
        """
        Sets the value of epsilon in the (rotated) lab frame.

        Parameters
        ----------
        f : float
            frequency (in Hz)
        Returns
        -------
            None

        Notes
        ------
        The values are set according to the epsilon_fi (i=1..3) functions
        defined using the :py:func:`set_epsilon` method, at the given
        frequency f. The rotation with respect to the lab frame is computed
        using the Euler angles.

        Use only explicitely if you *don't* use the :py:func:`Layer.update`
        function!
        """
        epsilon_xstal = np.zeros((3, 3), dtype=np.complex128)
        epsilon_xstal[0, 0] = self.epsilon1_f(f)
        epsilon_xstal[1, 1] = self.epsilon2_f(f)
        epsilon_xstal[2, 2] = self.epsilon3_f(f)
        self.epsilon = np.matmul(lag.inv(self.euler),
                                 np.matmul(epsilon_xstal, self.euler))
        return self.epsilon.copy()

    def set_euler(self, theta, phi, psi):
        """
        Sets the values for the Euler rotations angles.

        Parameters
        ----------
        theta : float
              Euler angle theta (colatitude) in rad
        phi : float
            Euler angle phi in rad
        psi : float
            Euler angle psi in rad

        Returns
        -------
            None
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
