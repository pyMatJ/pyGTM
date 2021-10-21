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
from .layer import Layer


class Structure:
    """
    System class. An instance is an optical system with substrate,
    superstrate and layers.

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

    """
    def __init__(self, substrate=None, superstrate=None, layers=[]):

        self.layers = []
        if len(layers) > 0:
            self.layers = layers

        if substrate is not None:
            self.substrate = substrate
        else:
            self.substrate = Layer()  # should default to 1µm of vacuum
        if superstrate is not None:
            self.superstrate = superstrate
        else:
            self.superstrate = Layer()  # should default to 1µm of vacuum

    def set_substrate(self, sub):
        """Sets the substrate

        Parameters
        ----------
        sub : Layer
            Instance of the layer class, substrate
        Returns
        -------
        None
        """
        self.substrate = sub

    def set_superstrate(self, sup):
        """Set the superstrate

        Parameters
        ----------
        sup : Layer 
            Instance of the layer class, superstrate
        Returns
        -------
        None
        """
        self.superstrate=sup

    def get_all_layers(self):
        """Returns the list of all layers in the system

        Returns
        -------
        l : list
            list of all layers
        """
        return self.layers

    def get_layer(self, pos):
        """Get the layer at a given position

        Parameters
        ----------
        pos : int
            position in the stack

        Returns
        -------
        L : Layer
            The layer at the position `pos`
        """
        return self.layers[pos]

    def get_superstrate(self):
        """Returns the System's superstrate

        Returns
        -------
        L : Layer
            The system superstrate
        """
        return self.superstrate

    def get_substrate(self):
        """Returns the System's substrate

        Returns
        -------
        L : Layer
            The system substrate
        """
        return self.substrate

    def add_layer(self, layer):
        """Add a layer instance.

        Parameters
        -----------
        layer : Layer
                The layer to be added on the stack

        Returns
        -------
        None

        Notes
        -----
        The layers are added *from superstrate to substrate* order.
        Light is incident *from the superstrate*.

        Note thate this function adds a reference to L to the list.
        If you are adding the same layer several times, be aware that if you
        change something for one of them, it changes all of them.
        """
        self.layers.append(layer)

    def del_layer(self, pos):
        """Remove a layer at given position. Does nothing for invalid position.

        Parameters
        ----------
        pos : int
            Index of layer to be removed.
        Returns
        -------
        None
        """
        if pos >= 0 and pos < len(self.layers):
            self.layers.pop(pos)
        else:
            print('Wrong position given. No layer deleted')

    def get_layers_boundaries(self):
        """
        Return the z-position of all boundaries, including the "top" of the
        superstrate and the "bottom" of the substrate. This corresponds to where
        the fields should be evaluated to get a minimum of information

        Returns
        -------
        zn : 1Darray
            Array of layer boundary positions
        """

        # Nb of layers
        laynum = len(self.layers)
        zn = np.zeros(laynum+3)  # superstrate+layers+substrate
        zn[0] = -self.superstrate.thick
        zn[1] = 0
        for ii, li in enumerate(self.layers):
            zn[ii+2] = zn[ii+1]+li.thick
        zn[-1] = zn[-2]+self.substrate.thick
        return np.array(zn)

    def get_spatial_permittivity(self, z):
        """
        Extract the permittivity tensor at given z in the structure

        Parameters
        ----------
        z : 1Darray
            Array of points to sample the permittivity
        Returns
        -------
        eps : 3x3xlen(z)-array
            Complex permittivity tensor as a function of z
        """
        laynum = len(self.layers)
        zn = np.zeros(laynum+2)  # superstrate+layers+substrate
        zn[-1] = 0.0  # initially with the substrate
        if laynum > 0:
            zn[-2] = zn[-1]-self.substrate.thick
            for kl in range(1, laynum)[::-1]:
                # subtract the thickness (building thickness array backwards)
                zn[kl] = zn[kl+1]-self.layers[kl].thick
            zn[0] = zn[1]-self.layers[0].thick
        else:
            zn[0] = -self.substrate.thick
        zn = zn-zn[0]
        # starting from the superstrate:
        current_layer = 0
        L = self.superstrate
        eps = np.ones((3, 3, len(z)), dtype=np.complex128)
        for ii, zc in enumerate(z):  # enumerates returns a tuple (index, value)
            if zc > zn[current_layer]:
                # change the layer
                # important to count here until laynum+1 to get the correct zn
                # in the substrate for dKiz

                current_layer += 1
                if current_layer == laynum+1:  # reached substrate
                    L = self.substrate
                else:
                    L = self.layers[current_layer-1]
            eps[:, :, ii] = L.epsilon
        return eps
