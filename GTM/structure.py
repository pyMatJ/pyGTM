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

    def initialize_sys(self, f):
        """Sets the values of epsilon at the given frequency in all the layers.

        Parameters
        ----------
        f : float
            Frequency (Hz)
        Returns
        -------
        None

        Notes
        -----
        This function allows to define the in-plane wavevector (:math:`zeta`)
        outside of the class, and thus to explore also guided modes of the
        system.
        """
        self.superstrate.calculate_epsilon(f)
        self.substrate.calculate_epsilon(f)
        for li in self.layers:
            li.calculate_epsilon(f)
