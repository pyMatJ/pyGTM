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
# Copyright (C) Mathieu Jeannin 2019-2023 <mathieu.jeannin@c2n.upsaclay.fr> 

"""
This program implements the generalized 4x4 transfer matrix (GTM) method 
poposed in Passler, N. C. and Paarmann, A., JOSA B 34, 2128 (2017) 
doi.org/10.1364/JOSAB.34.002128
and corrected in doi.org/10.1364/JOSAB.36.003246, 
and the layer-resolved absorption proposed in 
https://arxiv.org/abs/2002.03832. 

This code uses inputs from D. Dietze's FSRStools library
https://github.com/ddietze/FSRStools

Please cite the relevant associated publications if you use this code. 

Author: 
    - Mathieu Jeannin mathieu.jeannin@c2n.upsaclay.fr 

Affiliations: 
    - Laboratoire de Physique de l'Ecole Normale Superieure (2019)
    - Centre de Nanosciences et Nanotechnologies (from 2020)
..
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
    
    Copyright (C) Mathieu Jeannin 2019-2023 <mathieu.jeannin@c2n.upsaclay.fr>
    
"""