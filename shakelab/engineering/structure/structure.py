# ****************************************************************************
#
# Copyright (C) 2019-2026, ShakeLab Developers.
# This file is part of ShakeLab.
#
# ShakeLab is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# ShakeLab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# with this download. If not, see <http://www.gnu.org/licenses/>
#
# ****************************************************************************
"""
Building and half-space models for structural response analysis.
"""

from shakelab.libutils.utils import f0h2vs, zeta2Q


class building_model:
    """Base class for building models."""
    def __init__(self):
        pass


class F0BuildingModel(building_model):
    """Building model based on f0 (fundamental frequency)."""
    def __init__(self, f0, thickness, Q=None, zeta=None, rho=None):
        super().__init__()
        self.f0 = f0
        self.thickness = thickness
        self.vs = f0h2vs(f0, thickness)
        if Q is None and zeta is not None:
            self.Q = zeta2Q(zeta)
        else:
            self.Q = Q
        self.rho = rho


class ElasticParametersBuildingModel(building_model):
    """Building model based on elastic parameters."""
    def __init__(self, parameters):
        super().__init__()
        self.elastic_parameters = parameters


class VsBuildingModel(building_model):
    """Building model based on shear wave velocity (Vs)."""
    def __init__(self, vs, thickness, Q=None, zeta=None, rho=None):
        super().__init__()
        self.vs = vs
        self.thickness = thickness
        if Q is None and zeta is not None:
            self.Q = zeta2Q(zeta)
        else:
            self.Q = Q
        self.rho = rho


class HalfSpaceModel(building_model):
    """Half-space model based on shear wave velocity (Vs)."""
    def __init__(self, vs=300, Q=30, zeta=None, rho=3000, thickness=30):
        super().__init__()
        self.vs = vs
        if zeta is not None:
            self.Q = zeta2Q(zeta)
        else:
            self.Q = Q
        self.rho = rho
        self.thickness = thickness
