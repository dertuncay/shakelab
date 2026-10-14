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
Structural damage state evaluation based on empirical and engineering metrics.
"""

import json
import os


class DamageState:
    """Class to evaluate structural damage state based on different methods and inputs."""

    def __init__(self, method, pga=None, pgv=None, pgd=None, drift=None,
                 interstory_drift=None, psa03=None, psa10=None, psa30=None,
                 arias=None, housner=None,
                 construction_year=None, construction_material=None,
                 number_of_floor=None, total_height=None):
        self.method = method
        self.pga = pga
        self.pgv = pgv
        self.pgd = pgd
        self.drift = drift
        self.interstory_drift = interstory_drift
        self.psa03 = psa03
        self.psa10 = psa10
        self.psa30 = psa30
        self.arias = arias
        self.housner = housner
        self.construction_year = construction_year
        self.construction_material = construction_material
        self.number_of_floor = number_of_floor
        self.total_height = total_height

        # Load metrics
        self.metrics = self._load_metrics()

    def _load_metrics(self):
        """Loads the JSON metrics file based on the selected method."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "damage_metrics", f"{self.method}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Damage metrics for method '{self.method}' not found at {json_path}")
        with open(json_path, 'r') as f:
            return json.load(f)

    def _get_age_category_scaini(self, m, y):
        if m in ["URM", "URM/MX"]:
            if y < 1946:
                return "<1946"
            elif y <= 1980:
                return "1946-1980"
            else:
                return "1980+"
        elif m == "RC":
            if y < 1980:
                return "<1980"
            else:
                return "1980-2010"
        return None

    def _get_story_key_scaini(self, m, s):
        if m in ["URM", "URM/MX"]:
            return "1-2" if s <= 2 else "3+"
        elif m == "RC":
            if 1 <= s <= 3:
                return "1-3"
            elif 4 <= s <= 7:
                return "4-7"
        return None

    def get_damage_stage(self):
        """Evaluates and returns the damage stage based on the loaded method."""
        if self.method == "scaini_etal":
            if self.construction_material is None or self.construction_year is None or self.number_of_floor is None:
                raise ValueError("scaini_etal method requires construction_material, construction_year, and number_of_floor")
            if self.drift is None:
                raise ValueError("scaini_etal method requires drift")

            material = self.construction_material.upper()
            age_category = self._get_age_category_scaini(material, self.construction_year)
            story_key = self._get_story_key_scaini(material, self.number_of_floor)

            thresholds = self.metrics.get("thresholds", {})
            try:
                extensive, complete = thresholds[material][age_category][story_key]
            except KeyError:
                return "Threshold data not available for given input"

            if self.drift < extensive:
                return "Slight or None"
            elif self.drift < complete:
                return "Extensive"
            else:
                return "Complete"
        else:
            raise NotImplementedError(f"Method '{self.method}' is not implemented.")
